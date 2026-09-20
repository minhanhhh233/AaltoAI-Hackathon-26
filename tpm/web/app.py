"""
TPM Web Interface — FastAPI App
---------------------------------
Run from the project root (one level above tpm/):

    python -m uvicorn tpm.web.app:app --reload

Pages:
    GET  /                              dataset picker
    GET  /dataset/{name}                baseline dashboard (build form if
                                         not yet built; full dashboard once ready)
    GET  /dataset/{name}/check          upload form to check a new run
    POST /dataset/{name}/check          submit an upload, render results
    POST /dataset/{name}/quality-rules/delete   delete a saved rule, redirect to dashboard
    GET  /dataset/{name}/flags          data-quality flags raised so far, for human review

API (used by static/js/app.js for progress polling, chat, and the build
trigger — everything else is server-rendered):
    POST /api/datasets/{name}/build     start the baseline build (background job)
    GET  /api/jobs/{job_id}             poll job status
    POST /api/datasets/{name}/chat      one grounded chat turn (may include a
                                         pending_rule proposal — see layer4_llm.py)
    POST /api/datasets/{name}/quality-rules/confirm  save a chat-proposed rule
    POST /api/datasets/{name}/explain-figure  plain-language explanation of one
                                         dashboard chart ("Explain this" buttons)
"""

import os
import shutil
import tempfile
import traceback
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tpm.layer4_llm import LLMAnalyst
from tpm.utils.llm_client import get_default_client

from . import jobs, pipeline_service
from .datasets import DATASETS, get_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
ANALYSIS_OUTPUT_ABS = os.path.join(PROJECT_ROOT, pipeline_service.ANALYSIS_OUTPUT_DIR)
os.makedirs(ANALYSIS_OUTPUT_ABS, exist_ok=True)

app = FastAPI(title="Anomoly McAnomolyface")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _static_version(rel_path: str) -> int:
    """
    File mtime as a cache-busting query param (`?v=...`) for static CSS/JS
    — this app has no build step/bundler to hash filenames, and static
    assets have repeatedly been served stale from browser cache across
    this project's iteration (images, then app.js) whenever their content
    changed but the URL didn't. Tying the URL to mtime forces a fresh
    fetch exactly when the file actually changes, without needing a hard
    refresh every time.
    """
    path = os.path.join(BASE_DIR, "static", rel_path)
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


templates.env.globals["static_version"] = _static_version
app.mount("/figures", StaticFiles(directory=ANALYSIS_OUTPUT_ABS), name="figures")


class BuildRequest(BaseModel):
    n_runs: int = 3
    include_causal: bool = True


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class RuleConfirmRequest(BaseModel):
    rule: Dict


class ExplainRequest(BaseModel):
    figure_type: str
    target: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    rows = []
    for key, cfg in DATASETS.items():
        build_info = pipeline_service.get_build_info(key)
        rows.append({
            "key": key,
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "ready": pipeline_service.is_baseline_ready(key),
            "build_info": build_info,
        })
    return templates.TemplateResponse(request, "index.html", {"datasets": rows})


@app.get("/dataset/{name}", response_class=HTMLResponse)
def dataset_page(request: Request, name: str):
    try:
        dataset = get_dataset(name)
    except KeyError as exc:
        return HTMLResponse(str(exc), status_code=404)

    ready = pipeline_service.is_baseline_ready(name)
    context = {"name": name, "dataset": dataset, "ready": ready}
    if ready:
        context.update(pipeline_service.load_dashboard_context(name))
    return templates.TemplateResponse(request, "dataset.html", context)


@app.post("/api/datasets/{name}/build")
def build_dataset(name: str, req: BuildRequest):
    try:
        get_dataset(name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    job_id = jobs.start_job(lambda progress: pipeline_service.build_baseline(
        name, n_runs=req.n_runs, include_causal=req.include_causal, progress=progress,
    ))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return {
        "status": job.status,
        "step": job.step,
        "current": job.current,
        "total": job.total,
        "message": job.message,
        "error": job.error,
    }


@app.get("/dataset/{name}/check", response_class=HTMLResponse)
def check_page(request: Request, name: str):
    try:
        dataset = get_dataset(name)
    except KeyError as exc:
        return HTMLResponse(str(exc), status_code=404)

    return templates.TemplateResponse(request, "check.html", {
        "name": name, "dataset": dataset,
        "ready": pipeline_service.is_baseline_ready(name),
        "result": None, "error": None,
    })


@app.post("/dataset/{name}/check", response_class=HTMLResponse)
async def check_submit(request: Request, name: str, file: UploadFile = File(...)):
    try:
        dataset = get_dataset(name)
    except KeyError as exc:
        return HTMLResponse(str(exc), status_code=404)

    tmp_path = os.path.join(tempfile.gettempdir(), f"tpm_upload_{uuid.uuid4().hex}.csv")
    result, error = None, None
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        run_label = f"upload_{uuid.uuid4().hex[:6]}"
        result = pipeline_service.check_run(name, tmp_path, run_label=run_label)
    except Exception as exc:
        traceback.print_exc()
        error = str(exc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return templates.TemplateResponse(request, "check.html", {
        "name": name, "dataset": dataset,
        "ready": pipeline_service.is_baseline_ready(name),
        "result": result, "error": error,
    })


@app.post("/api/datasets/{name}/chat")
def chat_endpoint(name: str, req: ChatRequest):
    try:
        get_dataset(name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    context = pipeline_service.get_report_text(name)
    columns = pipeline_service.get_dataset_columns(name)
    analyst = LLMAnalyst(get_default_client())
    try:
        reply, history, pending_rule = analyst.chat(
            req.message, history=req.history, context=context, columns=columns,
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": f"Could not reach the chat model: {exc}"}, status_code=502)
    return {"reply": reply, "history": history, "pending_rule": pending_rule}


@app.post("/api/datasets/{name}/explain-figure")
def explain_figure_endpoint(name: str, req: ExplainRequest):
    try:
        get_dataset(name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    try:
        result = pipeline_service.explain_figure(name, req.figure_type, req.target)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        # Covers LLM-backend failures (network errors, auth, etc.) — the
        # button's own error handling expects JSON back either way, not
        # FastAPI's default HTML 500 page.
        traceback.print_exc()
        return JSONResponse({"error": f"Could not reach the explanation model: {exc}"}, status_code=502)
    return result


@app.post("/api/datasets/{name}/quality-rules/confirm")
def confirm_quality_rule(name: str, req: RuleConfirmRequest):
    try:
        get_dataset(name)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    try:
        saved = pipeline_service.confirm_quality_rule(name, req.rule)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"saved": True, "rule": saved}


@app.post("/dataset/{name}/quality-rules/delete")
def delete_quality_rule(name: str, rule_id: str = Form(...)):
    try:
        get_dataset(name)
    except KeyError as exc:
        return HTMLResponse(str(exc), status_code=404)

    pipeline_service.delete_quality_rule(name, rule_id)
    return RedirectResponse(f"/dataset/{name}", status_code=303)


@app.get("/dataset/{name}/flags", response_class=HTMLResponse)
def flags_page(request: Request, name: str):
    try:
        dataset = get_dataset(name)
    except KeyError as exc:
        return HTMLResponse(str(exc), status_code=404)

    return templates.TemplateResponse(request, "flags.html", {
        "name": name, "dataset": dataset,
        "flags": pipeline_service.get_flags(name),
    })
