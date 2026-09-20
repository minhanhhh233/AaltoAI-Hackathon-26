"""
Audit Log
---------
Append-only, structured record of every external model call this
pipeline makes — what was sent, to which model, and why — plus any
other system decision worth explaining alongside its evidence.

Layers 0-3's decisions are already fully auditable: every threshold,
z-score, and p-value they use is deterministic and already saved in
full by utils/results_exporter.py. This module exists for the part
that ISN'T already covered that way: Layer 4's LLM calls are external,
non-deterministic, and cost real money — the prompt sent and the
reasoning behind each response need their own explicit record so a
hypothesis can be traced back to exactly what evidence produced it.
`log_decision` is also available for any other component (present or
future) that makes a judgment call worth recording the same way.

One JSON object per line (JSONL): cheap to append without rewriting
the file, easy to stream or grep, one record per call.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Appends structured audit records to a JSONL file.

    Not thread-safe by design — pipeline runs are single-process, and
    append-only writes keep the common case simple. Wrap with a lock
    if used from multiple threads/processes against the same file.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _append(self, record: Dict[str, Any]):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_llm_call(
        self,
        task: str,
        model: str,
        purpose: str,
        system_prompt: str,
        user_content: str,
        response: str,
        parsed_result: Optional[Any] = None,
    ):
        """
        Record one external model call.

        Parameters
        ----------
        task : str
            Which Layer 4 method made the call (e.g. "root_cause_analysis").
        model : str
            Exact model name used.
        purpose : str
            Short human-readable reason for this specific call (e.g.
            "Root cause analysis for run 99").
        system_prompt, user_content : str
            Exactly what was sent — the full ground rules/task
            instructions and the evidence bundle.
        response : str
            The raw text returned by the model.
        parsed_result : optional
            The parsed/structured result derived from `response`, if
            parsing succeeded (or the parse-error payload if not).
        """
        self._append({
            "type": "llm_call",
            "task": task,
            "model": model,
            "purpose": purpose,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "response": response,
            "parsed_result": parsed_result,
        })

    def log_decision(
        self,
        component: str,
        decision: str,
        evidence: Any,
        reasoning: Optional[str] = None,
    ):
        """
        Record a system decision and the evidence behind it — for
        anything worth explaining after the fact, LLM-derived or not.
        E.g. component="layer4_llm.root_cause_analysis",
        decision="ranked flow1 as the top root-cause candidate",
        evidence={"onset_sample": 100, "causal_edge": "flow1->temp1"}.
        """
        self._append({
            "type": "decision",
            "component": component,
            "decision": decision,
            "evidence": evidence,
            "reasoning": reasoning,
        })

    def read_all(self) -> List[Dict[str, Any]]:
        """Load every record in order (for review or a UI to display)."""
        if not os.path.exists(self.filepath):
            return []
        records = []
        with open(self.filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
