// TPM web interface — progress polling, chat widget, upload spinner.
// Deliberately plain JS, no framework: this app is server-rendered,
// JS only handles the few genuinely dynamic bits.

(function () {
  "use strict";

  // ---------------- Build progress polling ----------------

  const buildForm = document.getElementById("build-form");
  if (buildForm) {
    const dataset = buildForm.dataset.dataset;
    const btn = document.getElementById("build-submit");
    const progressWrap = document.getElementById("build-progress");
    const progressFill = document.getElementById("progress-fill");
    const progressMessage = document.getElementById("progress-message");
    const errorBox = document.getElementById("build-error");

    const STEP_LABELS = {
      starting: "Starting…",
      loading: "Loading dataset",
      profiling: "Layer 1 — statistical profiling",
      correlating: "Layer 2 — pooled correlations",
      clustering: "Layer 2 — clustering variables",
      causal_discovery: "Layer 2 — causal discovery (PCMCI, slowest step)",
      llm_interpretation: "Layer 4 — LLM interpretation",
      reporting: "Writing report",
      plotting: "Rendering figures",
      done: "Done",
    };

    function poll(jobId) {
      fetch(`/api/jobs/${jobId}`)
        .then((r) => r.json())
        .then((job) => {
          if (job.error) {
            errorBox.hidden = false;
            errorBox.textContent = job.error;
            btn.disabled = false;
            return;
          }
          const label = STEP_LABELS[job.step] || job.step;
          const pct = job.total > 0 ? Math.round((job.current / job.total) * 90) + 5 : 12;
          progressFill.style.width = Math.min(pct, 97) + "%";
          progressMessage.textContent =
            job.message ? `${label} — ${job.message}` : label;

          if (job.status === "done") {
            progressFill.style.width = "100%";
            progressMessage.textContent = "Done — loading dashboard…";
            setTimeout(() => window.location.reload(), 600);
          } else if (job.status === "error") {
            errorBox.hidden = false;
            errorBox.textContent = job.error || "Build failed.";
            btn.disabled = false;
          } else {
            setTimeout(() => poll(jobId), 1200);
          }
        })
        .catch(() => setTimeout(() => poll(jobId), 2000));
    }

    buildForm.addEventListener("submit", (e) => {
      e.preventDefault();
      btn.disabled = true;
      errorBox.hidden = true;
      progressWrap.hidden = false;

      const nRuns = parseInt(document.getElementById("n_runs").value, 10) || 3;
      const includeCausal = document.getElementById("include_causal").checked;

      fetch(`/api/datasets/${dataset}/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_runs: nRuns, include_causal: includeCausal }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.job_id) {
            poll(data.job_id);
          } else {
            errorBox.hidden = false;
            errorBox.textContent = data.error || "Could not start build.";
            btn.disabled = false;
          }
        })
        .catch((err) => {
          errorBox.hidden = false;
          errorBox.textContent = String(err);
          btn.disabled = false;
        });
    });
  }

  // ---------------- Upload spinner ----------------

  const checkForm = document.getElementById("check-form");
  if (checkForm) {
    checkForm.addEventListener("submit", () => {
      const btn = document.getElementById("check-submit");
      const spinner = document.getElementById("check-spinner");
      if (btn) btn.disabled = true;
      if (spinner) spinner.hidden = false;
    });
  }

  // ---------------- Chat widget ----------------

  const chatWidget = document.getElementById("chat-widget");
  if (chatWidget) {
    const toggleBtn = document.getElementById("chat-toggle");
    const closeBtn = document.getElementById("chat-close");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const messages = document.getElementById("chat-messages");
    let history = [];

    toggleBtn.addEventListener("click", () => chatWidget.classList.add("open"));
    closeBtn.addEventListener("click", () => chatWidget.classList.remove("open"));

    function addMessage(role, text, pending) {
      const div = document.createElement("div");
      div.className = `chat-msg chat-msg-${role}` + (pending ? " chat-msg-pending" : "");
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
      return div;
    }

    function addRuleProposal(rule) {
      const card = document.createElement("div");
      card.className = "rule-proposal-card";

      const desc = document.createElement("div");
      desc.className = "rule-proposal-desc";
      desc.textContent = rule.description;
      card.appendChild(desc);

      const actions = document.createElement("div");
      actions.className = "rule-proposal-actions";

      const confirmBtn = document.createElement("button");
      confirmBtn.type = "button";
      confirmBtn.className = "btn btn-small";
      confirmBtn.textContent = "Confirm & save";

      const rejectBtn = document.createElement("button");
      rejectBtn.type = "button";
      rejectBtn.className = "btn btn-small btn-ghost";
      rejectBtn.textContent = "Not right";

      confirmBtn.addEventListener("click", () => {
        confirmBtn.disabled = true;
        rejectBtn.disabled = true;
        fetch(`/api/datasets/${window.TPM_DATASET}/quality-rules/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rule }),
        })
          .then((r) => r.json())
          .then((data) => {
            card.remove();
            if (data.saved) {
              addMessage("assistant", "Rule saved — it'll apply to future checks. Reload the dashboard to see it on that variable.");
            } else {
              addMessage("assistant", data.error || "Couldn't save that rule.");
            }
          })
          .catch((err) => {
            card.remove();
            addMessage("assistant", "Error saving rule: " + err);
          });
      });

      rejectBtn.addEventListener("click", () => card.remove());

      actions.appendChild(confirmBtn);
      actions.appendChild(rejectBtn);
      card.appendChild(actions);
      messages.appendChild(card);
      messages.scrollTop = messages.scrollHeight;
    }

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = "";
      addMessage("user", message);
      const pending = addMessage("assistant", "Thinking…", true);

      fetch(`/api/datasets/${window.TPM_DATASET}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history }),
      })
        .then((r) => r.json())
        .then((data) => {
          pending.remove();
          if (data.reply) {
            addMessage("assistant", data.reply);
            history = data.history || history;
            if (data.pending_rule) addRuleProposal(data.pending_rule);
          } else {
            addMessage("assistant", data.error || "Something went wrong.");
          }
        })
        .catch((err) => {
          pending.remove();
          addMessage("assistant", "Error: " + err);
        });
    });
  }

  // ---------------- "Explain this" figure buttons ----------------

  document.querySelectorAll(".explain-btn").forEach((btn) => {
    const row = btn.closest(".explain-row");
    const output = row ? row.querySelector(".explain-output") : null;
    if (!output) return;

    btn.addEventListener("click", () => {
      const figureType = btn.dataset.figureType;
      const target = btn.dataset.target || null;
      const originalLabel = btn.textContent;

      btn.disabled = true;
      btn.textContent = "Explaining…";
      output.hidden = true;

      fetch(`/api/datasets/${window.TPM_DATASET}/explain-figure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ figure_type: figureType, target }),
      })
        .then((r) => r.json())
        .then((data) => {
          btn.disabled = false;
          btn.textContent = originalLabel;
          output.textContent = data.explanation || data.error || "Could not generate an explanation.";
          output.hidden = false;
        })
        .catch((err) => {
          btn.disabled = false;
          btn.textContent = originalLabel;
          output.textContent = "Error: " + err;
          output.hidden = false;
        });
    });
  });
})();
