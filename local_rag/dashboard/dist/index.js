(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;
  const React = SDK.React;
  const h = React.createElement;
  const { useState, useEffect, useCallback, useRef } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button, Input, Label, Select, SelectOption } = SDK.components;
  const API = "/api/plugins/local_rag";
  const TERMS_URL = "https://huggingface.co/google/embeddinggemma-300m";
  const TOKEN_URL = "https://huggingface.co/settings/tokens/new?tokenType=read";

  function errorText(error) {
    const raw = String(error && error.message ? error.message : error || "Unknown error");
    const match = raw.match(/^\d{3}:\s*(.*)$/s);
    try { const body = JSON.parse(match ? match[1] : raw); return body.detail || raw; } catch (_) { return match ? match[1] : raw; }
  }

  function Check(props) {
    return h("li", { className: "local-rag-check" },
      h("span", { className: "local-rag-check-dot " + (props.ok ? "is-ok" : "is-pending"), "aria-hidden": true }, props.ok ? "✓" : "·"),
      h("span", null, props.label),
      props.note ? h("span", { className: "local-rag-check-note" }, props.note) : null
    );
  }

  function RadioChoice(props) {
    return h("label", { className: "local-rag-radio-choice" + (props.checked ? " is-selected" : "") },
      h("input", { type: "radio", name: props.name, value: props.value, checked: props.checked, onChange: props.onChange }),
      h("span", { className: "local-rag-radio-content" },
        h("span", { className: "local-rag-choice-title" }, props.title),
        props.copy ? h("span", { className: "local-rag-choice-copy" }, props.copy) : null
      )
    );
  }

  function Page() {
    const [status, setStatus] = useState(null);
    const [busy, setBusy] = useState(null);
    const [job, setJob] = useState(null);
    const [message, setMessage] = useState(null);
    const [mode, setMode] = useState("text");
    const [dimensions, setDimensions] = useState("512");
    const [retention, setRetention] = useState("forever");
    const [days, setDays] = useState("90");
    const [backfillPlan, setBackfillPlan] = useState(null);
    const [acceptedBackfill, setAcceptedBackfill] = useState([]);

    const [acceptedTerms, setAcceptedTerms] = useState(false);
    const [hfToken, setHfToken] = useState("");
    const initialized = useRef(false);

    const refresh = useCallback(function () {
      return SDK.fetchJSON(API + "/setup/status").then(function (data) {
        setStatus(data);
        if (!initialized.current) {
          initialized.current = true;
          const cfg = data.config || {};
          setDimensions(String(cfg.embedding_dimensions || 512));
          if (cfg.episodic_ttl_days == null && cfg.summary_ttl_days == null) setRetention("forever");
          else { setRetention("custom"); setDays(String(cfg.episodic_ttl_days || cfg.summary_ttl_days || 90)); }
          if (cfg.visual_enabled) setMode("visual");
        }
        return data;
      }).catch(function (e) { setMessage({ ok: false, text: "Could not load setup status: " + errorText(e) }); });
    }, []);

    const loadBackfillPlan = useCallback(function () {
      return SDK.fetchJSON(API + "/setup/backfill/plan").then(function (data) {
        setBackfillPlan(data);
        setAcceptedBackfill((data.items || []).map(function (item, index) { return item.accepted ? index : -1; }).filter(function (index) { return index >= 0; }));
      }).catch(function () { setBackfillPlan(null); setAcceptedBackfill([]); });
    }, []);

    useEffect(function () { refresh(); }, [refresh]);
    useEffect(function () {
      if (!job || job.state !== "running") return undefined;
      const timer = window.setInterval(function () {
        SDK.fetchJSON(API + "/setup/progress?job_id=" + encodeURIComponent(job.id)).then(function (next) {
          setJob(next);
          if (next.state !== "running") {
            setBusy(null);
            setMessage({ ok: next.state === "complete", text: next.state === "complete" ? "Operation completed successfully." : next.detail });
            refresh();
            if (next.state === "complete" && next.kind === "backfill-preview") loadBackfillPlan();
          }
        }).catch(function (e) { setBusy(null); setMessage({ ok: false, text: errorText(e) }); });
      }, 1500);
      return function () { window.clearInterval(timer); };
    }, [job && job.id, job && job.state, refresh, loadBackfillPlan]);

    function action(label, path, options) {
      setBusy(label); setMessage(null);
      return SDK.fetchJSON(API + path, options || { method: "POST" }).then(function (result) {
        if (result.job_id) setJob({ id: result.job_id, state: "running", detail: "Starting…", log: "" });
        else { setBusy(null); setMessage({ ok: true, text: label + " completed." }); refresh(); }
      }).catch(function (e) { setBusy(null); setMessage({ ok: false, text: errorText(e) }); });
    }

    function saveConfiguration() {
      const ttl = retention === "custom" ? Number(days) : null;
      const payload = { embedding_dimensions: Number(dimensions), episodic_ttl_days: ttl, summary_ttl_days: ttl, visual_enabled: mode === "visual" };
      return action("Save configuration", "/setup/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    }

    function signIn() {
      if (!hfToken.trim()) return;
      setBusy("Hugging Face sign-in"); setMessage(null);
      return SDK.fetchJSON(API + "/setup/auth", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: hfToken.trim() })
      }).then(function () {
        setHfToken(""); setBusy(null);
        setMessage({ ok: true, text: "Hugging Face sign-in completed." });
        refresh();
      }).catch(function (e) {
        setHfToken(""); setBusy(null);
        setMessage({ ok: false, text: errorText(e) });
      });
    }

    function previewBackfill() {
      return action("Create backfill preview", "/setup/backfill/preview", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true })
      });
    }

    function toggleBackfill(index) {
      setAcceptedBackfill(function (current) {
        return current.indexOf(index) >= 0 ? current.filter(function (value) { return value !== index; }) : current.concat([index]);
      });
    }

    function editBackfill(index, text) {
      setBackfillPlan(function (current) {
        return { ...current, items: current.items.map(function (item, itemIndex) {
          return itemIndex === index ? { ...item, text: text } : item;
        }) };
      });
    }

    function persistBackfillReview() {
      return SDK.fetchJSON(API + "/setup/backfill/plan", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
          revision: backfillPlan.revision,
          accepted_indices: acceptedBackfill,
          edits: Object.fromEntries(backfillPlan.items.map(function (item, index) { return [index, item.text]; }))
        })
      }).then(function (result) {
        setBackfillPlan(function (current) { return { ...current, revision: result.revision }; });
        return result;
      });
    }

    function saveBackfillReview() {
      setBusy("Save backfill review"); setMessage(null);
      return persistBackfillReview().then(function () {
        setBusy(null); setMessage({ ok: true, text: "Backfill review saved." });
      }).catch(function (e) { setBusy(null); setMessage({ ok: false, text: errorText(e) }); });
    }

    function applyBackfill() {
      setBusy("Apply reviewed memories"); setMessage(null);
      return persistBackfillReview().then(function (review) {
        return SDK.fetchJSON(API + "/setup/backfill/apply", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true, revision: review.revision })
        });
      }).then(function (result) {
        setJob({ id: result.job_id, state: "running", detail: "Starting…", log: "" });
      }).catch(function (e) {
        setBusy(null); setMessage({ ok: false, text: errorText(e) });
      });
    }

    if (!status) return h("div", { className: "local-rag-page" },
      h("p", { className: "text-sm text-muted-foreground" }, message ? message.text : "Checking Local RAG setup…"),
      message && h(Button, { variant: "outline", onClick: refresh }, "Retry")
    );
    const downloading = busy === "Download models";
    const configured = status.config && Number(status.config.embedding_dimensions) === Number(dimensions) && Boolean(status.config.visual_enabled) === (mode === "visual");

    return h("div", { className: "local-rag-page" },
      h("div", { className: "local-rag-hero" },
        h("div", null, h("h1", null, "Hermes Local RAG"), h("p", null, "Private, on-device text and optional visual memory. Complete each step, then run the health check.")),
        h(Button, { variant: "outline", onClick: refresh, disabled: !!busy }, "Refresh status")
      ),
      message ? h("div", { className: "local-rag-message " + (message.ok ? "is-ok" : "is-error"), role: "status" }, message.text) : null,
      h("div", { className: "local-rag-layout" },
        h(Card, { className: "local-rag-status" },
          h(CardHeader, null, h(CardTitle, null, "Setup status")),
          h(CardContent, null,
            h("ul", null,
              h(Check, { ok: status.hermes_found && status.python_found, label: "Hermes environment", note: status.python_found ? "Found" : "Not found" }),
              h(Check, { ok: status.dependencies_installed, label: "Runtime dependencies" }),
              h(Check, { ok: status.hf && status.hf.authenticated, label: "Hugging Face sign-in", note: status.hf && !status.hf.available ? "CLI unavailable" : null }),
              h(Check, { ok: status.text_model_downloaded, label: "Text embedding model" }),
              h(Check, { ok: mode === "text" || status.visual_model_downloaded, label: "Visual model", note: mode === "text" ? "Not selected" : null }),
              h(Check, { ok: configured, label: "Configuration" }),
              h(Check, { ok: status.active, label: "Memory provider active", note: status.provider ? "Current: " + status.provider : null })
            )
          )
        ),
        h("div", { className: "local-rag-steps" },
          h(Card, null, h(CardHeader, null, h("div", { className: "local-rag-step-head" }, h(Badge, null, "1"), h(CardTitle, null, "Access and dependencies"))),
            h(CardContent, { className: "local-rag-stack" },
              h("p", { className: "text-sm text-muted-foreground" }, "EmbeddingGemma is gated. Review and accept its terms on Hugging Face before downloading."),
              h("div", { className: "local-rag-actions" },
                h(Button, { type: "button", variant: "outline", onClick: function () { window.open(TERMS_URL, "_blank", "noopener,noreferrer"); } }, "Open Gemma Terms"),
                h(Button, { type: "button", onClick: function () { action("Install dependencies", "/setup/dependencies"); }, disabled: !!busy || status.dependencies_installed }, status.dependencies_installed ? "Dependencies installed" : "Install dependencies")
              ),
              h("div", { className: "local-rag-guidance" },
                h("strong", null, status.hf.authenticated ? "Hugging Face sign-in detected" : "Sign in to Hugging Face"),
                h("p", null, status.hf.guidance),
                !status.hf.authenticated && status.dependencies_installed ? h("div", { className: "local-rag-actions" },
                  h(Input, { type: "password", value: hfToken, autoComplete: "off", placeholder: "hf_… read token", "aria-label": "Hugging Face token", onChange: function (e) { setHfToken(e.target.value); } }),
                  h(Button, { type: "button", variant: "outline", onClick: function () { window.open(TOKEN_URL, "_blank", "noopener,noreferrer"); } }, "Create read token"),
                  h(Button, { type: "button", onClick: signIn, disabled: !!busy || !hfToken.trim() }, "Sign in")
                ) : null,
                h("label", { className: "local-rag-consent" }, h("input", { type: "checkbox", checked: acceptedTerms, onChange: function (e) { setAcceptedTerms(e.target.checked); } }), h("span", null, "I accepted the Gemma Terms for this model."))
              )
            )
          ),
          h(Card, null, h(CardHeader, null, h("div", { className: "local-rag-step-head" }, h(Badge, null, "2"), h(CardTitle, null, "Choose and download models"))),
            h(CardContent, { className: "local-rag-stack" },
              h("div", { className: "local-rag-choices" },
                h(RadioChoice, { name: "local-rag-model-mode", value: "text", checked: mode === "text", title: "Text only", copy: "EmbeddingGemma for messages, summaries, and files.", onChange: function () { setMode("text"); } }),
                h(RadioChoice, { name: "local-rag-model-mode", value: "visual", checked: mode === "visual", title: "Text + visual", copy: "Adds CLIP image search in a separate vector space.", onChange: function () { setMode("visual"); } })
              ),
              h(Button, { onClick: function () { action("Download models", "/setup/models", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ visual: mode === "visual", terms_accepted: acceptedTerms }) }); }, disabled: !!busy || !acceptedTerms || !status.hf.authenticated || (status.text_model_downloaded && (mode === "text" || status.visual_model_downloaded)) }, downloading ? "Downloading…" : "Download selected models"),
              job && job.kind === "download" ? h("div", { className: "local-rag-progress" },
                h("div", { className: "local-rag-progress-bar", "aria-label": "Download in progress" }, h("span", { className: job.state === "running" ? "is-running" : "" })),
                h("p", null, job.detail), job.log ? h("pre", null, job.log) : null
              ) : null
            )
          ),
          h(Card, null, h(CardHeader, null, h("div", { className: "local-rag-step-head" }, h(Badge, null, "3"), h(CardTitle, null, "Configure memory"))),
            h(CardContent, { className: "local-rag-stack" },
              h("div", { className: "local-rag-field" }, h(Label, null, "Embedding dimensions"),
                h(Select, { value: dimensions, onValueChange: setDimensions, onChange: function (e) { setDimensions(e.target.value); } },
                  ["256", "512", "768"].map(function (v) { return h(SelectOption, { key: v, value: v }, v + (v === "512" ? " (recommended)" : "")); }))
              ),
              h("div", { className: "local-rag-field" }, h(Label, null, "Retention"),
                h("div", { className: "local-rag-radio-row" },
                  h(RadioChoice, { name: "local-rag-retention", value: "forever", checked: retention === "forever", title: "Forever", onChange: function () { setRetention("forever"); } }),
                  h(RadioChoice, { name: "local-rag-retention", value: "custom", checked: retention === "custom", title: "Custom", onChange: function () { setRetention("custom"); } })
                ),
                retention === "custom" ? h(Input, { type: "number", min: "1", value: days, onChange: function (e) { setDays(e.target.value); }, "aria-label": "Retention days", placeholder: "Days" }) : null
              ),
              h("p", { className: "text-xs text-muted-foreground" }, "Changing dimensions after indexing requires reindexing; incompatible vectors are not mixed."),
              h(Button, { onClick: saveConfiguration, disabled: !!busy || (retention === "custom" && !(Number(days) > 0)) }, "Save configuration")
            )
          ),
          h(Card, null, h(CardHeader, null, h("div", { className: "local-rag-step-head" }, h(Badge, null, "4"), h(CardTitle, null, "Optional session backfill"))),
            h(CardContent, { className: "local-rag-stack" },
              h("p", { className: "text-sm text-muted-foreground" }, "Hermes extracts normalized reusable facts into a review plan. Raw transcripts are deleted after extraction and never written to memory."),
              h("div", { className: "local-rag-actions" },
                h(Button, { onClick: previewBackfill, disabled: !!busy }, "Create selective preview"),
                h(Button, { variant: "outline", onClick: loadBackfillPlan, disabled: !!busy }, "Load existing preview")
              ),
              backfillPlan ? h("div", { className: "local-rag-review" },
                h("div", { className: "local-rag-review-head" },
                  h("strong", null, String(backfillPlan.items.length) + " candidates"),
                  h("span", null, String(acceptedBackfill.length) + " accepted")
                ),
                backfillPlan.items.length ? h("div", { className: "local-rag-candidates" }, backfillPlan.items.map(function (item, index) {
                  const checked = acceptedBackfill.indexOf(index) >= 0;
                  return h("div", { className: "local-rag-candidate" + (checked ? " is-selected" : ""), key: index },
                    h("input", { type: "checkbox", checked: checked, onChange: function () { toggleBackfill(index); }, "aria-label": "Accept candidate " + (index + 1) }),
                    h("span", null,
                      h("textarea", { className: "local-rag-candidate-text", value: item.text, rows: 3, onChange: function (event) { editBackfill(index, event.target.value); } }),
                      h("span", { className: "local-rag-candidate-meta" }, item.scope + " · " + item.subject + " · confidence " + Math.round(item.confidence * 100) + "%")
                    )
                  );
                })) : h("p", { className: "text-sm text-muted-foreground" }, "No reusable memories were found."),
                h("div", { className: "local-rag-actions" },
                  h(Button, { variant: "outline", onClick: saveBackfillReview, disabled: !!busy }, "Save review"),
                  h(Button, { onClick: applyBackfill, disabled: !!busy || acceptedBackfill.length === 0 }, "Apply accepted memories")
                )
              ) : null,
              job && (job.kind === "backfill-preview" || job.kind === "backfill-apply") ? h("div", { className: "local-rag-progress" },
                h("div", { className: "local-rag-progress-bar" }, h("span", { className: job.state === "running" ? "is-running" : "" })),
                h("p", null, job.detail)
              ) : null
            )
          ),
          h(Card, null, h(CardHeader, null, h("div", { className: "local-rag-step-head" }, h(Badge, null, "5"), h(CardTitle, null, "Activate and verify"))),
            h(CardContent, { className: "local-rag-stack" },
              h("p", { className: "text-sm text-muted-foreground" }, "Activation updates memory.provider. Restart the gateway or start a new Desktop session afterward."),
              h("div", { className: "local-rag-actions" },
                h(Button, { disabled: !!busy || status.active, onClick: function () { action("Activate Local RAG", "/setup/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) }); } }, status.active ? "Local RAG active" : "Activate Local RAG"),
                h(Button, { variant: "outline", disabled: !!busy, onClick: function () { action("Health check", "/health"); } }, "Run health check")
              )
            )
          )
        )
      )
    );
  }

  window.__HERMES_PLUGINS__.register("local_rag", Page);
})();
