/**
 * AWIS Widget — Live Rejection Risk Panel
 * ========================================
 * Injects a fixed side-panel that listens to form changes,
 * validates, and calls POST http://127.0.0.1:8000/predict in real time.
 *
 * Requires: form elements annotated with data-awis="<field_name>"
 * No external dependencies.
 */
(function () {
  "use strict";

  /* =========================================================================
   * CONSTANTS
   * ======================================================================= */
  const API_BASE    = "http://127.0.0.1:8000";
  const API_PREDICT = API_BASE + "/predict";
  const API_HEALTH  = API_BASE + "/health";
  const DEBOUNCE_MS = 600;
  const CIRCUMFERENCE = 2 * Math.PI * 72; // SVG circle r=72

  /* =========================================================================
   * FIELD SCHEMA  — single source of truth, mirrors main.py ApplicationInput
   *
   * type     : "int"    → parseInt(value, 10)   sent as JSON number
   *            "float"  → parseFloat(value)      sent as JSON number
   *            "binary" → checkbox → 0 or 1      sent as JSON number
   *
   * required : if true, the SELECT/INPUT must have a non-empty chosen value
   *            before any prediction is attempted (idle state otherwise)
   *
   * min/max  : client-side range check before sending (matches Pydantic constraints)
   * ======================================================================= */
  const FIELD_SCHEMA = [
    // ── Categorical dropdowns ──────────────────────────────────────────────
    { name: "source_enc",        type: "int",    min: 0,    max: null, required: true  },
    { name: "permit_type_enc",   type: "int",    min: 0,    max: null, required: true  },
    { name: "zone_enc",          type: "int",    min: 0,    max: null, required: true  },
    { name: "city_enc",          type: "int",    min: 0,    max: null, required: true  },
    // ── Numeric inputs ─────────────────────────────────────────────────────
    { name: "area_sqm",          type: "float",  min: 0,    max: null, required: false },
    { name: "construction_cost", type: "float",  min: 0,    max: null, required: false },
    { name: "doc_count",         type: "int",    min: 0,    max: null, required: false },
    { name: "missing_doc_count", type: "int",    min: 0,    max: null, required: false },
    // ── Binary document flags (checkboxes → 0/1) ──────────────────────────
    { name: "has_title_deed",    type: "binary", min: 0,    max: 1,    required: false },
    { name: "has_site_plan",     type: "binary", min: 0,    max: 1,    required: false },
    { name: "has_noc_fire",      type: "binary", min: 0,    max: 1,    required: false },
    { name: "has_noc_env",       type: "binary", min: 0,    max: 1,    required: false },
    { name: "has_struct_cert",   type: "binary", min: 0,    max: 1,    required: false },
    { name: "has_crz_cert",      type: "binary", min: 0,    max: 1,    required: false },
    { name: "pan_valid",         type: "binary", min: 0,    max: 1,    required: false },
    // ── Compliance flags (checkboxes → 0/1) ──────────────────────────────
    { name: "zone_type_conflict",type: "binary", min: 0,    max: 1,    required: false },
    { name: "area_exceeds_fsi",  type: "binary", min: 0,    max: 1,    required: false },
    // ── Score & time ──────────────────────────────────────────────────────
    { name: "risk_score",        type: "int",    min: 0,    max: 100,  required: false },
    { name: "year",              type: "int",    min: 1990, max: 2100, required: true  },
    { name: "month",             type: "int",    min: 1,    max: 12,   required: true  },
  ];

  // Human-readable labels for each field (used in error messages & feature list)
  const FIELD_LABELS = {
    source_enc:         "Application Source",
    permit_type_enc:    "Permit Type",
    zone_enc:           "Zone",
    city_enc:           "City",
    area_sqm:           "Area (sqm)",
    construction_cost:  "Construction Cost",
    doc_count:          "Docs Submitted",
    missing_doc_count:  "Missing Docs",
    has_title_deed:     "Title Deed",
    has_site_plan:      "Site Plan",
    has_noc_fire:       "Fire NOC",
    has_noc_env:        "Env NOC",
    has_struct_cert:    "Structural Cert",
    has_crz_cert:       "CRZ Certificate",
    pan_valid:          "PAN Valid",
    zone_type_conflict: "Zone Conflict",
    area_exceeds_fsi:   "Area Exceeds FSI",
    risk_score:         "Risk Score",
    year:               "Year",
    month:              "Month",
  };

  /* =========================================================================
   * STYLES
   * ======================================================================= */
  const CSS = `
    #awis-panel {
      position:fixed; top:0; right:0; width:360px; height:100vh;
      background:#0f172a; color:#e2e8f0;
      font-family:'Inter',sans-serif; font-size:13px;
      display:flex; flex-direction:column;
      box-shadow:-4px 0 32px rgba(0,0,0,.45);
      z-index:9999; overflow:hidden;
    }
    #awis-panel * { box-sizing:border-box; margin:0; padding:0; }

    /* header */
    #awis-ph {
      background:linear-gradient(135deg,#4f46e5,#7c3aed);
      padding:16px 20px; flex-shrink:0;
    }
    #awis-ph h2 { font-size:14px; font-weight:700; color:#fff; letter-spacing:.3px; }
    #awis-ph p  { font-size:11px; color:rgba(255,255,255,.65); margin-top:2px; }
    #awis-status-row { display:flex; align-items:center; gap:6px; margin-top:8px; font-size:11px; }
    #awis-dot { width:8px; height:8px; border-radius:50%; background:#ef4444; flex-shrink:0; transition:background .4s; }
    #awis-dot.on { background:#22c55e; }

    /* scrollable body */
    #awis-body { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }
    #awis-body::-webkit-scrollbar { width:4px; }
    #awis-body::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }

    /* sections */
    #awis-idle {
      background:#1e293b; border-radius:12px; padding:20px;
      text-align:center; color:#475569; font-size:12px; line-height:1.7;
    }
    #awis-idle span { font-size:30px; display:block; margin-bottom:8px; }

    #awis-loading {
      display:none; flex-direction:column; align-items:center;
      justify-content:center; gap:10px; padding:24px;
    }
    .awis-spin {
      width:28px; height:28px; border:3px solid #1e293b;
      border-top-color:#6366f1; border-radius:50%;
      animation:_awspin .7s linear infinite;
    }
    @keyframes _awspin { to { transform:rotate(360deg); } }

    #awis-error {
      display:none; background:#450a0a; border:1px solid #7f1d1d;
      border-radius:10px; padding:12px; font-size:12px; color:#fca5a5; line-height:1.6;
    }
    #awis-error code { color:#fda4af; font-size:11px; }

    /* result */
    #awis-result { display:none; flex-direction:column; gap:14px; }

    /* gauge card */
    #awis-gc {
      background:#1e293b; border-radius:14px;
      padding:20px 16px 14px; display:flex; flex-direction:column; align-items:center;
    }
    #awis-gc svg { overflow:visible; }
    #awis-arc {
      stroke-linecap:round;
      transform-origin:center; transform:rotate(-90deg);
      transition:stroke-dashoffset .7s cubic-bezier(.4,0,.2,1);
    }
    #awis-snum {
      font-size:38px; font-weight:800; fill:#f8fafc;
      dominant-baseline:middle; text-anchor:middle;
      transition:fill .4s;
    }
    #awis-slbl { font-size:10px; fill:#64748b; text-anchor:middle; }

    /* badge */
    #awis-badge {
      display:inline-flex; align-items:center; gap:6px;
      padding:5px 14px; border-radius:999px; font-size:13px; font-weight:700;
      margin-top:10px; background:#334155; color:#94a3b8;
      transition:background .4s,color .4s;
    }
    #awis-badge.approved { background:#14532d; color:#4ade80; }
    #awis-badge.rejected { background:#450a0a; color:#f87171; }
    #awis-bdot { width:7px; height:7px; border-radius:50%; background:currentColor; }

    /* confidence */
    #awis-conf-row { display:flex; justify-content:space-between; font-size:11px; color:#64748b; margin-top:6px; width:100%; }
    #awis-cbg { height:4px; background:#0f172a; border-radius:4px; width:100%; margin-top:4px; overflow:hidden; }
    #awis-cfill { height:100%; width:0%; border-radius:4px; transition:width .6s ease; }

    /* features card */
    #awis-fc { background:#1e293b; border-radius:14px; padding:14px; }
    #awis-fc h3 {
      font-size:11px; font-weight:600; color:#94a3b8;
      text-transform:uppercase; letter-spacing:.6px; margin-bottom:10px;
    }
    .aw-fr { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #0f172a; }
    .aw-fr:last-child { border-bottom:none; }
    .aw-fi { font-size:13px; flex-shrink:0; width:18px; text-align:center; }
    .aw-fn { flex:1; color:#cbd5e1; font-size:12px; }
    .aw-fb { width:70px; height:5px; background:#0f172a; border-radius:4px; overflow:hidden; }
    .aw-fbi { height:100%; border-radius:4px; transition:width .5s ease; }
    .aw-fv { font-size:11px; color:#64748b; width:40px; text-align:right; flex-shrink:0; }

    /* validation warning */
    #awis-warn {
      display:none; background:#1c1917; border:1px solid #44403c;
      border-radius:10px; padding:12px; font-size:12px; color:#fbbf24; line-height:1.6;
    }
  `;

  /* =========================================================================
   * PANEL HTML
   * ======================================================================= */
  const PANEL_HTML = `
    <div id="awis-ph">
      <h2>&#x1F916; AI Risk Assessment</h2>
      <p>Adaptive Workflow Intervention System</p>
      <div id="awis-status-row">
        <div id="awis-dot"></div>
        <span id="awis-stxt">Connecting&hellip;</span>
      </div>
    </div>

    <div id="awis-body">
      <div id="awis-idle">
        <span>&#x1F4CB;</span>
        Select <strong>Application Source</strong>, <strong>Year</strong>
        and <strong>Month</strong> to start live predictions.
      </div>

      <div id="awis-loading">
        <div class="awis-spin"></div>
        <span style="color:#64748b;font-size:12px">Analysing&hellip;</span>
      </div>

      <div id="awis-warn"></div>
      <div id="awis-error"></div>

      <div id="awis-result">
        <div id="awis-gc">
          <svg width="160" height="160" viewBox="0 0 160 160">
            <defs>
              <linearGradient id="awisG" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stop-color="#22c55e"/>
                <stop offset="50%"  stop-color="#f59e0b"/>
                <stop offset="100%" stop-color="#ef4444"/>
              </linearGradient>
            </defs>
            <circle cx="80" cy="80" r="72" fill="none" stroke="#0f172a" stroke-width="12"/>
            <circle id="awis-arc" cx="80" cy="80" r="72" fill="none"
              stroke="url(#awisG)" stroke-width="12"
              stroke-dasharray="${CIRCUMFERENCE.toFixed(2)}"
              stroke-dashoffset="${CIRCUMFERENCE.toFixed(2)}"/>
            <text id="awis-snum" x="80" y="78">--</text>
            <text id="awis-slbl" x="80" y="100">RISK SCORE</text>
          </svg>
          <div id="awis-badge"><div id="awis-bdot"></div><span id="awis-btxt">--</span></div>
          <div id="awis-conf-row"><span>Confidence</span><span id="awis-cpct">--</span></div>
          <div id="awis-cbg"><div id="awis-cfill"></div></div>
        </div>

        <div id="awis-fc">
          <h3>&#x25B2; Top Contributing Factors</h3>
          <div id="awis-fl"></div>
        </div>
      </div>
    </div>
  `;

  /* =========================================================================
   * UTILITIES
   * ======================================================================= */
  function debounce(fn, ms) {
    let t;
    return function (...a) { clearTimeout(t); t = setTimeout(() => fn.apply(this, a), ms); };
  }

  function riskColor(score) {
    if (score < 35) return "#22c55e";
    if (score < 65) return "#f59e0b";
    return "#ef4444";
  }

  function show(ids, which) {
    // ids = { idle, loading, warn, error, result }
    Object.entries(ids).forEach(([k, el]) => {
      const isResult = k === "result";
      el.style.display = (k === which)
        ? (isResult ? "flex" : "block")
        : "none";
    });
  }

  /* =========================================================================
   * INJECT PANEL
   * ======================================================================= */
  function inject() {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const panel = document.createElement("div");
    panel.id = "awis-panel";
    panel.innerHTML = PANEL_HTML;
    document.body.appendChild(panel);

    document.body.style.marginRight = "360px";
  }

  /* =========================================================================
   * DOM REFS  (called after inject)
   * ======================================================================= */
  function getRefs() {
    return {
      dot:    document.getElementById("awis-dot"),
      stxt:   document.getElementById("awis-stxt"),
      // section containers
      secs: {
        idle:    document.getElementById("awis-idle"),
        loading: document.getElementById("awis-loading"),
        warn:    document.getElementById("awis-warn"),
        error:   document.getElementById("awis-error"),
        result:  document.getElementById("awis-result"),
      },
      // gauge
      arc:    document.getElementById("awis-arc"),
      snum:   document.getElementById("awis-snum"),
      // badge
      badge:  document.getElementById("awis-badge"),
      btxt:   document.getElementById("awis-btxt"),
      // confidence
      cpct:   document.getElementById("awis-cpct"),
      cfill:  document.getElementById("awis-cfill"),
      // features
      fl:     document.getElementById("awis-fl"),
      // error/warn text targets
      errEl:  document.getElementById("awis-error"),
      warnEl: document.getElementById("awis-warn"),
    };
  }

  /* =========================================================================
   * COLLECT & CAST PAYLOAD
   *
   * Rules (exactly matching main.py ApplicationInput):
   *   binary  → checkbox.checked → 0 | 1        (JSON integer)
   *   int     → parseInt(value, 10)              (JSON integer)
   *   float   → parseFloat(value)               (JSON float)
   *
   * Missing/empty non-required fields default to 0 (API accepts ge=0).
   * Required fields that are still empty are flagged so we stay idle.
   * ======================================================================= */
  function collectPayload() {
    const payload = {};
    const missingRequired = [];

    FIELD_SCHEMA.forEach((spec) => {
      const el = document.querySelector(`[data-awis="${spec.name}"]`);

      // ── binary (checkbox) ────────────────────────────────────────────────
      if (spec.type === "binary") {
        if (!el) { payload[spec.name] = 0; return; }
        payload[spec.name] = el.type === "checkbox"
          ? (el.checked ? 1 : 0)
          : Math.max(0, Math.min(1, parseInt(el.value, 10) || 0));
        return;
      }

      // ── select / number input ────────────────────────────────────────────
      const raw = el ? el.value.trim() : "";

      if (raw === "") {
        // Field has NOT been filled by the user yet
        if (spec.required) missingRequired.push(spec.name);
        payload[spec.name] = spec.type === "float" ? 0.0 : 0;
        return;
      }

      // Cast to the correct numeric type
      const num = spec.type === "float"
        ? parseFloat(raw)
        : parseInt(raw, 10);

      if (isNaN(num)) {
        if (spec.required) missingRequired.push(spec.name);
        payload[spec.name] = spec.type === "float" ? 0.0 : 0;
        return;
      }

      payload[spec.name] = num;
    });

    return { payload, missingRequired };
  }

  /* =========================================================================
   * CLIENT-SIDE VALIDATION  (range checks only — type safety is guaranteed
   * by collectPayload, and the model_validator in main.py is a safety net)
   * ======================================================================= */
  function validate(payload) {
    const errors = [];

    FIELD_SCHEMA.forEach((spec) => {
      const v = payload[spec.name];
      if (v === undefined || v === null || (typeof v === "number" && isNaN(v))) return;
      if (spec.min !== null && v < spec.min)
        errors.push(`<b>${FIELD_LABELS[spec.name]}</b>: must be &ge; ${spec.min} (sent <code>${v}</code>)`);
      if (spec.max !== null && v > spec.max)
        errors.push(`<b>${FIELD_LABELS[spec.name]}</b>: must be &le; ${spec.max} (sent <code>${v}</code>)`);
    });

    return errors;
  }

  /* =========================================================================
   * RENDER RESULT
   * ======================================================================= */
  function renderResult(r, data) {
    const score  = data.rejection_risk_score;
    const color  = riskColor(score);
    const offset = CIRCUMFERENCE * (1 - score / 100);

    // gauge arc
    r.arc.style.strokeDashoffset = offset.toFixed(2);
    r.snum.textContent = Math.round(score);
    r.snum.style.fill  = color;

    // badge
    const rejected = data.prediction === "Rejected";
    r.badge.className = rejected ? "rejected" : "approved";
    r.btxt.textContent = data.prediction;

    // confidence bar
    const conf = Math.round(data.confidence * 100);
    r.cpct.textContent    = conf + "%";
    r.cfill.style.width   = conf + "%";
    r.cfill.style.background = color;

    // top contributing features
    const feats = data.top_contributing_features || [];
    const maxAbs = Math.max(...feats.map((f) => Math.abs(f.shap_value)), 0.0001);

    r.fl.innerHTML = feats.map((f) => {
      const pct   = Math.round((Math.abs(f.shap_value) / maxAbs) * 100);
      const up    = f.impact === "increases_risk";
      const clr   = up ? "#ef4444" : "#22c55e";
      const icon  = up ? "&#x25B2;" : "&#x25BC;";
      const label = FIELD_LABELS[f.feature] || f.feature;
      return `
        <div class="aw-fr">
          <span class="aw-fi" style="color:${clr}">${icon}</span>
          <span class="aw-fn">${label}</span>
          <div class="aw-fb"><div class="aw-fbi" style="width:${pct}%;background:${clr}"></div></div>
          <span class="aw-fv">${f.shap_value.toFixed(3)}</span>
        </div>`;
    }).join("");

    show(r.secs, "result");
  }

  /* =========================================================================
   * API CALL
   * ======================================================================= */
  async function callPredict(payload, r) {
    // 1. Client-side range validation
    const errs = validate(payload);
    if (errs.length) {
      r.warnEl.innerHTML =
        `<strong>&#x26A0; Fix before submitting:</strong><ul style="margin:6px 0 0 14px;line-height:1.9">` +
        errs.map((e) => `<li>${e}</li>`).join("") + `</ul>`;
      show(r.secs, "warn");
      return;
    }

    show(r.secs, "loading");

    // 2. Debug log — open browser DevTools Console to inspect
    console.group("[AWIS] /predict payload");
    console.table(
      FIELD_SCHEMA.map((s) => ({
        field:       s.name,
        sent_value:  payload[s.name],
        js_type:     typeof payload[s.name],
        schema_type: s.type,
        ok:          typeof payload[s.name] === "number" && !isNaN(payload[s.name]),
      }))
    );
    console.groupEnd();

    // 3. Fetch
    let res, body;
    try {
      res  = await fetch(API_PREDICT, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      body = await res.json().catch(() => null);
    } catch (netErr) {
      r.errEl.innerHTML =
        `<strong>&#x26A0; Cannot reach API</strong><br>` +
        `<span style="color:#94a3b8">Start the server:<br>` +
        `<code>uvicorn main:app --reload --port 8000</code></span>`;
      show(r.secs, "error");
      console.error("[AWIS] Network error:", netErr);
      return;
    }

    // 4. Handle HTTP errors
    if (!res.ok) {
      if (res.status === 422 && body && Array.isArray(body.detail)) {
        // Our custom handler returns structured field-level errors
        const rows = body.detail
          .map((e) => `<li><code>${e.field}</code>: ${e.error} &mdash; got <em>${e.received}</em></li>`)
          .join("");
        r.errEl.innerHTML =
          `<strong>&#x26A0; Server rejected request (422)</strong>` +
          `<ul style="margin:6px 0 0 14px;line-height:1.9">${rows}</ul>` +
          `<div style="margin-top:8px;font-size:11px;color:#94a3b8">` +
          `Tip: <code>GET /schema</code> shows the exact required format.</div>`;
      } else {
        const msg = (body && (body.detail || body.error)) || `HTTP ${res.status}`;
        r.errEl.innerHTML = `<strong>&#x26A0; API error ${res.status}</strong><br>${msg}`;
      }
      show(r.secs, "error");
      console.error("[AWIS] API error", res.status, body);
      return;
    }

    // 5. Render
    renderResult(r, body);
  }

  /* =========================================================================
   * HEALTH CHECK
   * ======================================================================= */
  async function checkHealth(r) {
    try {
      const res  = await fetch(API_HEALTH);
      const data = await res.json();
      if (data.status === "ok") {
        r.dot.className = "on";
        r.stxt.textContent = "Model connected & ready";
      } else {
        r.stxt.textContent = "Model not loaded";
      }
    } catch {
      r.dot.className = "";
      r.stxt.textContent = "API offline \u2014 start uvicorn";
    }
  }

  /* =========================================================================
   * INIT
   * ======================================================================= */
  function init() {
    inject();
    const r = getRefs();
    checkHealth(r);

    // ── form change handler ──────────────────────────────────────────────
    const onFormChange = debounce(() => {
      const { payload, missingRequired } = collectPayload();

      // Stay idle until the minimum required fields have been chosen
      if (missingRequired.length > 0) {
        show(r.secs, "idle");
        return;
      }

      callPredict(payload, r);
    }, DEBOUNCE_MS);

    // ── attach listeners to all [data-awis] elements ─────────────────────
    function attachListeners() {
      document.querySelectorAll("[data-awis]").forEach((el) => {
        el.removeEventListener("input",  onFormChange);
        el.removeEventListener("change", onFormChange);
        // checkboxes fire "change"; selects and number inputs fire "input"
        el.addEventListener(el.type === "checkbox" ? "change" : "input", onFormChange);
      });
    }

    attachListeners();

    // Re-attach if the form DOM changes dynamically
    new MutationObserver(attachListeners)
      .observe(document.body, { childList: true, subtree: true });

    // ── submit button enablement ─────────────────────────────────────────
    const checkSubmit = debounce(() => {
      const btn = document.getElementById("awis-submit-btn");
      if (!btn) return;
      const { missingRequired } = collectPayload();
      btn.disabled = missingRequired.length > 0;
    }, 150);

    document.addEventListener("input",  checkSubmit);
    document.addEventListener("change", checkSubmit);
  }

  // Bootstrap
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();
