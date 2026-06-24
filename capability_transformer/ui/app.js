"use strict";
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);
const el = (t, a = {}, kids = []) => {
  const e = document.createElement(t);
  for (const k in a) k === "html" ? (e.innerHTML = a[k]) : e.setAttribute(k, a[k]);
  for (const c of [].concat(kids)) e.append(c);
  return e;
};
const svg = (t, a = {}) => { const e = document.createElementNS(SVGNS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };

// ---- geometry ------------------------------------------------------------------------
const COLW = 13, ROWH = 30, LABELW = 178, ARCW = 76, TOPPAD = 126, LEFTPAD = 10;
const GROUP_BASE = { features: "#e9ecf6", policy: "#f1eafb", evidence: "#ebeafb" };
const GROUP_ON = { features: "#3b6cff", policy: "#a860ff", evidence: "#6d49fd" };
const GROUP_TINT = { features: "#3b6cff", policy: "#a860ff", evidence: "#6d49fd" };

const state = { trace: null, step: 0, playing: false, timer: null, cells: [], colOf: {} };

// ---- data load -----------------------------------------------------------------------
async function loadPresets() {
  const presets = await (await fetch("/trace/examples")).json();
  const box = $("#presets");
  box.innerHTML = "";
  presets.forEach((p) => {
    const tag = p.id.startsWith("allow") || p.id === "confirmed_send" || p.id === "delegation" ? "ALLOW"
      : p.id.startsWith("escalate") ? "ESCALATE" : "DENY";
    const btn = el("button", { class: "preset" }, [
      el("div", {}, [el("span", { class: "p-title", html: p.title.split("·")[0].trim() }),
        el("span", { class: `p-tag p-${tag}`, html: tag })]),
      el("div", { class: "p-desc", html: p.desc }),
    ]);
    btn.onclick = () => {
      document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("#bundle-json").value = JSON.stringify(p.bundle, null, 2);
      $("#require_signatures").checked = !!p.config.require_signatures;
      $("#require_bound").checked = !!p.config.require_bound_confirmations;
      run();
    };
    box.append(btn);
  });
}

async function run() {
  $("#error").textContent = "";
  let bundle;
  try { bundle = JSON.parse($("#bundle-json").value); }
  catch (e) { $("#error").textContent = "Invalid JSON: " + e.message; return; }
  const body = {
    bundle,
    require_signatures: $("#require_signatures").checked,
    require_bound_confirmations: $("#require_bound").checked,
  };
  const res = await fetch("/trace", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) { $("#error").textContent = "Trace failed: " + (await res.text()).slice(0, 400); return; }
  state.trace = await res.json();
  buildViz();
  setStep(0);
}

// ---- build the static SVG (cells, labels, group bands, legend, decision) --------------
function buildViz() {
  const t = state.trace, cols = t.layout.columns, rows = t.tokens;
  state.colOf = {};
  cols.forEach((c) => { if (c.label === c.name) state.colOf[c.name] = c.col; });

  const W = LEFTPAD + LABELW + ARCW + cols.length * COLW + 16;
  const H = TOPPAD + rows.length * ROWH + 16;
  const root = $("#viz");
  root.setAttribute("width", W); root.setAttribute("height", H);
  root.innerHTML = "";
  defsArrow(root);

  // group bands + per-column rotated labels
  const x0 = LEFTPAD + LABELW + ARCW;
  let g = root.appendChild(svg("g"));
  let runStart = 0;
  const rowsH = rows.length * ROWH;
  cols.forEach((c, i) => {
    const x = x0 + i * COLW;
    const lab = svg("text", { x: x + COLW / 2, y: TOPPAD - 7, transform: `rotate(-52 ${x + COLW / 2} ${TOPPAD - 7})`,
      "font-size": 9, "font-weight": 500,
      fill: c.group === "evidence" ? "#6d49fd" : c.group === "policy" ? "#8b54cf" : "#51526b",
      "font-family": "Geist Mono, monospace", "text-anchor": "start" });
    lab.textContent = c.label; g.append(lab);
    const next = cols[i + 1];
    if (!next || next.group !== c.group) {
      const bx = x0 + runStart * COLW, bw = (i - runStart + 1) * COLW;
      // faint full-height band behind the cells
      g.append(svg("rect", { x: bx, y: TOPPAD - 2, width: bw, height: rowsH + 4, rx: 6,
        fill: GROUP_TINT[c.group], opacity: 0.05 }));
      const t2 = svg("text", { x: bx + 4, y: TOPPAD - 96, "font-size": 10.5, "font-weight": 700,
        "letter-spacing": "0.12em", fill: GROUP_TINT[c.group] });
      t2.textContent = c.group.toUpperCase(); g.append(t2);
      g.append(svg("rect", { x: bx, y: TOPPAD - 90, width: bw, height: 84, rx: 6, fill: "none",
        stroke: GROUP_TINT[c.group], "stroke-opacity": 0.3, "stroke-dasharray": "2 4" }));
      runStart = i + 1;
    }
  });

  // token rows: left labels + heatmap cells
  state.cells = [];
  rows.forEach((tok, r) => {
    const y = TOPPAD + r * ROWH;
    const label = svg("g");
    const lr = svg("text", { x: LEFTPAD, y: y + 14, "font-size": 12.5, fill: "#090910", "font-weight": 600 });
    lr.textContent = roleLabel(tok); label.append(lr);
    const sub = svg("text", { x: LEFTPAD, y: y + 25, "font-size": 9.5, fill: "#5c5d66",
      "font-family": "Geist Mono, monospace" });
    sub.textContent = subLabel(tok); label.append(sub);
    label.style.cursor = "default";
    label.onmousemove = (ev) => tip(ev, tokenTip(tok));
    label.onmouseleave = hideTip;
    root.append(label);

    const cellRow = [];
    cols.forEach((c, i) => {
      const cell = svg("rect", { x: x0 + i * COLW, y: y + 2, width: COLW - 1.5, height: ROWH - 5,
        rx: 1.5, fill: GROUP_BASE[c.group] });
      cell.onmousemove = (ev) => tip(ev, `<b>${roleLabel(tok)}</b> · ${c.label}<br>value = <b>${val(r, i)}</b>`);
      cell.onmouseleave = hideTip;
      root.append(cell); cellRow.push(cell);
    });
    state.cells.push(cellRow);
  });

  state.arcLayer = root.appendChild(svg("g", { id: "arcs" }));

  // slider + layer pipeline
  $("#slider").max = t.steps.length - 1;
  const pipe = $("#pipeline"); pipe.innerHTML = "";
  state.layerOrder = t.layer_schedule.map((L) => L.id);
  t.layer_schedule.forEach((L, i) => {
    const st = el("div", { class: "stage", "data-layer": L.id, title: L.desc }, [
      el("span", { class: "s-num", html: "L" + i }),
      el("span", { class: "s-name", html: L.name })]);
    st.onclick = () => jumpToLayer(L.id);
    pipe.append(st);
  });

  // decision banner
  const db = $("#decision-banner");
  db.className = "decision " + t.decision;
  $("#decision-text").textContent = t.decision;
  $("#reasons").textContent = (t.reasons || []).join(", ");
  const rb = $("#ref-badge");
  rb.className = "badge " + (t.matches_reference ? "ok" : "bad");
  rb.textContent = t.matches_reference ? "✓ matches reference" : "✗ differs from reference";
}

// ---- per-step rendering --------------------------------------------------------------
function setStep(i) {
  const t = state.trace; if (!t) return;
  state.step = Math.max(0, Math.min(i, t.steps.length - 1));
  const step = t.steps[state.step];
  $("#slider").value = state.step;
  $("#step-label").textContent = `step ${state.step}/${t.steps.length - 1} · ${step.op}`;

  // recolor heatmap from this step's snapshot
  const snap = step.snapshot, cols = t.layout.columns;
  for (let r = 0; r < snap.length; r++)
    for (let c = 0; c < cols.length; c++) {
      const v = snap[r][c];
      const cell = state.cells[r][c];
      cell.setAttribute("fill", v > 0.5 ? GROUP_ON[cols[c].group] : GROUP_BASE[cols[c].group]);
      cell.setAttribute("stroke", "none");
      cell.setAttribute("filter", "none");
    }
  // highlight changed cells (what this step just wrote)
  (step.changed || []).forEach(([r, c]) => {
    const cell = state.cells[r][c];
    cell.setAttribute("stroke", "#ff6a00");
    cell.setAttribute("stroke-width", "2");
    cell.setAttribute("filter", "drop-shadow(0 0 4px rgba(255,106,0,.6))");
  });

  drawArcs(step);
  renderDetail(step);
  renderCapEvidence(snap);

  // layer pipeline: done / active / pending + progress fill
  const curIdx = state.layerOrder.indexOf(step.layer);
  document.querySelectorAll("#pipeline .stage").forEach((s) => {
    const i = state.layerOrder.indexOf(s.dataset.layer);
    s.classList.toggle("active", i === curIdx);
    s.classList.toggle("done", i < curIdx);
  });
  $("#progress").style.width = (state.step / Math.max(1, t.steps.length - 1)) * 100 + "%";
}

function drawArcs(step) {
  const layer = state.arcLayer; layer.innerHTML = "";
  const yMid = (r) => TOPPAD + r * ROWH + ROWH / 2;
  const arcR = LEFTPAD + LABELW + ARCW - 6, arcL = LEFTPAD + LABELW + 10;
  if (step.kind === "match") {
    step.detail.matches.forEach((m) => {
      const y1 = yMid(m.query), y2 = yMid(m.key);
      const on = m.value > 0.5;
      const p = svg("path", { d: `M ${arcR} ${y1} C ${arcL} ${y1}, ${arcL} ${y2}, ${arcR} ${y2}`,
        fill: "none", stroke: on ? "#a860ff" : "#c9cbd8", "stroke-width": on ? 2.2 : 1,
        "marker-end": "url(#arrow)", opacity: on ? 1 : 0.7 });
      layer.append(p);
      const tx = svg("text", { x: arcL - 3, y: (y1 + y2) / 2 + 3, "font-size": 10, "text-anchor": "end",
        "font-family": "Geist Mono, monospace", "font-weight": 600,
        fill: on ? "#7c4fe0" : "#9092a0" });
      tx.textContent = m.score; layer.append(tx);
    });
  } else if (step.kind === "pool") {
    const inC = state.colOf[step.in_slot], outC = state.colOf[step.out_slot];
    const x0 = LEFTPAD + LABELW + ARCW;
    const outY = yMid(state.trace.tokens.findIndex((t) => t.role === "output"));
    const outX = x0 + outC * COLW + COLW / 2;
    step.detail.members.forEach((m, k) => {
      const on = m.value > 0.5;
      const x1 = x0 + inC * COLW + COLW / 2, y1 = yMid(m.token);
      const p = svg("path", { d: `M ${x1} ${y1} Q ${x1 - 40} ${(y1 + outY) / 2} ${outX} ${outY}`,
        fill: "none", stroke: on ? "#36d399" : "#2a3550", "stroke-width": on ? 1.8 : 0.8,
        opacity: on ? 0.9 : 0.4 });
      layer.append(p);
    });
  }
}

function renderDetail(step) {
  $("#op-title").textContent = step.title;
  $("#op-desc").textContent = step.desc;
  const m = $("#op-math"); m.innerHTML = "";
  const tok = (i) => roleLabel(state.trace.tokens[i]);
  const bit = (v) => `<span class="${v > 0.5 ? "bit1" : "bit0"}">${v}</span>`;

  if (step.kind === "match") {
    const link = el("span", { class: "head-link", html: `inspect Q/K matrices →` });
    link.onclick = () => showHead(step.head);
    m.append(el("div", { class: "row", html: `head <b>${step.head}</b> &nbsp; ` }));
    m.lastChild.append(link);
    step.detail.matches.forEach((d) => {
      m.append(el("div", { class: "row", html:
        `${tok(d.query)} <span class="hl">·</span> ${tok(d.key)} &nbsp; Q·K = <b>${d.score}</b> → ${bit(d.value)}` }));
    });
  } else if (step.kind === "gate") {
    step.detail.tokens.forEach((row) => {
      const ins = row.inputs.map((x) => `${x.name}=${bit(x.value)}`).join(`  <span class="hl">${step.gate_op.toUpperCase()}</span>  `);
      m.append(el("div", { class: "row", html: `${tok(row.token)}: ${ins} = <b>${bit(row.output)}</b>` }));
    });
  } else if (step.kind === "pool") {
    const parts = step.detail.members.map((x) => `${tok(x.token)}=${bit(x.value)}`).join(", ");
    m.append(el("div", { class: "row", html: `<span class="hl">∃</span> max( ${parts || "—"} ) = <b>${bit(step.detail.max_value)}</b>` }));
    m.append(el("div", { class: "row", html: `→ written to <b>${roleLabel(state.trace.tokens[step.out_token])}</b>.${step.out_slot}` }));
  } else if (step.kind === "output") {
    const L = step.detail.logits;
    m.append(el("div", { class: "row", html:
      `logits: ALLOW=<b>${L.ALLOW}</b>  DENY=<b>${L.DENY}</b>  ESCALATE=<b>${L.ESCALATE}</b>` }));
    m.append(el("div", { class: "row", html: `argmax → <b>${step.detail.decision}</b> (margin ${step.detail.margin})` }));
  } else {
    m.append(el("div", { class: "row", html: step.desc }));
  }

  const lp = $("#logits-panel");
  if (step.kind === "output") { lp.hidden = false; renderLogits(step.detail); } else lp.hidden = true;
}

function renderLogits(detail) {
  const box = $("#logits-bars"); box.innerHTML = "";
  const max = Math.max(1, ...Object.values(detail.logits));
  const colors = { ALLOW: "#36d399", DENY: "#f87272", ESCALATE: "#fbbd23" };
  detail.classes.forEach((c) => {
    const v = detail.logits[c];
    const row = el("div", { class: "logit" + (c === detail.decision ? " win" : "") }, [
      el("span", { class: "lname", html: c }),
      el("div", { class: "bar" }, [el("span", { style: `width:${(v / max) * 100}%;background:${colors[c]}` })]),
      el("span", { class: "lval", html: String(v) }),
    ]);
    box.append(row);
  });
}

function renderCapEvidence(snap) {
  const box = $("#cap-evidence"); box.innerHTML = "";
  const t = state.trace;
  const slots = ["subject_match", "object_match", "right_match", "issuer_trusted",
    "not_revoked", "chain_ok", "atten_ok"];
  t.tokens.forEach((tok, r) => {
    if (tok.role !== "capability") return;
    const valid = snap[r][state.colOf["valid_capability"]] > 0.5;
    const pills = slots.map((s) => {
      const v = snap[r][state.colOf[s]] > 0.5;
      return el("span", { class: "pill " + (v ? "on" : "off"), html: s.replace("_match", "").replace("_", " ") });
    });
    pills.push(el("span", { class: "pill " + (valid ? "valid" : "off"), html: "valid" }));
    box.append(el("div", { class: "cap-row" + (valid ? " valid" : "") }, [
      el("div", { class: "c-id", html: `${roleLabel(tok)} <span class="mono">· ${subLabel(tok) || ""}</span>` }),
      el("div", { class: "c-bits" }, pills)]));
  });
  if (!box.children.length) box.append(el("div", { class: "op-desc", html: "no capability tokens" }));
}

// ---- matrix inspector ----------------------------------------------------------------
async function showHead(name) {
  const m = await (await fetch("/model/head/" + name)).json();
  $("#modal-title").textContent = `head · ${name}  (${m.query_set} → ${m.key_token})`;
  const body = $("#modal-body"); body.innerHTML = "";
  body.append(matrixView("Wq (query projection)", m.Wq));
  body.append(matrixView("Wk (key projection)", m.Wk));
  $("#modal").hidden = false;
}
function matrixView(title, M) {
  const cols = state.trace.layout.columns;
  const wrap = el("div", {}, [el("h4", { html: title })]);
  const s = svg("svg", { width: Math.min(cols.length, M[0].length) * 6 + 60, height: M.length * 16 + 10 });
  M.forEach((row, r) => {
    row.forEach((v, c) => {
      if (Math.abs(v) < 1e-9) return;
      const rect = svg("rect", { x: 60 + c * 6, y: r * 16, width: 5, height: 14, rx: 1,
        fill: v > 0 ? "#6d49fd" : "#ef4444", opacity: Math.min(1, Math.abs(v)) });
      const lab = cols[c] ? cols[c].label : c;
      rect.append(svg("title")); rect.lastChild.textContent = `${lab} = ${v}`;
      s.append(rect);
    });
    const rl = svg("text", { x: 0, y: r * 16 + 11, "font-size": 9, fill: "#5c5d66" });
    rl.textContent = "row " + r; s.append(rl);
  });
  wrap.append(s);
  return wrap;
}

// ---- helpers -------------------------------------------------------------------------
function roleLabel(tok) {
  if (tok.role === "capability") return "cap " + (capId(tok) || tok.index);
  if (tok.role === "confirmation") return "conf";
  return tok.role;
}
function capId(tok) { return null; }
function subLabel(tok) {
  const f = tok.fields || {};
  if (tok.role === "request") return `${f.subject || "?"} · ${(f["rights/action"] || []).join(",")} · ${f.object || "?"} · ${f.provenance || ""}`;
  if (tok.role === "capability") return `${f.subject || "?"} · ${f.object || "?"} · [${(f["rights/action"] || []).join(",")}]`;
  if (tok.role === "confirmation") return `${f.subject || ""}·${f.object || ""}·${(f["rights/action"] || []).join(",")}`;
  if (tok.role === "policy") return "fixed masks";
  if (tok.role === "output") return "aggregator";
  return "";
}
function tokenTip(tok) {
  const f = tok.fields || {};
  let s = `<b>${roleLabel(tok)}</b> (${tok.type || "—"})`;
  for (const k in f) if (k !== "bits") s += `<br>${k}: ${Array.isArray(f[k]) ? f[k].join(", ") : f[k]}`;
  if (f.bits) s += `<br>bits: ${Object.keys(f.bits).join(", ")}`;
  return s;
}
function val(r, c) { return state.trace.steps[state.step].snapshot[r][c]; }
function tip(ev, html) { const t = $("#tooltip"); t.hidden = false; t.innerHTML = html; t.style.left = ev.clientX + 12 + "px"; t.style.top = ev.clientY + 12 + "px"; }
function hideTip() { $("#tooltip").hidden = true; }
function defsArrow(root) {
  const defs = svg("defs");
  const m = svg("marker", { id: "arrow", markerWidth: 7, markerHeight: 7, refX: 5, refY: 3, orient: "auto" });
  m.append(svg("path", { d: "M0,0 L6,3 L0,6 Z", fill: "#a860ff" }));
  defs.append(m); root.append(defs);
}
function jumpToLayer(id) {
  const i = state.trace.steps.findIndex((s) => s.layer === id);
  if (i >= 0) setStep(i);
}

// ---- scrubber ------------------------------------------------------------------------
function togglePlay() {
  state.playing = !state.playing;
  $("#play").textContent = state.playing ? "⏸" : "▶";
  if (state.playing) tick(); else clearTimeout(state.timer);
}
function tick() {
  if (!state.playing) return;
  if (state.step >= state.trace.steps.length - 1) { togglePlay(); return; }
  setStep(state.step + 1);
  state.timer = setTimeout(tick, +$("#speed").value);
}

function wire() {
  $("#run-btn").onclick = run;
  $("#first").onclick = () => setStep(0);
  $("#prev").onclick = () => setStep(state.step - 1);
  $("#next").onclick = () => setStep(state.step + 1);
  $("#last").onclick = () => setStep(state.trace.steps.length - 1);
  $("#play").onclick = togglePlay;
  $("#slider").oninput = (e) => { if (state.playing) togglePlay(); setStep(+e.target.value); };
  $("#modal-close").onclick = () => ($("#modal").hidden = true);
  $("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").hidden = true; };
  document.onkeydown = (e) => {
    if (!state.trace) return;
    if (e.key === "ArrowRight") setStep(state.step + 1);
    else if (e.key === "ArrowLeft") setStep(state.step - 1);
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  };
}

wire();
loadPresets().then(() => document.querySelector(".preset")?.click());
