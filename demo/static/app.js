"use strict";
/* Frozen Assistant Studio — vanilla ES2020, no build step.
   Arena: POST /api/stream (SSE over fetch). Explorers: /api/summary, /api/ablation_data, /api/logs, /api/prompts. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : `${Number(v).toFixed(d)}%`);
const fmtBytes = (b) => (b == null ? "–" : b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`);
const showTok = (s) => JSON.stringify(s).slice(1, -1);
const COLS = ["raw", "caa", "engine", "instruct"];
const COL_LETTER = { raw: "R", caa: "C", engine: "E", instruct: "I" };
const CAT_LABEL = { direct_qa: "Direct QA", format_constraints: "Format", negative_constraints: "Negation", reasoning_math: "Math", code_and_sql: "Code & SQL" };

const S = {
  model: "3B", prompts: null, presets: [], byId: {}, cur: null, maxNew: 160, ctrl: null, running: false,
  cols: {}, promptView: {}, artifact: null, scope: [], sel: null, hover: null,
  opts: { modules: { lifecycle: true, boundary_hinge: true, predictor: true, pin_attention: true, unembed_ban: true, stem: true },
          kappa_boundary: 1.5, gamma_unembed: 1.5, beta_attn: 0.5 },
  cache: {}, eco: { pool: "heldout_no_negation", budget: "512", sort: "order" },
  br: { tag: "test40", cat: "all", conds: null, pid: null, disagree: false, q: "" },
  loaded: {},
};

async function api(path) {
  if (S.cache[path]) return S.cache[path];
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  const j = await r.json();
  S.cache[path] = j;
  return j;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ============================== header ============================== */
function initHeader() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $$("#model-switch button").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
  pollStatus();
  setInterval(() => { if (!document.hidden) pollStatus(); }, 3000);
}

function showView(v) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  if (v === "economics") loadEconomics();
  if (v === "browser") loadBrowser();
  if (v === "arena") requestAnimationFrame(drawScope);
}

function setModel(m) {
  if (m === S.model) return;
  if (S.running) stopRun();
  S.model = m;
  $$("#model-switch button").forEach((b) => b.classList.toggle("on", b.dataset.model === m));
  renderPresets();
  resetCards();
  pollStatus();
  const active = $(".tab.active").dataset.view;
  if (active === "economics") loadEconomics();
  if (active === "browser") loadBrowser();
  toast(`Switched to Qwen2.5-${m}. The next live run loads it (the other family is evicted).`);
}

async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    const used = s.ram.used / 2 ** 30, tot = s.ram.total / 2 ** 30;
    $("#ram-val").textContent = `${used.toFixed(1)} / ${tot.toFixed(0)} GB`;
    $("#ram-fill").style.width = `${(100 * s.ram.used) / s.ram.total}%`;
    const res = s.resident.map((x) => x.name.replace("Qwen/Qwen2.5-", "")).join(" + ");
    const mps = s.mps && s.mps.allocated ? ` · MPS ${(s.mps.allocated / 2 ** 30).toFixed(1)} GB` : "";
    const pol = Object.values(s.policy || {})[0];
    $("#ram-models").textContent = s.loading ? `loading ${s.loading.replace("Qwen/", "")}…` : (res ? `${res}${mps}${pol ? ` · Instruct: ${pol}` : ""}` : "no model resident (loads on first run)");
    $("#ram").title = (s.events || []).join("\n");
    const f = s.footprint && s.footprint[S.model];
    if (f) {
      $("#dm-eng").textContent = `${fmtBytes(f.artifact_bytes)} · 0% weights`;
      const wd = f.weight_delta || {};
      if (wd.params_changed_frac != null) {
        $("#dm-ins").textContent = `${fmtBytes(wd.instruct_weights_bytes)} · ${(100 * wd.params_changed_frac).toFixed(1)}% rewritten`;
        $("#dm-ins-bar").style.width = `${100 * wd.params_changed_frac}%`;
      }
    }
  } catch (e) { /* server restarting */ }
}

/* ============================== arena: prompts ============================== */
async function initArena() {
  const p = await api("/api/prompts");
  S.prompts = p.prompts;
  S.presets = p.presets;
  S.byId = Object.fromEntries(p.prompts.map((x) => [x.id, x]));
  const sel = $("#prompt-select");
  sel.innerHTML = `<option value="">— pick a test prompt —</option>` + p.categories.map((c) =>
    `<optgroup label="${CAT_LABEL[c] || c}">` + p.prompts.filter((x) => x.category === c).map((x) =>
      `<option value="${x.id}">${x.id} · ${esc(x.instruction.slice(0, 70))}</option>`).join("") + `</optgroup>`).join("");
  sel.addEventListener("change", () => { if (sel.value) choosePrompt(sel.value, true); });
  $("#use-custom").addEventListener("click", () => {
    const t = $("#custom-prompt").value.trim();
    if (!t) return toast("Type a prompt first.");
    choosePrompt(null, true, t);
  });
  renderPresets();
  $("#run-btn").addEventListener("click", () => S.cur ? run(COLS) : toast("Pick a scenario first."));
  $("#stop-btn").addEventListener("click", stopRun);
  $$("#tok-switch button").forEach((b) => b.addEventListener("click", () => {
    S.maxNew = +b.dataset.n;
    $$("#tok-switch button").forEach((x) => x.classList.toggle("on", x === b));
    renderPresets();
  }));
  $("#instruct-ban").addEventListener("change", () => {
    updateInstructHeader();
    if (S.cur && !S.running) run(["instruct"]);
  });
  $("#instruct-frame").addEventListener("change", () => { updateInstructHeader(); if (S.cur && !S.running) run(["instruct"]); });
  initDrawer();
  resetCards();
  const scope = $("#scope");
  scope.addEventListener("mousemove", scopeHover);
  scope.addEventListener("mouseleave", () => { S.hover = null; $("#scope-tip").style.display = "none"; markHover(null); drawScope(); });
  scope.addEventListener("click", () => { if (S.hover != null) inspect("engine", S.hover); });
  window.addEventListener("resize", () => {
    requestAnimationFrame(drawScope);
    clearTimeout(S._rz);
    S._rz = setTimeout(() => { if ($("#view-economics").classList.contains("active")) loadEconomics(); }, 200);
  });
  choosePrompt("t_ng_01", false);
}

function outcomeChips(pid) {
  const lg = ((S.presets.find((p) => p.prompts.includes(pid)) || {}).logged || {})[S.model] || {};
  const per = lg[pid] || {};
  return `<span class="outcomes" title="Logged outcome at 160 tokens (protocol stack): ✔ = all four checks pass">` + COLS.map((c) => {
    const r = per[c];
    if (!r) return `<span class="oc na">${COL_LETTER[c]}</span>`;
    const ok = r.metrics.composite === 1;
    return `<span class="oc ${ok ? "pass" : "fail"}" title="${c}: ${r.condition} composite ${(100 * r.metrics.composite).toFixed(0)}% (${r.stop_reason}, ${r.n_new_tokens} tok)">${COL_LETTER[c]}</span>`;
  }).join("") + `</span>`;
}

function renderPresets() {
  const box = $("#presets");
  box.innerHTML = S.presets.map((p, k) => `
    <div class="preset${S.cur && p.prompts.includes(S.cur.pid) ? " on" : ""}" data-k="${k}">
      <div class="preset-title"><span>${esc(p.title)}</span><span class="preset-num">#${k + 1}</span></div>
      <div class="preset-prompts">${p.prompts.map((pid) => `
        <div class="pp${S.cur && S.cur.pid === pid ? " on" : ""}" data-pid="${pid}"><span class="pid">${pid}</span>
          <span class="txt">${esc(S.byId[pid].instruction)}</span>${outcomeChips(pid)}</div>`).join("")}</div>
      <div class="preset-blurb">${esc(p.blurb)}</div>
    </div>`).join("");
  $$(".pp", box).forEach((el) => el.addEventListener("click", (e) => { e.stopPropagation(); choosePrompt(el.dataset.pid, true); }));
  $$(".preset", box).forEach((el) => el.addEventListener("click", () => choosePrompt(S.presets[+el.dataset.k].prompts[0], true)));
}

function choosePrompt(pid, autorun, custom) {
  if (S.running) stopRun();
  S.cur = pid ? { pid, instruction: S.byId[pid].instruction } : { pid: null, instruction: custom };
  $("#cur-pid").textContent = pid || "custom";
  $("#cur-prompt").textContent = S.cur.instruction;
  if (pid) $("#prompt-select").value = pid;
  renderPresets();
  resetCards();
  if (autorun) run(COLS);
}

/* ============================== arena: cards ============================== */
function card(col) { return $(`.card[data-col="${col}"]`); }

function resetCards(cols = COLS) {
  for (const c of cols) {
    S.cols[c] = { tokens: [], text: "", done: null, start: null };
    const st = $("[data-role=stream]", card(c));
    st.innerHTML = `<span class="placeholder">${S.cur ? "Ready — press Run." : "Waiting for a prompt."}</span>`;
    $("[data-role=badges]", card(c)).innerHTML = "";
  }
  if (cols.includes("engine")) {
    S.scope = []; S.sel = null; S.hover = null;
    $("#inspector").innerHTML = `<div class="empty">Run a prompt, then click any engine token.</div>`;
    requestAnimationFrame(drawScope);
  }
  updateInstructHeader();
}

function updateInstructHeader() {
  const ban = $("#instruct-ban").checked, frame = $("#instruct-frame").value;
  $("[data-role=title]", card("instruct")).textContent = ban ? "Official Instruct + our ban" : "Official Instruct (post-trained)";
  $("[data-role=isub]", card("instruct")).textContent = ban ? "6.17 GB checkpoint + 0-byte lexical operator (cond 13)" : "6.17 GB checkpoint (cond 12)";
  card("instruct").classList.toggle("hero", ban);
}

function setStatus(msg, err = false) {
  const el = $("#status-line");
  el.textContent = msg || "";
  el.classList.toggle("err", err);
}

function stopRun() {
  if (S.ctrl) S.ctrl.abort();
  fetch("/api/cancel", { method: "POST" }).catch(() => {});
  S.running = false;
  $("#run-btn").disabled = false;
  $("#stop-btn").disabled = true;
  setStatus("Stopped.");
}

async function run(cols) {
  if (!S.cur) return;
  if (S.running) { stopRun(); await new Promise((r) => setTimeout(r, 150)); }
  resetCards(cols);
  for (const c of cols) $("[data-role=stream]", card(c)).innerHTML = `<span class="placeholder">queued…</span>`;
  const body = {
    model: S.model, prompt_id: S.cur.pid, instruction: S.cur.pid ? null : S.cur.instruction, columns: cols,
    instruct_ban: $("#instruct-ban").checked, instruct_frame: $("#instruct-frame").value,
    modules: S.opts.modules, kappa_boundary: S.opts.kappa_boundary, gamma_unembed: S.opts.gamma_unembed,
    beta_attn: S.opts.beta_attn, max_new_tokens: S.maxNew, reference_kl: $("#exact-kl").checked, replay: $("#replay").checked,
  };
  S.ctrl = new AbortController();
  S.running = true;
  $("#run-btn").disabled = true;
  $("#stop-btn").disabled = false;
  setStatus("Starting…");
  const t0 = performance.now();
  try {
    const res = await fetch("/api/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: S.ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)), t0);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") setStatus(String(e.message || e), true);
  } finally {
    S.running = false;
    $("#run-btn").disabled = false;
    $("#stop-btn").disabled = true;
  }
}

function onEvent(ev, t0) {
  switch (ev.type) {
    case "status": setStatus(ev.msg); break;
    case "error": setStatus(ev.msg, true); break;
    case "cancelled": setStatus("Cancelled."); break;
    case "start": onStart(ev); break;
    case "token": onToken(ev); break;
    case "done": onDone(ev); break;
    case "end": setStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s · decoders interleaved token by token on one GPU; tok/s counts each column's own forward passes.`); break;
    default: break;
  }
}

function onStart(ev) {
  const c = ev.col, st = S.cols[c];
  st.start = ev;
  const el = card(c);
  $("[data-role=stream]", el).innerHTML = "";
  if (c === "engine") {
    $("[data-role=title]", el).textContent = ev.label;
    const chip = $("[data-role=cond]", el);
    chip.textContent = ev.condition ? "cond 11 · logged" : (ev.label.includes("v2.3") ? "v2.3 demo" : "custom");
    S.promptView.engine = ev.prompt_view;
    S.artifact = ev.artifact;
    $("#drift-layer").textContent = ev.artifact ? ev.artifact.stop_layer : "–";
    renderCfgDiff(ev.config_diff);
  } else if (c === "instruct") {
    S.promptView.instruct = ev.prompt_view;
  }
  if (ev.source === "cached log") $("[data-role=badges]", el).innerHTML = `<span class="badge neutral">replaying cached log…</span>`;
  setStatus(`Streaming ${ev.label}${ev.reference_kl ? " (with exact step-KL)" : ""}…`);
}

function tokClass(tel) {
  const cls = [];
  if (!tel) return cls;
  if (tel.t < 4 && tel.init_norm > 0) cls.push("init");
  if (tel.hinge_layers_clamped > 0) cls.push("hinge");
  if (tel.ban_flipped) cls.push("ban");
  else if (tel.forbidden_mass_pre != null && tel.forbidden_mass_pre - tel.forbidden_mass_post > 0.02) cls.push("banlite");
  if (tel.stem_guard) cls.push("guard");
  return cls;
}

function tokTitle(tel, i) {
  if (!tel) return `token ${i}`;
  const parts = [`t=${tel.t}`];
  if (tel.drift_z != null) parts.push(`z=${fmt(tel.drift_z)} κ=${fmt(tel.hinge_threshold)}`);
  if (tel.in_boundary_window) parts.push("boundary window");
  if (tel.step_kl_nats != null) parts.push(`KL ${fmt(tel.step_kl_nats, 3)} nats`);
  if (tel.ban_flipped) parts.push(`ban flipped “${tel.top_token_without_ban}”`);
  return parts.join(" · ") + " — click to inspect";
}

function onToken(ev) {
  const c = ev.col, st = S.cols[c];
  const stream = $("[data-role=stream]", card(c));
  const nearBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 40;
  const idx = st.tokens.length;
  const rec = { i: ev.i, s: ev.s, tel: ev.tel || null, eos: ev.eos, start: st.text.length, end: st.text.length, el: null };
  let el;
  if (ev.eos) {
    el = document.createElement("span");
    el.className = "eos-pill";
    el.textContent = "EOS";
  } else {
    el = document.createElement("span");
    el.className = "tok new";
    el.textContent = ev.s;
    st.text += ev.s;
    rec.end = st.text.length;
    for (const k of tokClass(rec.tel)) el.classList.add(k);
  }
  if (rec.tel) {
    el.classList.add("inspectable");
    el.title = tokTitle(rec.tel, ev.i);
    el.addEventListener("click", () => inspect(c, idx));
  }
  rec.el = el;
  st.tokens.push(rec);
  stream.appendChild(el);
  setTimeout(() => el.classList.remove("new"), 600);
  if (ev.contam_from != null) markContam(c, ev.contam_from);
  if (nearBottom) stream.scrollTop = stream.scrollHeight;
  if (c === "engine" && rec.tel) {
    S.scope.push({ ...rec.tel, s: ev.s, eos: ev.eos });
    scheduleScope();
  }
}

function markContam(c, from) {
  const cls = c === "raw" ? "contam" : "runon";
  for (const t of S.cols[c].tokens) if (!t.eos && t.end > from && t.el) t.el.classList.add(cls);
}

function onDone(ev) {
  const c = ev.col, st = S.cols[c];
  st.done = ev;
  const stream = $("[data-role=stream]", card(c));
  if (ev.cut_char != null) {
    let first = null;
    for (const t of st.tokens) if (!t.eos && t.end > ev.cut_char) { t.el.classList.add("cut"); first = first || t; }
    if (first) {
      const pill = document.createElement("span");
      pill.className = "stop-pill";
      pill.textContent = `stop string ${showTok(ev.stop_string || "")}`;
      pill.title = "Stop strings truncate the response here (everything after is discarded)";
      stream.insertBefore(pill, first.el);
    }
  }
  if (ev.stop_reason === "max_tokens") {
    const cap = document.createElement("span");
    cap.className = "cap-pill";
    cap.textContent = `CAP ${ev.n_new_tokens}`;
    cap.title = "Hit max_new_tokens without stopping";
    stream.appendChild(cap);
  }
  for (const [a] of ev.contamination_spans || []) markContam(c, a);
  const m = ev.metrics;
  const b = (ok, label, title = "") => `<span class="badge ${ok ? "pass" : "fail"}" title="${esc(title)}">${label} ${ok ? "✔" : "✘"}</span>`;
  const detail = Object.entries(m.constraint_detail || {}).map(([k, v]) => `${k.split(":")[1]}: ${v ? "pass" : "fail"}`).join("\n");
  const html = [
    `<span class="badge score" title="Composite = mean of the four checks">${(100 * m.composite).toFixed(0)}%</span>`,
    b(m.turn_cross, "TurnCross"),
    m.n_checks ? b(m.constraint, "Constraint", detail) : `<span class="badge neutral" title="No machine-checkable rule found in this custom prompt">Constraint n/a</span>`,
    b(m.clean_stop, "CleanStop", (m.contamination || []).join(", ")),
    b(m.fluency, "Fluency"),
  ];
  if (ev.stem_strict != null) html.push(b(ev.stem_strict, "stem-strict", "No word starts with a banned stem (stricter than the frozen verifier)"));
  const sr = ev.stop_reason === "eos" ? "EOS" : ev.stop_reason === "stop_string" ? "Stop string" : "Cap";
  html.push(`<span class="badge ${ev.stop_reason === "max_tokens" ? "warn" : "neutral"}">${sr}</span>`);
  html.push(`<span class="badge neutral">${ev.n_new_tokens} tok</span>`);
  if (ev.tok_s) html.push(`<span class="badge neutral" title="${ev.source === "cached log" ? "logged speed" : "own forward passes, interleaved with the other columns"}">${ev.tok_s.toFixed(1)} tok/s</span>`);
  if (ev.logged) html.push(`<span class="badge ${ev.logged.identical ? "pass" : "warn"}" title="Compared with the cached benchmark log of ${ev.logged.condition}">${ev.logged.identical ? "= logged ✔" : "≠ logged"}</span>`);
  if (ev.source === "cached log") html.push(`<span class="badge neutral">cached log</span>`);
  const tr = ev.engine_trace || {};
  if (tr.stem_guard_steps) html.push(`<span class="badge neutral" title="Steps where the sub-word guard masked stem completions">guard ×${tr.stem_guard_steps}</span>`);
  $("[data-role=badges]", card(c)).innerHTML = html.join("");
  if (c === "engine") scheduleScope();
}

/* ============================== x-ray oscilloscope ============================== */
let _scopeQueued = false;
function scheduleScope() {
  if (_scopeQueued) return;
  _scopeQueued = true;
  requestAnimationFrame(() => { _scopeQueued = false; drawScope(); });
}

const SC = { L: 46, R: 46, A: [18, 118], B: [162, 80], C: [268, 50] };

function scopeGeom(W) {
  const N = Math.max(S.scope.length, 40);
  const bw = (W - SC.L - SC.R) / N;
  return { N, bw, x: (i) => SC.L + (i + 0.5) * bw };
}

function drawScope() {
  const cv = $("#scope");
  if (!cv || !cv.clientWidth) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const d = S.scope;
  const { bw, x } = scopeGeom(W);
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  g.font = "10.5px -apple-system, Inter, sans-serif";
  const trackLabel = (y, txt) => { g.fillStyle = col("--muted"); g.textAlign = "left"; g.fillText(txt, SC.L + 2, y - 5); };
  const frame = ([y, h]) => { g.strokeStyle = "rgba(148,163,184,0.16)"; g.lineWidth = 1; g.strokeRect(SC.L + 0.5, y + 0.5, W - SC.L - SC.R - 1, h - 1); };

  // --- Track A: drift z vs hinge threshold
  const [ay, ah] = SC.A;
  const zs = d.map((t) => t.drift_z).filter((v) => v != null);
  const ks = d.map((t) => t.hinge_threshold).filter((v) => v != null);
  const zmax = Math.max(3.6, ...zs, ...ks) + 0.4, zmin = Math.min(-3, ...zs) - 0.3;
  const yA = (v) => ay + ah - ((v - zmin) / (zmax - zmin)) * ah;
  d.forEach((t, i) => { if (t.in_boundary_window) { g.fillStyle = "rgba(148,163,184,0.09)"; g.fillRect(x(i) - bw / 2, ay, bw, ah); } });
  g.strokeStyle = "rgba(148,163,184,0.12)";
  [zmin, 0, zmax].forEach((v) => { if (v === 0) { g.beginPath(); g.moveTo(SC.L, yA(0)); g.lineTo(W - SC.R, yA(0)); g.stroke(); } });
  d.forEach((t, i) => {
    if (t.drift_z != null && t.hinge_threshold != null && t.drift_z > t.hinge_threshold) {
      g.fillStyle = "rgba(251,191,36,0.45)";
      g.fillRect(x(i) - bw / 2, yA(t.drift_z), bw, yA(t.hinge_threshold) - yA(t.drift_z));
    }
  });
  // kappa dashed step line
  g.setLineDash([4, 3]); g.strokeStyle = "#e2e8f0"; g.lineWidth = 1.2; g.beginPath();
  let started = false;
  d.forEach((t, i) => {
    if (t.hinge_threshold == null) { started = false; return; }
    const y = yA(t.hinge_threshold), x0 = x(i) - bw / 2, x1 = x(i) + bw / 2;
    if (!started) { g.moveTo(x0, y); started = true; } else g.lineTo(x0, y);
    g.lineTo(x1, y);
  });
  g.stroke(); g.setLineDash([]);
  // z solid line
  g.strokeStyle = col("--hinge"); g.lineWidth = 1.8; g.beginPath(); started = false;
  d.forEach((t, i) => { if (t.drift_z == null) { started = false; return; } const y = yA(t.drift_z); if (!started) { g.moveTo(x(i), y); started = true; } else g.lineTo(x(i), y); });
  g.stroke();
  d.forEach((t, i) => { if (t.hinge_layers_clamped > 0) { g.fillStyle = col("--hinge"); g.beginPath(); g.moveTo(x(i), ay + 7); g.lineTo(x(i) - 3.5, ay + 1); g.lineTo(x(i) + 3.5, ay + 1); g.fill(); } });
  frame(SC.A);
  trackLabel(ay, "A · drift z(t) vs hinge κ(t)  (σ units; grey bands = boundary window)");
  axisLabels(g, yA, [zmin, 0, zmax].map((v) => Math.round(v * 10) / 10), "σ", true);

  // --- Track B: steering schedule + completion cosine
  const [by, bh] = SC.B;
  const nmax = Math.max(5, ...d.map((t) => (t.init_norm || 0) + (t.sustain_norm || 0)));
  const yB = (v) => by + bh - (v / nmax) * bh;
  d.forEach((t, i) => {
    if (t.init_norm > 0) { g.fillStyle = "rgba(74,222,128,0.75)"; g.fillRect(x(i) - Math.max(bw * 0.35, 1), yB(t.init_norm), Math.max(bw * 0.7, 2), by + bh - yB(t.init_norm)); }
  });
  g.strokeStyle = col("--eng"); g.lineWidth = 1.6; g.beginPath(); started = false;
  d.forEach((t, i) => { if (t.sustain_norm == null) return; const y = yB(t.sustain_norm); if (!started) { g.moveTo(x(i), y); started = true; } else g.lineTo(x(i), y); });
  g.stroke();
  const yE = (v) => by + bh - ((v + 0.4) / 1.4) * bh;
  if (S.artifact) { g.setLineDash([2, 3]); g.strokeStyle = "rgba(56,189,248,0.7)"; g.beginPath(); g.moveTo(SC.L, yE(S.artifact.tau_stop)); g.lineTo(W - SC.R, yE(S.artifact.tau_stop)); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = col("--eos"); g.lineWidth = 1.3; g.beginPath(); started = false;
  d.forEach((t, i) => { if (t.eos_cos == null) { started = false; return; } const y = yE(t.eos_cos); if (!started) { g.moveTo(x(i), y); started = true; } else g.lineTo(x(i), y); });
  g.stroke();
  d.forEach((t, i) => { if (t.stop_events && t.stop_events.some((e) => e.reason)) { g.fillStyle = col("--eos"); g.beginPath(); g.arc(x(i), by + 6, 3, 0, 7); g.fill(); } if (t.eos) { g.fillStyle = col("--eos"); g.fillRect(x(i) - 1, by, 2, bh); } });
  frame(SC.B);
  trackLabel(by, "B · ‖init + M3‖ (bars) · ‖sustain‖ · cos(h, μ_eos) vs τ_stop (dotted)");
  axisLabels(g, yB, [0, Math.round(nmax)], "", true);
  axisLabels(g, yE, [-0.4, 0, 1], "", false, W);

  // --- Track C: per-token KL
  const [cy, ch] = SC.C;
  const kls = d.map((t) => t.step_kl_nats ?? t.lexical_kl_nats ?? 0);
  const kmax = Math.max(0.5, ...kls);
  const yC = (v) => cy + ch - (Math.sqrt(Math.max(v, 0)) / Math.sqrt(kmax)) * ch;
  d.forEach((t, i) => {
    const v = kls[i];
    if (!v) return;
    g.fillStyle = t.ban_flipped ? "rgba(192,132,252,0.95)" : "rgba(244,114,182,0.7)";
    g.fillRect(x(i) - Math.max(bw * 0.38, 0.8), yC(v), Math.max(bw * 0.76, 1.6), cy + ch - yC(v));
  });
  frame(SC.C);
  const exact = d.some((t) => t.step_kl_nats != null);
  trackLabel(cy, `C · ${exact ? "exact" : "lexical-only"} per-token KL vs. unsteered (√ scale; purple = ban flipped the top token)`);
  axisLabels(g, yC, [0, Math.round(kmax * 10) / 10], "", true);

  // cursor + selection
  for (const [idx, color] of [[S.hover, "rgba(226,232,240,0.5)"], [S.sel, "rgba(45,212,191,0.9)"]]) {
    if (idx == null || idx >= d.length) continue;
    g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(x(idx), 4); g.lineTo(x(idx), cy + ch); g.stroke();
  }
  g.fillStyle = col("--faint"); g.textAlign = "center";
  const step = Math.max(1, Math.ceil(d.length / 16 / 5) * 5);
  for (let i = 0; i < Math.max(d.length, 1); i += step) g.fillText(String(i), x(i), cy + ch + 12);
  if (!d.length) { g.fillStyle = col("--faint"); g.textAlign = "center"; g.font = "13px -apple-system, sans-serif"; g.fillText(S.running ? "waiting for engine tokens…" : "telemetry appears here while the engine streams", W / 2, ay + ah / 2 + 30); }
}

function axisLabels(g, yf, vals, unit, left, W) {
  g.fillStyle = "#5d6779"; g.textAlign = left ? "right" : "left";
  for (const v of vals) g.fillText(`${v}${unit}`, left ? SC.L - 5 : W - SC.R + 5, yf(v) + 3);
}

function scopeHover(e) {
  const cv = $("#scope"), r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left;
  const { bw } = scopeGeom(cv.clientWidth);
  const i = Math.floor((mx - SC.L) / bw);
  if (i < 0 || i >= S.scope.length) { S.hover = null; $("#scope-tip").style.display = "none"; markHover(null); drawScope(); return; }
  S.hover = i;
  const t = S.scope[i];
  const tip = $("#scope-tip");
  tip.innerHTML = `<b>${esc(showTok(t.eos ? "<EOS>" : t.s))}</b> · t=${t.t}<br>z ${fmt(t.drift_z)} / κ ${fmt(t.hinge_threshold)}${t.in_boundary_window ? " (boundary)" : ""}<br>` +
    `init ${fmt(t.init_norm, 1)} · sustain ${fmt(t.sustain_norm, 1)} · cos ${fmt(t.eos_cos)}<br>KL ${fmt(t.step_kl_nats ?? t.lexical_kl_nats, 3)} nats${t.ban_flipped ? " · <span style='color:var(--ban)'>ban flip</span>" : ""}`;
  tip.style.display = "block";
  tip.style.left = `${Math.min(mx + 12, r.width - 230)}px`;
  tip.style.top = "8px";
  markHover(i);
  drawScope();
}

function markHover(i) {
  $$(".tok.hover").forEach((x) => x.classList.remove("hover"));
  if (i == null) return;
  const t = S.cols.engine && S.cols.engine.tokens[i];
  if (t && t.el) t.el.classList.add("hover");
}

/* ============================== inspector ============================== */
function inspect(col, idx) {
  const t = S.cols[col].tokens[idx];
  if (!t || !t.tel) return;
  $$(".tok.sel, .eos-pill.sel").forEach((x) => x.classList.remove("sel"));
  t.el.classList.add("sel");
  if (col === "engine") { S.sel = idx; drawScope(); }
  const tel = t.tel, pv = S.promptView[col];
  const tags = [];
  if (tel.t < 4 && tel.init_norm > 0) tags.push(`<span class="tag init">init pulse</span>`);
  if (tel.in_boundary_window) tags.push(`<span class="tag bnd">boundary window</span>`);
  if (tel.hinge_layers_clamped > 0) tags.push(`<span class="tag hinge">hinge clamped ×${tel.hinge_layers_clamped} layers</span>`);
  if (tel.ban_flipped) tags.push(`<span class="tag ban">ban flipped the top token</span>`);
  if (tel.stem_guard) tags.push(`<span class="tag guard">sub-word guard</span>`);
  if (t.eos) tags.push(`<span class="tag bnd">EOS chosen</span>`);
  const kv = (rows) => `<div class="kv">${rows.filter(Boolean).map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>`;
  let h = `<div><span class="insp-tok">${esc(showTok(t.eos ? "<EOS>" : t.s))}</span> <span class="hint">${col} · step t = ${tel.t}</span></div><div>${tags.join("")}</div>`;
  if (tel.init_norm != null) {
    h += `<div class="insp-sec"><h4>M1 / M3 · lifecycle schedule</h4>${kv([
      ["‖initiation + M3 shift‖ (Σ layers)", fmt(tel.init_norm, 2)],
      tel.t === 0 ? ["‖M3 predictor part‖ (t = 0 only)", fmt(tel.predictor_norm, 2)] : null,
      ["‖sustain‖", fmt(tel.sustain_norm, 2)],
      ["‖total residual edit‖", fmt(tel.steer_norm, 2)],
      ["cos(h, μ_eos) @ stop layer", `${fmt(tel.eos_cos, 3)} <span class="hint">τ_stop ${fmt(tel.tau_stop, 3)}</span>`],
      tel.stop_events ? ["stop check", tel.stop_events.map((e) => e.reason ? `<b style="color:var(--eos)">${e.reason} → EOS boost</b>` : `score ${fmt(e.score, 3)}`).join(", ")] : null,
    ])}</div>`;
    const rows = (tel.layers || []).map((l) => `<tr><td>L${l.layer}${S.artifact && l.layer === S.artifact.stop_layer ? " ★" : ""}</td><td>${fmt(l.z)}</td><td>${fmt(l.kappa)}</td><td style="color:${l.excess > 0 ? "var(--hinge)" : "inherit"}">${fmt(l.excess)}</td></tr>`).join("");
    h += `<div class="insp-sec"><h4>M2 · boundary-coupled hinge</h4>${kv([
      [`drift z(t) @ layer ${tel.drift_layer}`, `${fmt(tel.drift_z)} σ`],
      ["active threshold κ(t)", `${fmt(tel.hinge_threshold)} σ ${tel.in_boundary_window ? "(boundary: κ_boundary)" : "(clean-token p99, floor 2.6)"}`],
      ["clamped Δz = max(0, z − κ)", fmt(tel.hinge_clamped_delta)],
      ["‖erased vector‖ (Σ layers)", fmt(tel.erase_norm, 3)],
    ])}<table class="mini-table" style="margin-top:5px"><tr><th>layer</th><th>z</th><th>κ</th><th>clamped</th></tr>${rows}</table></div>`;
  }
  if (tel.suppressed_targets || tel.stem_guard) {
    const sup = (tel.suppressed_targets || []).map((s) => `<tr><td>${esc(showTok(s.token))}</td><td>${probCell(s.p_before)}</td><td>${probCell(s.p_after, true)}</td><td>${fmt(s.logit_before, 1)} → ${fmt(s.logit_after, 1)}</td></tr>`).join("");
    const sib = (tel.protected_siblings || []).map((s) => `<tr><td>${esc(showTok(s.token))}</td><td>${probCell(s.p_before)}</td><td>${probCell(s.p_after, true)}</td><td>${fmt(s.centred_logit_shift, 2)}</td></tr>`).join("");
    h += `<div class="insp-sec"><h4>Sibling-orthogonal unembedding ban</h4>${kv([
      ["projection fired", tel.unembed_ban_active ? "yes" : "no"],
      ["forbidden mass before → after", `${fmt(100 * (tel.forbidden_mass_pre || 0), 2)}% → ${fmt(100 * (tel.forbidden_mass_post || 0), 4)}%`],
      ["top token without the ban", esc(showTok(tel.top_token_without_ban || ""))],
      ["KL from the projection alone", `${fmt(tel.lexical_kl_nats, 3)} nats`],
    ])}`;
    if (sup) h += `<table class="mini-table" style="margin-top:6px"><tr><th>suppressed</th><th>p before</th><th>p after</th><th>logit</th></tr>${sup}</table>`;
    if (sib) h += `<table class="mini-table" style="margin-top:6px"><tr><th>protected sibling (K=4)</th><th>p before</th><th>p after</th><th>Δ centred logit</th></tr>${sib}</table>`;
    if (tel.stem_guard) h += `<p class="hint">Sub-word guard: the unfinished word “${esc(tel.stem_guard.partial_word)}” is a prefix of a banned stem, so ${tel.stem_guard.n_masked} completing tokens are masked (e.g. ${tel.stem_guard.examples.map((x) => `“${esc(showTok(x))}”`).join(", ")}).</p>`;
    h += `</div>`;
  }
  if (tel.step_kl_nats != null || tel.lexical_kl_nats != null) {
    h += `<div class="insp-sec"><h4>Alignment tax on this token</h4>${kv([
      ["D_KL(P_unsteered ‖ P_steered)", tel.step_kl_nats != null ? `${fmt(tel.step_kl_nats, 4)} nats${tel.step_kl_floored ? " (masked token, floored)" : ""}` : "enable exact step-KL"],
    ])}<p class="hint">Exact: the same tokens teacher-forced through the ${col === "engine" ? "frozen base" : "Instruct model"} with the hooks suspended. Benchmark-wide mean on clean answers: see the Alignment Tax tab.</p></div>`;
  }
  if (tel.pin_attention && pv) {
    const w = tel.pin_attention, mx = Math.max(...w.slice(pv.extra_tokens || 0), 1e-6);
    const inst = new Set(pv.inst_positions), neg = new Set(pv.neg_target_positions || []);
    const spans = pv.tokens.map((tk, i) => {
      const a = Math.min(1, (w[i] || 0) / mx);
      return `<span class="${inst.has(i) ? "inst" : ""}${neg.has(i) ? " neg" : ""}" style="background:rgba(45,212,191,${(0.06 + 0.7 * a).toFixed(3)})" title="attention ${fmt(w[i], 4)}${inst.has(i) ? ` · +β ${pv.beta_attn} bias` : ""}">${esc(tk)}</span>`;
    }).join("");
    h += `<div class="insp-sec"><h4>M4 · instruction pinning (pinned heads → prompt)</h4>${kv([
      ["attention mass on instruction, with bias", `${fmt(100 * tel.pin_mass, 1)}%`],
      ["… re-normalised without the bias", `${fmt(100 * tel.pin_mass_unbiased, 1)}%`],
      ["β_attn · pinned heads", `${pv.beta_attn} · ${pv.pin_heads.length}`],
    ])}<div class="attn-map" style="margin-top:6px">${spans}</div><p class="hint">Shade = mean attention of the ${pv.pin_heads.length} pinned heads at this step; underline = instruction tokens receiving +β (purple = negated target).</p></div>`;
  }
  $("#inspector").innerHTML = h;
}

function probCell(p, after) {
  const w = Math.max(1, Math.round(60 * Math.min(1, Math.sqrt(p || 0))));
  return `<span class="pbar${after ? " after" : ""}" style="width:${w}px"></span> ${(100 * (p || 0)).toFixed(p < 0.001 ? 4 : 1)}%`;
}

/* ============================== drawer ============================== */
function initDrawer() {
  const dr = $("#drawer");
  $("#open-drawer").addEventListener("click", () => dr.classList.add("open"));
  $("#close-drawer").addEventListener("click", () => dr.classList.remove("open"));
  $$("[data-mod]").forEach((cb) => cb.addEventListener("change", () => { S.opts.modules[cb.dataset.mod] = cb.checked; markOptsDirty(); }));
  const sl = [["#s-kappa", "#o-kappa", "kappa_boundary"], ["#s-gamma", "#o-gamma", "gamma_unembed"], ["#s-beta", "#o-beta", "beta_attn"]];
  for (const [s, o, k] of sl) $(s).addEventListener("input", () => { S.opts[k] = +$(s).value; $(o).textContent = $(s).value; markOptsDirty(); });
  const setAll = (stem) => {
    S.opts = { modules: { lifecycle: true, boundary_hinge: true, predictor: true, pin_attention: true, unembed_ban: true, stem },
               kappa_boundary: 1.5, gamma_unembed: 1.5, beta_attn: 0.5 };
    $$("[data-mod]").forEach((cb) => { cb.checked = S.opts.modules[cb.dataset.mod]; });
    for (const [s, o, k] of sl) { $(s).value = S.opts[k]; $(o).textContent = S.opts[k]; }
    markOptsDirty();
  };
  $("#reset-v23").addEventListener("click", () => setAll(true));
  $("#reset-11").addEventListener("click", () => setAll(false));
  $("#rerun-engine").addEventListener("click", () => { if (S.cur) run(["engine"]); else toast("Pick a prompt first."); });
}

function markOptsDirty() {
  const m = S.opts.modules;
  const def = m.lifecycle && m.boundary_hinge && m.predictor && m.pin_attention && m.unembed_ban && S.opts.kappa_boundary === 1.5 && S.opts.gamma_unembed === 1.5 && S.opts.beta_attn === 0.5;
  $("#cfg-diff").innerHTML = def ? (m.stem ? "Engine = condition 11 + stem morphology (v2.3 demo)." : "Engine = condition 11 exactly (its logged outputs are reproduced byte for byte).") :
    "Custom ablation — press “Re-run engine only”.";
}

function renderCfgDiff(diff) {
  if (!diff) return;
  const keys = Object.keys(diff);
  $("#cfg-diff").innerHTML = keys.length ? `Last run vs condition 11:<br>` + keys.map((k) => `${k}: ${JSON.stringify(diff[k][0])} → <b>${JSON.stringify(diff[k][1])}</b>`).join("<br>") : "Last run = condition 11 exactly.";
}

/* ============================== economics ============================== */
async function loadEconomics() {
  const [eco, ab] = await Promise.all([api(`/api/summary?model=${S.model}`), api(`/api/ablation_data?model=${S.model}`)]);
  if (!S.loaded.ecoInit) {
    S.loaded.ecoInit = true;
    $("#kl-pool").addEventListener("change", () => { S.eco.pool = $("#kl-pool").value; loadEconomics(); });
    $$("#budget-switch button").forEach((b) => b.addEventListener("click", () => { S.eco.budget = b.dataset.b; $$("#budget-switch button").forEach((x) => x.classList.toggle("on", x === b)); loadEconomics(); }));
    $$("#wf-sort button").forEach((b) => b.addEventListener("click", () => { S.eco.sort = b.dataset.s; $$("#wf-sort button").forEach((x) => x.classList.toggle("on", x === b)); loadEconomics(); }));
  }
  renderKpis(eco);
  renderScatter(eco);
  renderKlBars(eco);
  renderUnitTable(eco);
  renderWaterfall(ab);
  renderAblationTable(ab);
}

const rowOf = (eco, c) => eco.rows.find((r) => r.condition === c);
const klOf = (r, pool) => (r.kl.reference ? { mean_kl: 0, top1_agree: 1 } : r.kl[pool]);

function renderKpis(eco) {
  const eng = rowOf(eco, "11_engine_v2_2_lean"), e10 = rowOf(eco, "10_engine_v2_1"), ins = rowOf(eco, "12_official_instruct_chat"), urial = rowOf(eco, "3_urial_3shot_prefix");
  const wd = eco.weight_delta || {};
  const rec = (k) => eco.recovery[k];
  const rtxt = (k) => { const r = rec(k); if (!r || r.point == null) return "–"; if (r.undefined_frac > 0.025) return "not resolved"; return `${Math.round(100 * r.point)}%`; };
  const rci = (k) => { const r = rec(k); return r && r.lo != null && r.undefined_frac <= 0.025 ? `[${Math.round(100 * r.lo)}%, ${Math.round(100 * r.hi)}%]` : "denominator unresolved"; };
  const kE = eng && eng.kl.heldout_no_negation, kI = ins && ins.kl.heldout_no_negation;
  const mins = eco.calibration_seconds ? `${Math.round(eco.calibration_seconds / 60)} min` : "not recorded";
  const card = (label, ev, es, iv, is) => `<div class="kpi glass"><div class="kpi-label">${label}</div><div class="kpi-vs">
      <div><div class="kpi-v eng">${ev}</div><div class="kpi-s">${es}</div></div><div><div class="kpi-v ins">${iv}</div><div class="kpi-s">${is}</div></div></div></div>`;
  $("#kpis").innerHTML = [
    card("Weights modified", "0%", "engine (hooks only)", pct(100 * (wd.params_changed_frac || 0)), `Instruct · of ${((wd.params || 0) / 1e9).toFixed(2)}B params`),
    card("Disk artifact", fmtBytes(eco.artifact_bytes), "calibration file", fmtBytes(wd.instruct_weights_bytes), `whole checkpoint · ×${Math.round((wd.instruct_weights_bytes || 0) / (eco.artifact_bytes || 1))}`),
    card("One-off cost", mins, "forward passes, 80 prompts, M4 Pro", "SFT + RL", "> 1M SFT examples + multi-stage RL"),
    card("Extra prompt tokens", "0", `URIAL needs ${Math.round(urial ? urial.extra_prompt_tokens : 768)}`, `${Math.round(ins ? ins.extra_prompt_tokens : 0)}`, "chat template + system prompt"),
    card("Tokens per answer @512", eng ? Math.round(eng.avg_tokens_512) : "–", e10 ? `v2.1: ${Math.round(e10.avg_tokens_512)}` : "", ins ? Math.round(ins.avg_tokens_512) : "–", eng && ins ? `${(ins.avg_tokens_512 / eng.avg_tokens_512).toFixed(1)}× longer` : ""),
    card("KL tax (held-out, no-neg.)", kE ? fmt(kE.mean_kl, 3) : "–", kE ? `nats/token · top-1 ${pct(100 * kE.top1_agree)}` : "", kI ? fmt(kI.mean_kl, 3) : "–", kE && kI ? `${(kI.mean_kl / kE.mean_kl).toFixed(0)}× more · top-1 ${pct(100 * kI.top1_agree)}` : ""),
  ].join("") + `<div class="kpi glass" style="grid-column: 1 / -1"><div class="kpi-label">Post-training recovery (C_engine − C_raw) / (C_Instruct − C_raw), paired bootstrap 95% CI</div>
    <div class="kpi-vs" style="grid-template-columns: repeat(4, 1fr)">
      <div><div class="kpi-v eng">${rtxt("10_engine_v2_1@512")}</div><div class="kpi-s">v2.1 @ 512 tok ${rci("10_engine_v2_1@512")}</div></div>
      <div><div class="kpi-v eng">${rtxt("11_engine_v2_2_lean@512")}</div><div class="kpi-s">v2.2 lean @ 512 tok ${rci("11_engine_v2_2_lean@512")}</div></div>
      <div><div class="kpi-v" style="color:var(--muted)">${rtxt("10_engine_v2_1@160")}</div><div class="kpi-s">v2.1 @ 160 tok · ${rci("10_engine_v2_1@160")}</div></div>
      <div><div class="kpi-v" style="color:var(--muted)">${rtxt("11_engine_v2_2_lean@160")}</div><div class="kpi-s">v2.2 lean @ 160 tok · Instruct's own gain is not resolved at 160 tokens</div></div>
    </div></div>`;
}

const KIND_COLOR = { base: "#94a3b8", prompt: "#fbbf24", steer: "#fb923c", engine: "#2dd4bf", instruct: "#8b9cfb" };
const POOL_LABEL = { heldout_no_negation: "held-out, no-negation prompts", heldout_all: "held-out, all prompts", calibration_no_negation: "calibration, no-negation", calibration_all: "calibration, all prompts" };

function renderScatter(eco) {
  const pool = S.eco.pool, budget = S.eco.budget;
  const pts = eco.rows.map((r) => {
    const k = klOf(r, pool), y = budget === "512" ? r.composite_512 : r.composite_160;
    return k && y != null ? { r, x: k.mean_kl, y, top1: k.top1_agree } : null;
  }).filter(Boolean);
  const missing = eco.rows.filter((r) => !pts.find((p) => p.r === r)).map((r) => r.label);
  const W = Math.max(480, $("#scatter").clientWidth || 640), H = 380, L = 52, R = 24, T = 18, B = 44;
  const xmax = Math.max(0.1, ...pts.map((p) => p.x)) * 1.12;
  const ymin = Math.floor(Math.min(...pts.map((p) => p.y), 75) / 5) * 5, ymax = 100;
  const X = (v) => L + (v / xmax) * (W - L - R), Y = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
  for (let v = ymin; v <= ymax; v += 5) s += `<g class="grid"><line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/></g><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  const xt = niceTicks(xmax);
  for (const v of xt) s += `<g class="grid"><line y1="${T}" y2="${H - B}" x1="${X(v)}" x2="${X(v)}"/></g><text x="${X(v)}" y="${H - B + 15}" text-anchor="middle">${v}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">clean-token KL from the raw base (nats/token) — ${POOL_LABEL[pool]}</text>`;
  s += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Composite at ${budget} tokens (%)</text>`;
  const eng = pts.find((p) => p.r.condition === "11_engine_v2_2_lean") || pts.find((p) => p.r.condition === "10_engine_v2_1");
  const ins = pts.find((p) => p.r.condition === "12_official_instruct_chat");
  if (eng && ins && eng.x > 0) {
    s += `<line x1="${X(ins.x) - 18}" y1="${Y(ins.y) + 6}" x2="${X(eng.x) + 16}" y2="${Y(eng.y) + 4}" stroke="rgba(226,232,240,0.35)" stroke-dasharray="3 3" marker-end="url(#arr)"/>
      <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="rgba(226,232,240,0.5)"/></marker></defs>
      <text class="lbl" x="${(X(ins.x) + X(eng.x)) / 2}" y="${Math.min(Y(ins.y), Y(eng.y)) - 12}" text-anchor="middle">${(ins.x / eng.x).toFixed(0)}× less distortion, ${eng.y >= ins.y ? "+" : ""}${(eng.y - ins.y).toFixed(1)} Composite</text>`;
  }
  const placed = [];
  for (const p of pts.sort((a, b) => (b.r.bytes || 0) - (a.r.bytes || 0))) {
    const rr = 5 + 3 * Math.log10(1 + (p.r.bytes || 0) / 1e6);
    const cx = X(p.x), cy = Y(p.y);
    s += `<circle cx="${cx}" cy="${cy}" r="${rr}" fill="${KIND_COLOR[p.r.kind]}" fill-opacity="0.28" stroke="${KIND_COLOR[p.r.kind]}" stroke-width="1.5"><title>${esc(p.r.label)}: KL ${fmt(p.x, 3)}, top-1 ${pct(100 * p.top1)}, Composite ${fmt(p.y, 1)}, ${fmtBytes(p.r.bytes) === "–" ? "no weights" : fmtBytes(p.r.bytes)}</title></circle>`;
    let ly = cy - rr - 5;
    while (placed.some((q) => Math.abs(q.y - ly) < 13 && Math.abs(q.x - cx) < 120)) ly -= 13;
    placed.push({ x: cx, y: ly });
    const anchor = cx > W - 170 ? "end" : "start";
    s += `<text class="lbl" x="${cx + (anchor === "end" ? -rr : rr) * 0.3}" y="${ly}" text-anchor="${anchor}">${esc(p.r.label)} <tspan class="lbl dim">${fmt(p.y, 1)}%</tspan></text>`;
  }
  s += `</svg>`;
  $("#scatter").innerHTML = s;
  $("#scatter-note").textContent = `Bubble area ~ log(bytes changed or shipped). ${missing.length ? `Not shown (no ${budget === "512" ? "512-token run" : "KL measurement"}): ${missing.join(", ")}.` : ""} ${eco.kl_note}`;
}

function niceTicks(max) {
  const step = max > 0.4 ? 0.1 : max > 0.2 ? 0.05 : 0.02;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

function renderKlBars(eco) {
  const pool = S.eco.pool;
  const rows = eco.rows.filter((r) => klOf(r, pool)).map((r) => ({ r, k: klOf(r, pool) }));
  const W = Math.max(420, $("#klbars").clientWidth || 520), rowH = 30, L = 190, R = 150, H = rows.length * rowH + 24;
  const max = Math.max(...rows.map((x) => x.k.mean_kl), 0.05);
  const X = (v) => L + (v / max) * (W - L - R);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
  rows.forEach(({ r, k }, i) => {
    const y = 8 + i * rowH;
    s += `<text class="lbl" x="${L - 8}" y="${y + 15}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<rect x="${L}" y="${y + 4}" width="${Math.max(X(k.mean_kl) - L, 1.5)}" height="16" rx="3" fill="${KIND_COLOR[r.kind]}" fill-opacity="0.75"/>`;
    s += `<text x="${X(k.mean_kl) + 6}" y="${y + 16}">${r.kl.reference ? "0 (reference)" : `${fmt(k.mean_kl, 3)} · top-1 ${pct(100 * k.top1_agree)}`}</text>`;
  });
  s += `</svg>`;
  $("#klbars").innerHTML = s;
  $("#kl-pool-label").textContent = POOL_LABEL[pool];
  $("#kl-note").textContent = `Floor: the raw base scored against itself, single pass vs. step-by-step, is ${fmt(eco.kl_floor, 4)} nats. The lexical ban pays its tax only where the banned word would have been written, so the no-negation pool isolates the cost of the residual hooks.`;
}

function renderUnitTable(eco) {
  const h = `<tr><th>Approach</th><th>Weights changed</th><th>Artifact / delta</th><th class="l">One-off cost</th><th>Extra prompt tok</th>
    <th>KL held-out all / no-neg.</th><th>Composite 160</th><th>Composite 512</th><th>Tok/answer 160 / 512</th><th>tok/s</th></tr>`;
  const rows = eco.rows.map((r) => {
    const ha = r.kl.heldout_all, hn = r.kl.heldout_no_negation;
    const kl = r.kl.reference ? "0 (reference)" : ha ? `${fmt(ha.mean_kl, 3)} / ${fmt(hn && hn.mean_kl, 3)}` : "–";
    return `<tr class="${r.kind === "engine" ? "row-eng" : r.kind === "instruct" ? "row-ins" : ""}"><td>${esc(r.label)} <span class="grp mono">${r.condition}</span></td>
      <td class="num">${r.weights_changed_frac ? pct(100 * r.weights_changed_frac) : "0%"}</td><td class="num">${r.bytes ? fmtBytes(r.bytes) : r.kind === "prompt" ? "768-token prompt" : "–"}</td>
      <td class="l">${esc(r.one_off_cost)}</td><td class="num">${Math.round(r.extra_prompt_tokens)}</td><td class="num">${kl}</td>
      <td class="num"><b>${fmt(r.composite_160, 1)}</b></td><td class="num">${r.composite_512 != null ? `<b>${fmt(r.composite_512, 1)}</b>` : "–"}</td>
      <td class="num">${Math.round(r.avg_tokens_160)} / ${r.avg_tokens_512 != null ? Math.round(r.avg_tokens_512) : "–"}</td>
      <td class="num" title="${r.tok_per_s_source}">${fmt(r.tok_per_s, 1)}</td></tr>`;
  }).join("");
  $("#ue-table").innerHTML = h + rows;
  $("#ue-model").textContent = eco.model;
  const rec = Object.entries(eco.recovery).map(([k, r]) => {
    const [c, b] = k.split("@");
    const val = r.point == null ? "undefined" : r.undefined_frac > 0.025 ? `not resolved (point ${Math.round(100 * r.point)}%, denominator ≤ 0 in ${Math.round(100 * r.undefined_frac)}% of resamples)` : `${Math.round(100 * r.point)}% [${Math.round(100 * r.lo)}%, ${Math.round(100 * r.hi)}%]`;
    return `${c} @ ${b} tok: ${val}`;
  }).join(" · ");
  $("#recovery-note").textContent = `Recovery vs. Instruct (chat): ${rec}. tok/s: same-session benchmark where available, otherwise from the logs.`;
}

const GROUP_COLOR = { "Baselines": "#94a3b8", "Phase 1 · v1": "#fb923c", "Phase 2 · v2": "#fbbf24", "Phase 3 · v2.1": "#4ade80", "Phase 4 · lean": "#2dd4bf", "Phase 4 · post-trained": "#8b9cfb" };

function renderWaterfall(ab) {
  let rows = [...ab.rows];
  if (S.eco.sort === "score") rows.sort((a, b) => b.composite - a.composite);
  const raw = ab.rows.find((r) => r.condition === "1_raw_base").composite;
  const W = Math.max(600, $("#waterfall").clientWidth || 900), rowH = 21, L = 250, R = 90, T = 22, H = T + rows.length * rowH + 26;
  const lo = Math.min(...rows.map((r) => raw + r.ci_vs_raw[0]), raw) - 2, hi = 100;
  const X = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
  for (let v = Math.ceil(lo / 5) * 5; v <= hi; v += 5) s += `<g class="grid"><line x1="${X(v)}" x2="${X(v)}" y1="${T - 6}" y2="${H - 22}"/></g><text x="${X(v)}" y="${H - 8}" text-anchor="middle">${v}</text>`;
  s += `<line x1="${X(raw)}" x2="${X(raw)}" y1="${T - 10}" y2="${H - 22}" stroke="#e2e8f0" stroke-dasharray="3 3"/><text class="lbl" x="${X(raw) + 4}" y="${T - 10}">raw base ${fmt(raw, 1)}</text>`;
  rows.forEach((r, i) => {
    const y = T + i * rowH;
    const c = GROUP_COLOR[r.group] || "#94a3b8";
    const x0 = X(Math.min(raw, r.composite)), x1 = X(Math.max(raw, r.composite));
    s += `<text class="lbl${r.instruct ? "" : ""}" x="${L - 8}" y="${y + 13}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<rect x="${x0}" y="${y + 3}" width="${Math.max(x1 - x0, 1.5)}" height="13" rx="2" fill="${c}" fill-opacity="${r.composite >= raw ? 0.8 : 0.35}"><title>${r.condition}: ${fmt(r.composite, 1)} (Δ ${fmt(r.delta_vs_raw, 1)}, 95% CI [${fmt(r.ci_vs_raw[0], 1)}, ${fmt(r.ci_vs_raw[1], 1)}])</title></rect>`;
    if (r.condition !== "1_raw_base") {
      const a = X(raw + r.ci_vs_raw[0]), b = X(raw + r.ci_vs_raw[1]);
      s += `<line x1="${a}" x2="${b}" y1="${y + 9.5}" y2="${y + 9.5}" stroke="#e2e8f0" stroke-opacity="0.75"/><line x1="${a}" x2="${a}" y1="${y + 6}" y2="${y + 13}" stroke="#e2e8f0" stroke-opacity="0.75"/><line x1="${b}" x2="${b}" y1="${y + 6}" y2="${y + 13}" stroke="#e2e8f0" stroke-opacity="0.75"/>`;
    }
    s += `<text x="${W - R + 8}" y="${y + 13}">${fmt(r.composite, 1)} <tspan fill="${r.delta_vs_raw > 0 ? "#34d399" : r.delta_vs_raw < 0 ? "#f87171" : "#8d97aa"}">${r.condition === "1_raw_base" ? "" : (r.delta_vs_raw >= 0 ? "+" : "") + fmt(r.delta_vs_raw, 1)}</tspan></text>`;
  });
  s += `</svg>`;
  $("#waterfall").innerHTML = s;
  $("#wf-note").textContent = `${ab.model}: bar = Composite relative to the raw base; whisker = 95% paired-bootstrap CI of the difference (CIs that cross the dashed line are not resolved). Colours: ${Object.keys(GROUP_COLOR).join(" · ")}. ${ab.note}`;
}

function renderAblationTable(ab) {
  const cols = [["label", "Condition", "l"], ["group", "Phase", "l"], ["composite", "Composite"], ["delta_vs_raw", "Δ raw [95% CI]"], ["turn_cross", "Turn"], ["constraint", "Constr."], ["clean_stop", "Stop"], ["fluency", "Flu."], ["natural_eos", "EOS"],
    ...ab.categories.map((c) => [`cat:${c}`, CAT_LABEL[c] || c]), ["composite_512", "512 tok"], ["composite_sdpa", "SDPA"], ["avg_new_tokens", "tok/ans"], ["tok_per_s", "tok/s"]];
  const st = S.eco.tableSort || { k: null, asc: false };
  let rows = [...ab.rows];
  const val = (r, k) => (k.startsWith("cat:") ? r.categories[k.slice(4)] : r[k]);
  if (st.k) rows.sort((a, b) => { const va = val(a, st.k), vb = val(b, st.k); const c = typeof va === "string" ? va.localeCompare(vb) : (va ?? -1) - (vb ?? -1); return st.asc ? c : -c; });
  const heat = (v) => v == null ? "–" : `<span class="heat" style="background:rgba(${v >= 90 ? "52,211,153" : v >= 75 ? "251,191,36" : "248,113,113"},${(0.08 + 0.25 * Math.abs(v - 70) / 30).toFixed(2)})">${fmt(v, 0)}</span>`;
  const head = `<tr>${cols.map(([k, l, cls]) => `<th data-k="${k}" class="${cls || ""}${st.k === k ? ` sorted${st.asc ? " asc" : ""}` : ""}">${l}</th>`).join("")}</tr>`;
  const body = rows.map((r) => `<tr class="${r.group.includes("lean") || r.condition === "10_engine_v2_1" ? "row-eng" : r.instruct ? "row-ins" : ""}">
    <td>${esc(r.label)} <span class="grp mono">${r.condition}</span></td><td class="l grp">${esc(r.group)}</td><td class="num"><b>${fmt(r.composite, 1)}</b></td>
    <td class="num">${r.condition === "1_raw_base" ? "–" : `${r.delta_vs_raw >= 0 ? "+" : ""}${fmt(r.delta_vs_raw, 1)} [${fmt(r.ci_vs_raw[0], 1)}, ${fmt(r.ci_vs_raw[1], 1)}]`}</td>
    ${["turn_cross", "constraint", "clean_stop", "fluency", "natural_eos"].map((k) => `<td class="num">${fmt(r[k], 0)}</td>`).join("")}
    ${ab.categories.map((c) => `<td class="num">${heat(r.categories[c])}</td>`).join("")}
    <td class="num">${fmt(r.composite_512, 1)}</td><td class="num">${fmt(r.composite_sdpa, 1)}</td><td class="num">${fmt(r.avg_new_tokens, 0)}</td><td class="num">${fmt(r.tok_per_s, 1)}</td></tr>`).join("");
  const t = $("#ab-table");
  t.innerHTML = head + body;
  $$("th", t).forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k;
    S.eco.tableSort = { k, asc: st.k === k ? !st.asc : false };
    renderAblationTable(ab);
  }));
}

/* ============================== benchmark browser ============================== */
const DEFAULT_CONDS = ["1_raw_base", "4_static_caa", "10_engine_v2_1", "11_engine_v2_2_lean", "12_official_instruct_chat", "13_instruct_plus_orth_ban_chat"];

async function loadBrowser() {
  if (!S.prompts) await api("/api/prompts");
  const tags = [["test40", "protocol · 160 tok"], ["test40_512tok", "512 tok"], ["test40_sdpa", "SDPA"]];
  $("#tag-switch").innerHTML = tags.map(([t, l]) => `<button data-t="${t}" class="${S.br.tag === t ? "on" : ""}">${l}</button>`).join("");
  $$("#tag-switch button").forEach((b) => b.addEventListener("click", () => { S.br.tag = b.dataset.t; S.br.conds = null; loadBrowser(); }));
  $("#cat-chips").innerHTML = [["all", "All"], ...Object.entries(CAT_LABEL)].map(([k, l]) => `<button class="cchip${S.br.cat === k ? " on" : ""}" data-c="${k}">${l}</button>`).join("");
  $$("#cat-chips .cchip").forEach((b) => b.addEventListener("click", () => { S.br.cat = b.dataset.c; loadBrowser(); }));
  if (!S.loaded.brInit) {
    S.loaded.brInit = true;
    $("#only-disagree").addEventListener("change", (e) => { S.br.disagree = e.target.checked; renderBrList(); });
    $("#br-search").addEventListener("input", (e) => { S.br.q = e.target.value.toLowerCase(); renderBrList(); });
  }
  const lg = await api(`/api/logs?model=${S.model}&tag=${S.br.tag}`);
  S.br.logs = lg;
  const avail = Object.keys(lg.conditions);
  if (!S.br.conds) S.br.conds = DEFAULT_CONDS.filter((c) => avail.includes(c));
  if (!S.br.conds.length) S.br.conds = avail.slice(0, 4);
  $("#cond-chips").innerHTML = `<span class="hint" style="margin-right:6px">Conditions (${avail.length} logged for ${esc(lg.tag_label)}):</span>` + avail.map((c) =>
    `<button class="cchip${c.startsWith("12_") || c.startsWith("13_") ? " ins" : ""}${S.br.conds.includes(c) ? " on" : ""}" data-c="${c}" title="${esc(lg.labels[c] || c)}">${c}</button>`).join("");
  $$("#cond-chips .cchip").forEach((b) => b.addEventListener("click", () => {
    const c = b.dataset.c;
    S.br.conds = S.br.conds.includes(c) ? S.br.conds.filter((x) => x !== c) : [...S.br.conds, c];
    b.classList.toggle("on");
    renderBrList();
    renderBrDetail();
  }));
  renderBrList();
  renderBrDetail();
}

function brRecs(pid) { return S.br.conds.map((c) => [c, (S.br.logs.conditions[c] || {})[pid]]).filter(([, r]) => r); }

function renderBrList() {
  const items = S.prompts.filter((p) => {
    if (S.br.cat !== "all" && p.category !== S.br.cat) return false;
    if (S.br.q && !(`${p.id} ${p.instruction}`.toLowerCase().includes(S.br.q))) return false;
    if (S.br.disagree) { const cs = new Set(brRecs(p.id).map(([, r]) => r.metrics.composite)); if (cs.size < 2) return false; }
    return true;
  });
  $("#br-list").innerHTML = items.map((p) => `<div class="br-item${S.br.pid === p.id ? " on" : ""}" data-pid="${p.id}">
      <div class="top"><span><span class="pid">${p.id}</span> <span class="cat">${CAT_LABEL[p.category]}</span></span>
      <span class="outcomes">${brRecs(p.id).map(([c, r]) => `<span class="oc ${r.metrics.composite === 1 ? "pass" : r.metrics.composite >= 0.75 ? "part" : "fail"}" title="${c}: ${(100 * r.metrics.composite).toFixed(0)}%"></span>`).join("")}</span></div>
      <div class="ins-txt">${esc(p.instruction)}</div></div>`).join("") || `<div class="empty">No prompt matches.</div>`;
  $$("#br-list .br-item").forEach((el) => el.addEventListener("click", () => { S.br.pid = el.dataset.pid; renderBrList(); renderBrDetail(); }));
}

function highlight(text, contam, stems) {
  let cut = -1;
  for (const m of contam || []) { const i = text.indexOf(m); if (i >= 0 && (cut < 0 || i < cut)) cut = i; }
  const mark = (s) => {
    let h = esc(s);
    for (const st of stems || []) h = h.replace(new RegExp(`\\b(${st}[a-z]*)`, "gi"), `<mark class="ban">$1</mark>`);
    return h;
  };
  return cut < 0 ? mark(text) : mark(text.slice(0, cut)) + `<mark class="contam">${mark(text.slice(cut))}</mark>`;
}

function renderBrDetail() {
  const pid = S.br.pid, box = $("#br-detail");
  if (!pid) { box.innerHTML = `<div class="empty">Select a prompt on the left.</div>`; return; }
  const p = S.byId[pid];
  const checks = p.checks.map((c) => c.type).join(", ");
  const recs = brRecs(pid);
  box.innerHTML = `<div class="br-head"><div><h2>${esc(p.instruction)}</h2><div class="hint"><span class="pid">${pid}</span> ${CAT_LABEL[p.category]} · checks: ${esc(checks)}${p.stems.length ? ` · banned stems: ${p.stems.join(", ")}` : ""}</div></div>
    <button class="btn primary" id="br-rerun">Re-run live in Arena →</button></div>
    <div class="br-cards">${recs.map(([c, r]) => {
      const m = r.metrics;
      const b = (ok, l) => `<span class="badge ${ok ? "pass" : "fail"}">${l} ${ok ? "✔" : "✘"}</span>`;
      return `<div class="br-card"><header><b>${esc(S.br.logs.labels[c] || c)}</b><span class="badge score">${(100 * m.composite).toFixed(0)}%</span></header>
        <div class="hint" style="padding:0 10px"><span class="mono">${c}</span> · ${r.stop_reason}${r.stop_string ? ` ${esc(showTok(r.stop_string))}` : ""} · ${r.n_new_tokens} tok${r.extra_prompt_tokens ? ` · +${r.extra_prompt_tokens} prompt tok` : ""}</div>
        <div class="resp">${highlight(r.response, m.contamination, p.stems)}</div>
        <footer>${b(m.turn_cross, "Turn")}${b(m.constraint, "Constraint")}${b(m.clean_stop, "CleanStop")}${b(m.fluency, "Fluency")}${m.contamination.length ? `<span class="badge fail" title="${esc(m.contamination.join(", "))}">run-on: ${esc(m.contamination[0])}</span>` : ""}</footer></div>`;
    }).join("") || `<div class="empty">None of the selected conditions has a log for this prompt.</div>`}</div>`;
  $("#br-rerun").addEventListener("click", () => {
    showView("arena");
    if (S.br.tag === "test40_512tok") { S.maxNew = 512; $$("#tok-switch button").forEach((x) => x.classList.toggle("on", x.dataset.n === "512")); }
    choosePrompt(pid, true);
  });
}

/* ============================== boot ============================== */
initHeader();
initArena().catch((e) => setStatus(`Could not load prompts: ${e.message}`, true));
