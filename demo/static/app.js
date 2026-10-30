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

/* Enter / Space activate any non-native [role=button] (keyboard contract for clickable divs). */
document.addEventListener("keydown", (e) => {
  const el = e.target;
  if ((e.key !== "Enter" && e.key !== " ") || e.altKey || e.ctrlKey || e.metaKey || !el.matches?.("[role=button]") || el.matches("button,a[href],input,select,textarea,summary")) return;
  e.preventDefault();
  el.click();
});

/* ============================== header ============================== */
function initHeader() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $(".tabs").addEventListener("keydown", (e) => { // roving tabindex: ←/→ wrap, Home/End jump, selection follows focus
    const tabs = $$(".tab"), i = tabs.indexOf(e.target), j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (i < 0 || j == null) return;
    e.preventDefault();
    const t = tabs[(j + tabs.length) % tabs.length];
    t.focus(); showView(t.dataset.view);
  });
  $$("#model-switch button").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
  const tb = $(".topbar"); // --topbar-h follows the real header height (it wraps to 2–4 rows below 1440 px)
  new ResizeObserver(() => document.documentElement.style.setProperty("--topbar-h", `${tb.offsetHeight}px`)).observe(tb);
  pollStatus();
  setInterval(() => { if (!document.hidden) pollStatus(); }, 3000);
}

function showView(v) {
  if (!$(`#view-${v}`).classList.contains("active")) window.scrollTo(0, 0);
  $$(".tab").forEach((b) => { const on = b.dataset.view === v; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; });
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === `view-${v}`));
  if (v === "economics") loadEconomics();
  if (v === "browser") loadBrowser();
  if (v === "arena") requestAnimationFrame(drawScope);
}

function setModel(m) {
  if (m === S.model) return;
  if (S.running) stopRun();
  if (Z.ctrl) zenStop(false);
  S.model = m;
  $$("#model-switch button").forEach((b) => { b.classList.toggle("on", b.dataset.model === m); b.setAttribute("aria-pressed", b.dataset.model === m); });
  renderPresets();
  resetCards();
  pollStatus();
  const active = $(".tab.active").dataset.view;
  if (active === "economics") loadEconomics();
  if (active === "browser") loadBrowser();
  if (Z.on) { zenSync(); zenReset(); zenStatus("Press ↵ to ask both models."); }
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
    const fp = JSON.stringify(s.footprint || null);
    if (fp !== S._fp) { S._fp = fp; S.footprint = s.footprint; if (typeof updateInstructHeader === "function") updateInstructHeader(); if (Z.on) zenSync(); }
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
  $("#stop-btn").addEventListener("click", () => stopRun(true));
  $$("#tok-switch button").forEach((b) => b.addEventListener("click", () => {
    S.maxNew = +b.dataset.n;
    $$("#tok-switch button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", x === b); });
    renderPresets();
    markStale();
  }));
  $("#exact-kl").addEventListener("change", markStale);
  $("#replay").addEventListener("change", markStale);
  const instChange = () => {                   // mid-run: queue an Instruct re-run instead of relabelling a card that has not rerun
    if (S.running) { S.pendingInstruct = true; toast("Applies when this run finishes"); markStale(); }
    else if (S.cur) run(["instruct"]);
    else updateInstructHeader();
  };
  $("#instruct-ban").addEventListener("change", instChange);
  $("#instruct-frame").addEventListener("change", instChange);
  for (const c of COLS) {
    $("[data-role=stale]", card(c)).addEventListener("click", () => run([c]));
    const st = $("[data-role=stream]", card(c)), tel = c === "engine" || c === "instruct";
    st.tabIndex = 0;
    st.setAttribute("role", "region");
    st.setAttribute("aria-label", `${$(".card-title", card(c)).textContent} output${tel ? "; ←/→ step through tokens, Enter inspects" : ""}`);
    st.addEventListener("keydown", (e) => streamKey(c, st, e));
    st.addEventListener("blur", () => {
      $$(".kbd", st).forEach((x) => x.classList.remove("kbd"));
      if (c === "engine" && S.hover != null && S.hover === st._cur) { S.hover = null; markHover(null); scheduleScope(); }
    });
  }
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
  setStatus("Press Run 4-way ▶, or click a scenario to run it now: one prompt, four decoders side by side. The first run loads the weights; watch RAM, top right.");
}

/* Roving keyboard cursor over a stream's inspectable tokens (one tab stop per card instead of one per token). */
function streamKey(c, st, e) {
  const toks = (S.cols[c] || {}).tokens || [], idx = toks.map((t, i) => (t.tel ? i : -1)).filter((i) => i >= 0);
  if (!idx.length || e.altKey || e.ctrlKey || e.metaKey) return;
  const k = idx.indexOf(st._cur), last = idx.length - 1;
  const j = { ArrowRight: k < 0 ? 0 : Math.min(k + 1, last), ArrowLeft: k < 0 ? last : Math.max(k - 1, 0), Home: 0, End: last }[e.key];
  if (e.key === "Enter" && st._cur != null) { e.preventDefault(); inspect(c, st._cur); return; }
  if (j == null) return;
  e.preventDefault();
  st._cur = idx[j];
  $$(".kbd", st).forEach((x) => x.classList.remove("kbd"));
  const el = toks[st._cur].el;
  el.classList.add("kbd");
  el.scrollIntoView({ block: "nearest" });
  if (c === "engine") { S.hover = st._cur; markHover(st._cur); drawScope(); } // tokens[i] ↔ S.scope[i]
}

function outcomeChips(pid) {
  const lg = ((S.presets.find((p) => p.prompts.includes(pid)) || {}).logged || {})[S.model] || {};
  const per = lg[pid] || {}, said = [];
  const chips = COLS.map((c) => {
    const r = per[c];
    if (!r) { said.push(`${COL_LETTER[c]} no log`); return `<span class="oc na">${COL_LETTER[c]}</span>`; }
    const ok = r.metrics.composite === 1;
    said.push(`${COL_LETTER[c]} ${ok ? "pass" : "fail"}`);
    return `<span class="oc ${ok ? "pass" : "fail"}" title="${c}: ${r.condition} composite ${(100 * r.metrics.composite).toFixed(0)}% (${r.stop_reason}, ${r.n_new_tokens} tok)">${COL_LETTER[c]}</span>`;
  }).join("");
  return `<span class="outcomes" role="img" aria-label="${said.join(", ")}" title="Logged outcome at 160 tokens (protocol stack): filled = all four checks pass, hollow = fail">${chips}</span>`;
}

function renderPresets() {
  const box = $("#presets"), sl = box.scrollLeft, a = document.activeElement;
  const refocus = a && box.contains(a) ? (a.dataset.pid ? `.pp[data-pid="${a.dataset.pid}"]` : `.preset[data-k="${a.closest(".preset").dataset.k}"] .preset-title`) : null;
  box.innerHTML = S.presets.map((p, k) => `
    <div class="preset${S.cur && p.prompts.includes(S.cur.pid) ? " on" : ""}" data-k="${k}">
      <button type="button" class="preset-title"><span>${esc(p.title)}</span><span class="preset-num">#${k + 1}</span></button>
      <div class="preset-prompts">${p.prompts.map((pid) => `
        <button type="button" class="pp${S.cur && S.cur.pid === pid ? " on" : ""}" data-pid="${pid}"${S.cur && S.cur.pid === pid ? ` aria-current="true"` : ""}><span class="pid">${pid}</span>
          <span class="txt">${esc(S.byId[pid].instruction)}</span>${outcomeChips(pid)}</button>`).join("")}</div>
      <div class="preset-blurb">${esc(p.blurb)}</div>
    </div>`).join("");
  $$(".pp", box).forEach((el) => el.addEventListener("click", (e) => { e.stopPropagation(); choosePrompt(el.dataset.pid, true); }));
  $$(".preset", box).forEach((el) => el.addEventListener("click", () => choosePrompt(S.presets[+el.dataset.k].prompts[0], true)));
  box.scrollLeft = sl;                         // the ≤1100 px strip keeps its place; keyboard focus survives the re-render
  const on = $(".preset.on", box), d = on ? on.getBoundingClientRect().left - box.getBoundingClientRect().left : 0;
  if (on && box.scrollWidth > box.clientWidth && (d < 0 || d + on.offsetWidth > box.clientWidth + 1)) box.scrollLeft += d; // …unless the current scenario is off-strip
  if (refocus && $(refocus, box)) $(refocus, box).focus({ preventScroll: true });
}

function choosePrompt(pid, autorun, custom) {
  if (S.running) stopRun();
  S.cur = pid ? { pid, instruction: S.byId[pid].instruction } : { pid: null, instruction: custom };
  $("#cur-pid").textContent = pid || "custom";
  $("#cur-prompt").textContent = S.cur.instruction;
  if (pid) $("#prompt-select").value = pid;
  $$("#tok-switch button").forEach((x) => { x.classList.toggle("on", +x.dataset.n === S.maxNew); x.setAttribute("aria-pressed", +x.dataset.n === S.maxNew); });
  renderPresets();
  resetCards();
  const fix = S.presets.find((p) => p.id === "fix_instruct"), sw = $(".card[data-col=instruct] .switch");
  if (pid && fix && fix.prompts.includes(pid) && !matchMedia("(prefers-reduced-motion: reduce)").matches) {   // "Flip + ban": point at the switch the scenario is about
    sw.classList.remove("pulse"); void sw.offsetWidth; sw.classList.add("pulse");
    sw.addEventListener("animationend", () => sw.classList.remove("pulse"), { once: true });
  }
  if (autorun) run(COLS);
}

/* ============================== arena: cards ============================== */
function card(col) { return $(`.card[data-col="${col}"]`); }

function resetCards(cols = COLS) {
  for (const c of cols) {
    S.cols[c] = { tokens: [], text: "", done: null, start: null };
    const st = $("[data-role=stream]", card(c));
    st.innerHTML = `<span class="placeholder">${S.cur ? "Ready — press Run." : "Waiting for a prompt."}</span>`;
    st._cur = null;
    $("[data-role=badges]", card(c)).innerHTML = "";
    $("[data-role=score]", card(c)).textContent = "";
  }
  if (cols.includes("instruct")) S.pendingInstruct = false;   // a queued Instruct rerun belongs to the outputs just cleared
  if (cols.includes("engine")) {
    S.scope = []; S.sel = null; S.hover = null;
    $("#inspector").innerHTML = `<div class="empty">Run a prompt, then click any engine token.</div>`;
    requestAnimationFrame(drawScope);
  }
  updateInstructHeader();
  markStale();
}

/* Instruct header from what ran (its start event), else from the run's request, else a preview of the toggles; so a
   mid-run toggle never relabels the card before it reruns. Checkpoint/hook sizes come from S.footprint when known. */
function updateInstructHeader() {
  const el = card("instruct"), st = S.cols.instruct || {}, ev = st.start, sn = st.snap || snapNow();
  const f = S.footprint && S.footprint[S.model], wd = (f && f.weight_delta) || {};
  if (wd.instruct_weights_bytes) $("[data-role=ckpt]", el).textContent = fmtBytes(wd.instruct_weights_bytes);
  if (f && f.artifact_bytes) $("[data-role=hook]", card("engine")).textContent = fmtBytes(f.artifact_bytes);
  const ban = ev ? ev.label.includes("our ban") : sn.ban;
  const cond = ev ? ev.condition : !sn.ban ? "12" : !sn.opts.modules.stem && sn.opts.gamma_unembed === 1.5 ? "13" : null;
  const title = ev ? ev.label : ban ? `Official Instruct + our ban${cond ? "" : " (stem)"}` : "Official Instruct (post-trained)";
  const t = $("[data-role=title]", el);
  t.textContent = title; t.title = title;
  $("[data-role=icond]", el).textContent = (ban ? " + 0-byte ban" : "") + " · " +
    (cond ? `cond ${cond.split("_")[0]}` : `cond 13 + ${sn.opts.modules.stem ? "stem ban" : `γ ${sn.opts.gamma_unembed}`} · not logged`);
  el.classList.toggle("hero", ban);
}

/* Does column c's output predate the current settings? Only the settings that reach that column count. */
function staleDiff(c, a, b) {
  const replayable = !!(S.cur && S.cur.pid && S.cols[c] && S.cols[c].start && S.cols[c].start.condition);   // only logged conditions replay
  if (a.maxNew !== b.maxNew || (a.replay !== b.replay && replayable)) return true;
  if (c === "engine") return a.kl !== b.kl || JSON.stringify(a.opts) !== JSON.stringify(b.opts);
  if (c === "instruct") return a.ban !== b.ban || a.frame !== b.frame ||
    (b.ban && (a.kl !== b.kl || a.opts.modules.stem !== b.opts.modules.stem || a.opts.gamma_unembed !== b.opts.gamma_unembed));
  return false;
}

/* Dim cards whose last run used other settings and offer a one-column re-run (A calls this from the drawer too). */
function markStale() {
  if (!S.cols.raw) return;                     // before initArena
  const now = snapNow();
  let any = false;
  for (const c of COLS) {
    const st = S.cols[c], stale = !!(st && st.snap && st.start && staleDiff(c, st.snap, now));
    any = any || stale;
    card(c).classList.toggle("stale", stale);
    $("[data-role=stale]", card(c)).hidden = !stale;
  }
  $("#run-btn").textContent = any ? "Re-run 4-way ▶" : "Run 4-way ▶";
}

function setStatus(msg, err = false) {
  const el = $("#status-line");
  el.textContent = msg || "";
  el.classList.toggle("err", err);
}

/* Stop the run in flight. user (the Stop button) also cancels on the server. Otherwise this is a pre-emption (new run,
   prompt or model): the abort plus the server's own pre-emption cancel it, and a late /api/cancel could hit the new run. */
function stopRun(user) {
  const R = S.running ? S.run : null;
  if (S.ctrl) S.ctrl.abort();
  S.ctrl = null; S.run = null; S.running = false;
  $("#run-btn").disabled = false;
  $("#stop-btn").disabled = true;
  if (user === true) {                         // kept until settled, so every run started meanwhile waits for it
    S.pendingInstruct = false;
    const p = S.cancelP = fetch("/api/cancel", { method: "POST" }).catch(() => {}).finally(() => { if (S.cancelP === p) S.cancelP = null; });
  }
  if (R) finishRun(R, user === true ? "stopped" : "preempted");
}

/* Settings a column ran with; markStale() compares them with the current ones. */
const snapNow = () => ({ maxNew: S.maxNew, kl: $("#exact-kl").checked, replay: $("#replay").checked, ban: $("#instruct-ban").checked,
  frame: $("#instruct-frame").value, opts: structuredClone(S.opts) });

/* Close out a run: partial columns get "stopped at N tok" + "incomplete · not scored", unstarted ones say so.
   reason: end | stopped | preempted | cancelled | error | closed (stream ended without `end`). */
function finishRun(R, reason) {
  for (const c of R.cols) {
    const st = S.cols[c], el = card(c);
    if (!st || st.run !== R || st.done || st.partial) continue;
    if (st.start) {
      st.partial = true;
      const pill = document.createElement("span");
      pill.className = "cap-pill";
      pill.textContent = `stopped at ${st.tokens.length} tok`;
      pill.title = "The run ended before this column finished";
      $("[data-role=stream]", el).appendChild(pill);
      $("[data-role=badges]", el).innerHTML = `<span class="badge warn" title="Partial output; not scored">incomplete · not scored</span>`;
    } else $("[data-role=stream]", el).innerHTML = `<span class="placeholder">Not run: the run ended before this column started.</span>`;
  }
  const msg = { stopped: "Stopped by you. Partial outputs are not scored.", preempted: "Previous run stopped.", cancelled: "Cancelled: a newer run (this tab or another window) took the GPU.",
    closed: "Stream closed before the run finished (server restarted?)." }[reason];
  if (msg) setStatus(msg, reason === "closed");
  markStale();
}

async function run(cols) {
  if (!S.cur) return;
  if (Z.ctrl) zenStop(false);                  // a zen run is pre-empted the same way
  if (S.ctrl) stopRun();                       // pre-empt silently; claim the run synchronously so a 2nd click aborts this one
  const mine = S.ctrl = new AbortController(), cancelP = S.cancelP, snap = snapNow();
  const R = S.run = { cols, t0: performance.now(), ended: false, failed: false, cancelled: false };
  if (cols.includes("instruct")) S.pendingInstruct = false;
  S.running = true;
  $("#run-btn").disabled = true;
  $("#stop-btn").disabled = false;
  resetCards(cols);
  for (const c of cols) { Object.assign(S.cols[c], { run: R, snap }); $("[data-role=stream]", card(c)).innerHTML = `<span class="placeholder">queued…</span>`; }
  markStale();
  setStatus("Starting…");
  const body = {
    model: S.model, prompt_id: S.cur.pid, instruction: S.cur.pid ? null : S.cur.instruction, columns: cols,
    instruct_ban: snap.ban, instruct_frame: snap.frame, modules: snap.opts.modules, kappa_boundary: snap.opts.kappa_boundary,
    gamma_unembed: snap.opts.gamma_unembed, beta_attn: snap.opts.beta_attn, max_new_tokens: snap.maxNew, reference_kl: snap.kl, replay: snap.replay,
  };
  let res;
  try {
    if (cancelP) await cancelP;                // a Stop pressed just before must reach the server before this run does
    if (S.ctrl !== mine) return;
    res = await fetch("/api/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: mine.signal });
    if (S.ctrl !== mine) return;
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (S.ctrl === mine) {
      let r;
      try { r = await reader.read(); } catch (e) { if (e.name === "AbortError") throw e; break; }   // body dropped mid-stream: finally says "closed"
      const { value, done } = r;
      if (done || S.ctrl !== mine) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while (S.ctrl === mine && (i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (S.ctrl === mine && line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)), R);
      }
    }
  } catch (e) {                                // HTTP or rendering error in our own run: report it and stop the server generating
    if (e.name !== "AbortError" && S.ctrl === mine) { R.failed = true; setStatus(String(e.message || e), true); mine.abort(); }
  } finally {
    if (S.ctrl === mine) {                     // own run only: a pre-empted run's late finally must not touch the new run
      S.ctrl = null; S.run = null; S.running = false;
      $("#run-btn").disabled = false;
      $("#stop-btn").disabled = true;
      finishRun(R, R.ended ? "end" : R.cancelled ? "cancelled" : R.failed ? "error" : "closed");
      scheduleScope();
      const ist = S.cols.instruct, again = S.pendingInstruct && R.ended && S.cur && (!ist.snap || staleDiff("instruct", ist.snap, snapNow()));
      S.pendingInstruct = false;
      if (again) run(["instruct"]);
    }
  }
}

function onEvent(ev, R) {
  const swapWait = () => { if (R.cols.includes("instruct") && !S.cols.instruct.start) $("[data-role=stream]", card("instruct")).innerHTML = `<span class="placeholder">Waiting: Instruct loads after the base columns finish (memory policy: swap).</span>`; };
  switch (ev.type) {
    case "status": setStatus(ev.msg); if (ev.msg.includes("no cached log")) R.noLog = true; if (ev.msg.includes("swap")) swapWait(); break;
    case "error": R.failed = true; setStatus(ev.msg, true); break;
    case "cancelled": R.cancelled = true; setStatus("Cancelled: a newer run (this tab or another window) took the GPU."); break;
    case "start": onStart(ev); break;
    case "token":                              // every phase-1 column has started by the first token; Instruct missing = swap policy
      if (!R.tok1) { R.tok1 = true; if (!R.noLog) swapWait(); }
      onToken(ev); break;
    case "done": onDone(ev); break;
    case "end": R.ended = true; setStatus(`Done in ${((performance.now() - R.t0) / 1000).toFixed(1)} s · decoders interleaved token by token on one GPU; tok/s counts each column's own forward passes.`); break;
    default: break;
  }
}

/* "Streaming N columns…" while any column of this run is mid-stream. */
function streamingStatus() {
  const live = COLS.filter((c) => S.run && S.cols[c] && S.cols[c].run === S.run && S.cols[c].start && !S.cols[c].done);
  if (live.length) setStatus(`Streaming ${live.length} column${live.length > 1 ? "s" : ""}${live.some((c) => S.cols[c].start.reference_kl) ? " (with exact step-KL)" : ""}…`);
}

function onStart(ev) {
  const c = ev.col, st = S.cols[c];
  st.start = ev;
  const el = card(c);
  $("[data-role=stream]", el).innerHTML = "";
  if (c === "engine") {
    $("[data-role=title]", el).textContent = ev.label;
    $("[data-role=title]", el).title = ev.label;  // ellipsised on narrow cards
    const chip = $("[data-role=cond]", el);
    chip.textContent = ev.condition ? "cond 11 · logged" : (ev.label.includes("v2.3") ? "v2.3 demo" : "custom");
    S.promptView.engine = ev.prompt_view;
    S.artifact = ev.artifact;
    $("#drift-layer").textContent = ev.artifact ? ev.artifact.stop_layer : "–";
    renderCfgDiff(ev.config_diff);
  } else if (c === "instruct") {
    S.promptView.instruct = ev.prompt_view;
    updateInstructHeader();                    // relabel from what actually runs (label + condition), not the toggles
  }
  if (ev.source === "cached log") $("[data-role=badges]", el).innerHTML = `<span class="badge neutral">replaying cached log…</span>`;
  streamingStatus();
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
    el.addEventListener("click", () => { stream._cur = idx; inspect(c, idx); }); // the keyboard cursor continues from a clicked token
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
  const score = $("[data-role=score]", card(c));
  score.textContent = `${(100 * m.composite).toFixed(0)}%`;
  score.title = "Composite = mean of the four checks";
  const html = [
    b(m.turn_cross, "TurnCross"),
    m.n_checks ? b(m.constraint, "Constraint", detail) : `<span class="badge neutral" title="No machine-checkable rule found in this custom prompt">Constraint n/a</span>`,
    b(m.clean_stop, "CleanStop", (m.contamination || []).join(", ")),
    b(m.fluency, "Fluency"),
  ];
  if (ev.stem_strict != null) html.push(b(ev.stem_strict, "stem-strict", "No word starts with a banned stem (stricter than the frozen verifier)"));
  const [sr, srTitle] = ev.stop_reason === "eos" ? ["EOS", "Emitted end-of-sequence"] : ev.stop_reason === "stop_string" ? ["Stop string", "Cut at a stop string"] : ["Cap", `Hit the ${ev.n_new_tokens}-token cap`];
  html.push(`<span class="fbreak"></span>`);     // checks on row 1, metadata on row 2
  html.push(`<span class="badge ${ev.stop_reason === "max_tokens" ? "warn" : "neutral"}" title="${srTitle}">${sr}</span>`);
  html.push(`<span class="badge neutral">${ev.n_new_tokens} tok</span>`);
  if (ev.tok_s) html.push(`<span class="badge neutral" title="${ev.source === "cached log" ? "logged speed" : "own forward passes, interleaved with the other columns"}">${ev.tok_s.toFixed(1)} tok/s</span>`);
  if (ev.logged) html.push(`<span class="badge ${ev.logged.identical ? "pass" : "warn"}" title="${ev.logged.identical ? "Byte-identical to" : "Differs from"} the cached ${esc(ev.logged.condition)} output">${ev.logged.identical ? "matches log ✔" : "≠ logged"}</span>`);
  const sl = st.start || {};
  if (c === "engine" && !ev.logged && sl.label && sl.condition == null) html.push(sl.label.includes("v2.3")
    ? `<span class="badge neutral" title="v2.3 = cond 11 + stem ban, not a logged condition. Ablation controls → Exactly condition 11 reproduces the log byte for byte.">no log (v2.3)</span>`
    : `<span class="badge neutral" title="A custom ablation is not a logged condition, so there is no cached output to compare with.">no log (custom)</span>`);
  if (ev.source === "cached log") html.push(`<span class="badge neutral">cached log</span>`);
  const tr = ev.engine_trace || {};
  if (tr.stem_guard_steps) html.push(`<span class="badge neutral" title="Steps where the sub-word guard masked stem completions">guard ×${tr.stem_guard_steps}</span>`);
  $("[data-role=badges]", card(c)).innerHTML = html.join("");
  if (c === "engine") scheduleScope();
  streamingStatus();
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
  const plotW = W - SC.L - SC.R, xR = W - SC.R, px = (v) => Math.round(v) + 0.5; // px: 1px hairlines on whole device pixels (dpr 1 and 2)
  g.font = `11px ${col("--sans")}`;
  // titles: the long form when it fits, else drop the parentheticals, else ellipsis
  const fit = (s, w) => { if (g.measureText(s).width <= w) return s; while (s.length > 1 && g.measureText(s + "…").width > w) s = s.slice(0, -1); return s.trimEnd() + "…"; };
  const trackLabel = (y, long, short) => { g.fillStyle = col("--muted"); g.textAlign = "left"; g.fillText(g.measureText(long).width <= plotW - 4 ? long : fit(short, plotW - 4), SC.L + 2, y - 5); };
  const frame = ([y, h]) => { g.strokeStyle = "rgba(148,163,184,0.16)"; g.lineWidth = 1; g.strokeRect(SC.L + 0.5, y + 0.5, plotW - 1, h - 1); };
  const hline = (y, a = 0.08) => { g.strokeStyle = `rgba(148,163,184,${a})`; g.lineWidth = 1; g.beginPath(); g.moveTo(SC.L, px(y)); g.lineTo(xR, px(y)); g.stroke(); };
  // shared time axis: tick pitch ≥ 44px; boundary-window runs + tick hairlines sit under every track's data
  const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find((s) => bw * s >= 44) || 500;
  const xt = [], runs = [];
  for (let i = 0; i < d.length; i += step) xt.push(i);
  d.forEach((t, i) => { if (!t.in_boundary_window) return; const r = runs[runs.length - 1]; if (r && r[1] === i - 1) r[1] = i; else runs.push([i, i]); });
  const under = ([y, h], a) => {
    g.fillStyle = `rgba(148,163,184,${a})`; for (const [i0, i1] of runs) g.fillRect(x(i0) - bw / 2, y, (i1 - i0 + 1) * bw, h);
    g.strokeStyle = "rgba(148,163,184,0.08)"; g.lineWidth = 1; g.beginPath(); for (const i of xt) { g.moveTo(px(x(i)), y); g.lineTo(px(x(i)), y + h); } g.stroke();
  };

  // --- Track A: drift z vs hinge threshold
  const [ay, ah] = SC.A;
  const zs = d.map((t) => t.drift_z).filter((v) => v != null);
  const ks = d.map((t) => t.hinge_threshold).filter((v) => v != null);
  const zmax = Math.max(3.6, ...zs, ...ks) + 0.4, zmin = Math.min(-3, ...zs) - 0.3;
  const yA = (v) => ay + ah - ((v - zmin) / (zmax - zmin)) * ah;
  under(SC.A, 0.09);
  const zstep = [1, 2, 5, 10].find((s) => (s * ah) / (zmax - zmin) >= 18) || 10, zt = []; // integer-σ ticks, ≥18px apart
  for (let v = Math.ceil(zmin / zstep) * zstep; v <= zmax; v += zstep) zt.push(v || 0);
  zt.forEach((v) => hline(yA(v), v === 0 ? 0.16 : 0.06));
  d.forEach((t, i) => {
    if (t.drift_z != null && t.hinge_threshold != null && t.drift_z > t.hinge_threshold) {
      g.fillStyle = "rgba(251,191,36,0.45)";
      g.fillRect(x(i) - bw / 2, yA(t.drift_z), bw, yA(t.hinge_threshold) - yA(t.drift_z));
    }
  });
  // kappa dashed step line
  g.setLineDash([4, 3]); g.strokeStyle = "rgba(226,232,240,0.55)"; g.lineWidth = 1; g.beginPath(); // reference, not the hero mark
  let started = false;
  d.forEach((t, i) => {
    if (t.hinge_threshold == null) { started = false; return; }
    const y = px(yA(t.hinge_threshold)), x0 = x(i) - bw / 2, x1 = x(i) + bw / 2;
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
  trackLabel(ay, "A · drift z(t) vs hinge κ(t)  (σ units; grey bands = boundary window)", "A · drift z(t) vs hinge κ(t)");
  axisLabels(g, yA, zt, "σ", true);

  // --- Track B: steering schedule + completion cosine
  const [by, bh] = SC.B;
  const nmax = Math.max(5, ...d.map((t) => (t.init_norm || 0) + (t.sustain_norm || 0)));
  const yB = (v) => by + bh - (v / nmax) * bh;
  under(SC.B, 0.05);
  d.forEach((t, i) => {
    if (t.init_norm > 0) { g.fillStyle = "rgba(74,222,128,0.75)"; g.fillRect(x(i) - Math.max(bw * 0.35, 1), yB(t.init_norm), Math.max(bw * 0.7, 2), by + bh - yB(t.init_norm)); }
  });
  g.strokeStyle = col("--eng"); g.lineWidth = 1.6; g.beginPath(); started = false;
  d.forEach((t, i) => { if (t.sustain_norm == null) return; const y = yB(t.sustain_norm); if (!started) { g.moveTo(x(i), y); started = true; } else g.lineTo(x(i), y); });
  g.stroke();
  const yE = (v) => by + bh - ((v + 0.4) / 1.4) * bh;
  if (S.artifact) { const ty = px(yE(S.artifact.tau_stop)); g.setLineDash([2, 3]); g.strokeStyle = "rgba(56,189,248,0.7)"; g.lineWidth = 1; g.beginPath(); g.moveTo(SC.L, ty); g.lineTo(xR, ty); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = col("--eos"); g.lineWidth = 1.3; g.beginPath(); started = false;
  d.forEach((t, i) => { if (t.eos_cos == null) { started = false; return; } const y = yE(t.eos_cos); if (!started) { g.moveTo(x(i), y); started = true; } else g.lineTo(x(i), y); });
  g.stroke();
  d.forEach((t, i) => { if (t.stop_events && t.stop_events.some((e) => e.reason)) { g.fillStyle = col("--eos"); g.beginPath(); g.arc(x(i), by + 6, 3, 0, 7); g.fill(); } if (t.eos) { g.fillStyle = col("--eos"); g.fillRect(x(i) - 1, by, 2, bh); } });
  frame(SC.B);
  trackLabel(by, "B · ‖init + M3‖ (bars) · ‖sustain‖ · cos(h, μ_eos) vs τ_stop (dotted)", "B · ‖init + M3‖ · ‖sustain‖ · cos(h, μ_eos) vs τ_stop");
  axisLabels(g, yB, [0, Math.round(nmax)], "", true);
  axisLabels(g, yE, [-0.4, 0, 1], "", false, W);
  // unit words at mid-track (nudged off any tick); τ_stop labelled at its own height in the cos-axis colour
  const tauY = S.artifact ? yE(S.artifact.tau_stop) + 3 : null, rTicks = [-0.4, 0, 1].map((v) => yE(v) + 3).concat(tauY ?? []);
  const unitY = (ticks) => [0.5, 0.3, 0.7].map((f) => by + bh * f + 3).find((y) => ticks.every((t) => Math.abs(t - y) >= 12)) ?? by + bh / 2 + 3;
  g.fillStyle = col("--muted"); g.textAlign = "right"; g.fillText("‖Δh‖", SC.L - 5, unitY([yB(0) + 3, yB(Math.round(nmax)) + 3]));
  g.textAlign = "left"; g.fillText("cos", xR + 5, unitY(rTicks));
  if (S.artifact) { g.fillStyle = col("--eos"); g.fillText(`τ ${fmt(S.artifact.tau_stop, 2)}`, xR + 5, tauY); }

  // --- Track C: per-token KL
  const [cy, ch] = SC.C;
  const kls = d.map((t) => t.step_kl_nats ?? t.lexical_kl_nats ?? 0);
  const kmax = Math.max(0.5, ...kls);
  const yC = (v) => cy + ch - (Math.sqrt(Math.max(v, 0)) / Math.sqrt(kmax)) * ch;
  under(SC.C, 0.05);
  const kt = [0.1, 1, 10].filter((v) => v < kmax);
  kt.forEach((v) => hline(yC(v)));
  d.forEach((t, i) => {
    const v = kls[i];
    if (!v) return;
    g.fillStyle = t.ban_flipped ? "rgba(192,132,252,0.95)" : "rgba(148,163,184,0.55)"; // grey so the ban flips carry the colour
    g.fillRect(x(i) - Math.max(bw * 0.38, 0.8), yC(v), Math.max(bw * 0.76, 1.6), cy + ch - yC(v));
  });
  frame(SC.C);
  const exact = d.some((t) => t.step_kl_nats != null);
  trackLabel(cy, `C · ${exact ? "exact" : "lexical-only"} per-token KL vs. unsteered (√ scale; purple = ban flipped the top token)`, `C · ${exact ? "exact" : "lexical-only"} per-token KL vs. unsteered`);
  // √ scale crowds the low ticks: place 0 and max first, then 10 / 1 / 0.1 only where ≥9px clear
  const kl = [];
  for (const v of [0, Math.round(kmax * 10) / 10, ...kt.reverse()]) if (kl.every((u) => Math.abs(yC(u) - yC(v)) >= 9)) kl.push(v);
  axisLabels(g, yC, kl, "", true);

  // cursor + selection
  for (const [idx, color] of [[S.hover, "rgba(226,232,240,0.5)"], [S.sel, "rgba(45,212,191,0.9)"]]) {
    if (idx == null || idx >= d.length) continue;
    g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(px(x(idx)), 4); g.lineTo(px(x(idx)), cy + ch); g.stroke();
  }
  g.fillStyle = col("--muted"); g.textAlign = "center";
  let tx = xR + 5;
  for (const i of xt) { g.fillText(String(i), x(i), cy + ch + 12); tx = Math.max(tx, x(i) + g.measureText(String(i)).width / 2 + 5); }
  if (xt.length) { g.textAlign = "left"; g.font = `italic 11px ${col("--sans")}`; g.fillText("t", tx, cy + ch + 12); }
  const lbl = d.length ? `Engine telemetry over ${d.length} steps: ${d.filter((t) => t.hinge_layers_clamped > 0).length} hinge clamps, ${d.filter((t) => t.ban_flipped).length} ban flips` : "Engine telemetry: no engine tokens yet";
  if (cv.getAttribute("aria-label") !== lbl) cv.setAttribute("aria-label", lbl);
  const emp = d.length && $("#inspector > .empty"), hint = "Click any engine token, or a point on the scope, to see what M1–M4 and the ban did at that step.";
  if (emp && emp.textContent !== hint) emp.textContent = hint;
  if (!d.length) {
    g.fillStyle = col("--faint"); g.textAlign = "center"; g.font = `13px ${col("--sans")}`;
    const msg = S.running ? "waiting for engine tokens…" : "telemetry appears here while the engine streams", cx = SC.L + plotW / 2, my = ay + ah / 2 + 30;
    if (g.measureText(msg).width <= plotW - 16) g.fillText(msg, cx, my); else { const w = msg.split(" "), k = Math.ceil(w.length / 2); g.fillText(w.slice(0, k).join(" "), cx, my - 8); g.fillText(w.slice(k).join(" "), cx, my + 9); } // wrap inside the plot at 390
  }
}

function axisLabels(g, yf, vals, unit, left, W) {
  g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(); g.textAlign = left ? "right" : "left";
  for (const v of vals) g.fillText(`${String(v).replace("-", "−")}${unit}`, left ? SC.L - 5 : W - SC.R + 5, yf(v) + 3);
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
  tip.innerHTML = `<b>${esc(showTok(t.eos ? "<EOS>" : t.s))}</b> · t=${t.t}${t.ban_flipped ? " · <span style='color:var(--ban)'>ban flip</span>" : ""}<br>z ${fmt(t.drift_z)} / κ ${fmt(t.hinge_threshold)}${t.in_boundary_window ? " (boundary)" : ""}<br>` +
    `init ${fmt(t.init_norm, 1)} · sustain ${fmt(t.sustain_norm, 1)}<br>cos ${fmt(t.eos_cos)} · KL ${fmt(t.step_kl_nats ?? t.lexical_kl_nats, 3)} nats`; // short lines so it fits beside the cursor at 390
  tip.style.display = "block";
  // flip left of the cursor past mid-width; bottom-align on track C so track A's data stays clear
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = `${Math.max(0, Math.min(mx > r.width / 2 ? mx - 12 - tw : mx + 12, r.width - tw))}px`;
  tip.style.top = `${Math.max(SC.A[0] + SC.A[1] + 4, SC.C[0] + SC.C[1] - th)}px`;
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
  const ins = $("#inspector");
  ins.innerHTML = h;
  const ir = ins.getBoundingClientRect(); // off-screen (e.g. top ≈ 1179 in a 900px viewport at 1440): bring it into view
  if (ir.top > innerHeight - 60 || ir.bottom < 60) ins.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function probCell(p, after) {
  const w = Math.max(1, Math.round(60 * Math.min(1, Math.sqrt(p || 0))));
  return `<span class="pbar${after ? " after" : ""}" style="width:${w}px"></span> ${(100 * (p || 0)).toFixed(p < 0.001 ? 4 : 1)}%`;
}

/* ============================== drawer ============================== */
function initDrawer() {
  const dr = $("#drawer"), ob = $("#open-drawer");
  dr.inert = true; ob.setAttribute("aria-controls", "drawer"); ob.setAttribute("aria-expanded", "false");
  ob.addEventListener("click", () => setDrawer(!dr.classList.contains("open")));
  $("#close-drawer").addEventListener("click", () => setDrawer(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && dr.classList.contains("open") && (dr.contains(e.target) || e.target === document.body)) setDrawer(false); });
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

/* Non-modal drawer: inert (no tab stops, hidden from AT) while closed; focus goes in on open and back to the opener on close. */
function setDrawer(open) {
  const dr = $("#drawer"), ob = $("#open-drawer");
  if (dr.classList.contains("open") === open) return;
  const inside = dr.contains(document.activeElement);
  dr.classList.toggle("open", open);
  dr.inert = !open;
  ob.setAttribute("aria-expanded", open);
  if (open) $("#close-drawer").focus({ preventScroll: true });
  else if (inside) ob.focus({ preventScroll: true });   // only pull focus back if it was in the drawer
}

function markOptsDirty() {
  const m = S.opts.modules;
  const def = m.lifecycle && m.boundary_hinge && m.predictor && m.pin_attention && m.unembed_ban && S.opts.kappa_boundary === 1.5 && S.opts.gamma_unembed === 1.5 && S.opts.beta_attn === 0.5;
  $("#cfg-diff").innerHTML = def ? (m.stem ? "Engine = condition 11 + stem morphology (v2.3 demo)." : "Engine = condition 11 exactly (its logged outputs are reproduced byte for byte).") :
    "Custom ablation — press “Re-run engine only”.";
  const D = { kappa_boundary: 1.5, gamma_unembed: 1.5, beta_attn: 0.5 }; // v2.3 demo defaults (all modules on)
  const n = Object.values(m).filter((x) => !x).length + Object.keys(D).filter((k) => S.opts[k] !== D[k]).length;
  $("#open-drawer").textContent = n ? `Ablation controls · ${n} changed` : "Ablation controls";
  if (typeof markStale === "function") markStale();
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
    const press = (sel, b) => $$(`${sel} button`).forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", x === b); });
    $$("#budget-switch button").forEach((b) => b.addEventListener("click", () => { S.eco.budget = b.dataset.b; press("#budget-switch", b); loadEconomics(); }));
    $$("#wf-sort button").forEach((b) => b.addEventListener("click", () => { S.eco.sort = b.dataset.s; press("#wf-sort", b); loadEconomics(); }));
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
  const rtxt = (k) => { const r = rec(k); if (!r || r.point == null) return "–"; if (r.undefined_frac > 0.025) return `<span class="nr">not resolved</span>`; return `${Math.round(100 * r.point)}%`; };
  const rci = (k) => { const r = rec(k); return r && r.lo != null && r.undefined_frac <= 0.025 ? `[${Math.round(100 * r.lo)}%, ${Math.round(100 * r.hi)}%]` : "denominator unresolved"; };
  const kE = eng && eng.kl.heldout_no_negation, kI = ins && ins.kl.heldout_no_negation;
  const mins = eco.calibration_seconds ? `${Math.round(eco.calibration_seconds / 60)} min` : "not recorded";
  const card = (label, ev, es, iv, is) => `<div class="kpi glass"><div class="kpi-label">${label}</div><div class="kpi-vs">
      <div><div class="kpi-v eng">${ev}</div><div class="kpi-s">${es}</div></div><div><div class="kpi-v ins">${iv}</div><div class="kpi-s">${is}</div></div></div></div>`;
  $("#kpis").innerHTML = `<div class="kpi-key"><span><i class="eng"></i>Zero-weight engine</span><span><i class="ins"></i>Official Instruct</span></div>
    <div class="kpi glass kpi-wide"><div class="kpi-label">Post-training recovery <span class="f">(C_engine − C_raw) / (C_Instruct − C_raw)</span>, paired bootstrap 95% CI</div>
    <div class="kpi-vs kpi-vs4">
      <div><div class="kpi-v eng">${rtxt("10_engine_v2_1@512")}</div><div class="kpi-s">v2.1 @ 512 tok ${rci("10_engine_v2_1@512")}</div></div>
      <div><div class="kpi-v eng">${rtxt("11_engine_v2_2_lean@512")}</div><div class="kpi-s">v2.2 lean @ 512 tok ${rci("11_engine_v2_2_lean@512")}</div></div>
      <div><div class="kpi-v" style="color:var(--muted)">${rtxt("10_engine_v2_1@160")}</div><div class="kpi-s">v2.1 @ 160 tok · ${rci("10_engine_v2_1@160")}</div></div>
      <div><div class="kpi-v" style="color:var(--muted)">${rtxt("11_engine_v2_2_lean@160")}</div><div class="kpi-s">v2.2 lean @ 160 tok · Instruct's own gain is not resolved at 160 tokens</div></div>
    </div></div>` + [
    card("Weights modified", "0%", "engine (hooks only)", pct(100 * (wd.params_changed_frac || 0)), `Instruct · of ${((wd.params || 0) / 1e9).toFixed(2)}B params`),
    card("Disk artifact", fmtBytes(eco.artifact_bytes), "calibration file", fmtBytes(wd.instruct_weights_bytes), `whole checkpoint · ×${Math.round((wd.instruct_weights_bytes || 0) / (eco.artifact_bytes || 1))}`),
    card("One-off cost", mins, "forward passes, 80 prompts, M4 Pro", "SFT + RL", "> 1M SFT examples + multi-stage RL"),
    card("Extra prompt tokens", "0", `URIAL needs ${Math.round(urial ? urial.extra_prompt_tokens : 768)}`, `${Math.round(ins ? ins.extra_prompt_tokens : 0)}`, "chat template + system prompt"),
    card("Tokens per answer @512", eng ? Math.round(eng.avg_tokens_512) : "–", e10 ? `v2.1: ${Math.round(e10.avg_tokens_512)}` : "", ins ? Math.round(ins.avg_tokens_512) : "–", eng && ins ? `${(ins.avg_tokens_512 / eng.avg_tokens_512).toFixed(1)}× longer` : ""),
    card("KL tax (held-out, no-neg.)", kE ? fmt(kE.mean_kl, 3) : "–", kE ? `nats/token · top-1 ${pct(100 * kE.top1_agree)}` : "", kI ? fmt(kI.mean_kl, 3) : "–", kE && kI ? `${(kI.mean_kl / kE.mean_kl).toFixed(0)}× more · top-1 ${pct(100 * kI.top1_agree)}` : ""),
  ].join("");
}

const KIND_COLOR = { base: "#94a3b8", prompt: "#fbbf24", steer: "#fb923c", engine: "#2dd4bf", instruct: "#8b9cfb" };
const KIND_LABEL = { base: "Raw base", prompt: "Prompting (URIAL)", steer: "Static steering (CAA)", engine: "Zero-weight engine", instruct: "Post-trained (Instruct)" };
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
  const mc = (renderScatter._mc ||= document.createElement("canvas").getContext("2d"));
  mc.font = `12px ${getComputedStyle($("#scatter")).fontFamily}`;
  const placed = [], bub = [], labs = [], tw = (t) => mc.measureText(t).width + 2; // obstacle boxes {x0,x1,y0,y1}; tw = width of 12px label text
  let seg = []; // points sampled along the annotation arrow, so labels can dodge it
  if (eng && ins && eng.x > 0) {
    const note = `${(ins.x / eng.x).toFixed(0)}× less distortion, ${eng.y >= ins.y ? "+" : ""}${(eng.y - ins.y).toFixed(1)} Composite`, ax = (X(ins.x) + X(eng.x)) / 2, ay = Math.min(Y(ins.y), Y(eng.y)) - 12;
    placed.push({ x0: ax - tw(note) / 2, x1: ax + tw(note) / 2, y0: ay - 10, y1: ay + 3 });
    const [sx, sy, ex, ey] = [X(ins.x) - 18, Y(ins.y) + 6, X(eng.x) + 16, Y(eng.y) + 4], n = Math.ceil(Math.hypot(ex - sx, ey - sy) / 3);
    seg = Array.from({ length: n + 1 }, (_, i) => [sx + ((ex - sx) * i) / n, sy + ((ey - sy) * i) / n]);
    s += `<line x1="${X(ins.x) - 18}" y1="${Y(ins.y) + 6}" x2="${X(eng.x) + 16}" y2="${Y(eng.y) + 4}" stroke="rgba(226,232,240,0.35)" stroke-dasharray="3 3" marker-end="url(#arr)"/>
      <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="rgba(226,232,240,0.5)"/></marker></defs>
      <text class="lbl val" x="${ax}" y="${ay}" text-anchor="middle">${note}</text>`;
  }
  for (const p of pts.sort((a, b) => (b.r.bytes || 0) - (a.r.bytes || 0))) {
    const rr = 5 + 3 * Math.log10(1 + (p.r.bytes || 0) / 1e6);
    const cx = X(p.x), cy = Y(p.y);
    s += `<circle class="halo" cx="${cx}" cy="${cy}" r="${rr + 2}"/><circle cx="${cx}" cy="${cy}" r="${rr}" fill="${KIND_COLOR[p.r.kind]}" fill-opacity="0.28" stroke="${KIND_COLOR[p.r.kind]}" stroke-width="1.5"><title>${esc(p.r.label)}: KL ${fmt(p.x, 3)}, top-1 ${pct(100 * p.top1)}, Composite ${fmt(p.y, 1)}, ${fmtBytes(p.r.bytes) === "–" ? "no weights" : fmtBytes(p.r.bytes)}</title></circle>`;
    bub.push({ p, rr, cx, cy });
    placed.push({ x0: cx - rr - 2, x1: cx + rr + 2, y0: cy - rr - 2, y1: cy + rr + 2 });
  }
  // labels right of each bubble (left near the edge), placed top→bottom; a label nudges ±13 px to clear labels, bubbles and the
  // arrow, but never above the label of a higher point, so label order matches point order. A moved label gets a leader.
  const hit = (b) => placed.some((q) => q.x0 < b.x1 && b.x0 < q.x1 && q.y0 < b.y1 && b.y0 < q.y1) || seg.some(([x, y]) => x > b.x0 && x < b.x1 && y > b.y0 && y < b.y1);
  for (const { p, rr, cx, cy } of [...bub].sort((a, b) => a.cy - b.cy)) {
    const w = tw(`${p.r.label} ${fmt(p.y, 1)}%`), end = cx + rr + 6 + w > W - 4;
    const lx = end ? cx - rr - 6 : cx + rr + 6, x0 = end ? lx - w : lx, x1 = end ? lx : lx + w, box = (y) => ({ x0, x1, y0: y - 10, y1: y + 3 });
    const ok = (y) => y > T + 6 && y < H - B - 2 && !labs.some((q) => q.x0 < x1 && x0 < q.x1 && q.ly > y) && !hit(box(y));
    const ly = [0, -6.5, 6.5, -13, 13, -19.5, 19.5, -26, 26, 32.5, 39, 45.5, 52, 65, 78, 91].map((d) => cy + 4 + d).find(ok) ?? cy + 4;
    placed.push(box(ly)); labs.push({ x0, x1, ly });
    if (Math.abs(ly - cy - 4) > 4) s += `<line class="lead" x1="${end ? cx - rr : cx + rr}" y1="${cy}" x2="${end ? lx + 2 : lx - 2}" y2="${ly - 4}"/>`;
    s += `<text class="lbl val" x="${lx}" y="${ly}" text-anchor="${end ? "end" : "start"}">${esc(p.r.label)} <tspan class="lbl dim">${fmt(p.y, 1)}%</tspan></text>`;
  }
  s += `</svg>`;
  $("#scatter").innerHTML = s;
  const kinds = Object.keys(KIND_LABEL).filter((k) => pts.some((p) => p.r.kind === k));
  $("#scatter-key").innerHTML = kinds.map((k) => `<span><i style="--c:${KIND_COLOR[k]}"></i>${KIND_LABEL[k]}</span>`).join("");
  $("#scatter-note").textContent = `Bubble radius ∝ log₁₀ of the MB changed or shipped. ${missing.length ? `Not shown (no ${budget === "512" ? "512-token run" : "KL measurement"}): ${missing.join(", ")}.` : ""} ${eco.kl_note}`;
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

const GROUP_COLOR = { "Baselines": "#94a3b8", "Phase 1 · v1": "#0f766e", "Phase 2 · v2": "#0d9488", "Phase 3 · v2.1": "#14b8a6", "Phase 4 · lean": "#2dd4bf", "Phase 4 · post-trained": "#8b9cfb" };
// baseline rows take their approach colour (raw slate, URIAL amber, CAA orange); engine phases ride an ordinal teal ramp
const wfColor = (r) => (r.group === "Baselines" ? (/urial/.test(r.condition) ? KIND_COLOR.prompt : /caa/.test(r.condition) ? KIND_COLOR.steer : KIND_COLOR.base) : GROUP_COLOR[r.group] || KIND_COLOR.base);

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
  let wsk = "", chev = "";
  rows.forEach((r, i) => {
    const y = T + i * rowH;
    const c = wfColor(r);
    const x0 = X(Math.min(raw, r.composite)), x1 = X(Math.max(raw, r.composite));
    s += `<text class="lbl${/^1[01]_|^12_official_instruct_chat$/.test(r.condition) ? " key" : ""}" x="${L - 8}" y="${y + 13}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<rect x="${x0}" y="${y + 3}" width="${Math.max(x1 - x0, 1.5)}" height="13" rx="2" fill="${c}" fill-opacity="${r.composite >= raw ? 0.8 : 0.35}"><title>${r.condition}: ${fmt(r.composite, 1)} (Δ ${fmt(r.delta_vs_raw, 1)}, 95% CI [${fmt(r.ci_vs_raw[0], 1)}, ${fmt(r.ci_vs_raw[1], 1)}])</title></rect>`;
    if (r.condition !== "1_raw_base") {
      const a = X(raw + r.ci_vs_raw[0]), b = X(raw + r.ci_vs_raw[1]);
      wsk += `<line x1="${a}" x2="${b}" y1="${y + 9.5}" y2="${y + 9.5}"/><line x1="${a}" x2="${a}" y1="${y + 6}" y2="${y + 13}"/><line x1="${b}" x2="${b}" y1="${y + 6}" y2="${y + 13}"/>`;
      if (raw + r.ci_vs_raw[1] > hi) chev += `<text class="chev" x="${W - R - 5}" y="${y + 14}" text-anchor="middle"><title>CI runs past 100</title>›</text>`;
    }
    s += `<text class="val" x="${W - R + 8}" y="${y + 13}">${fmt(r.composite, 1)} <tspan class="${r.delta_vs_raw > 0 ? "up" : r.delta_vs_raw < 0 ? "dn" : ""}">${r.condition === "1_raw_base" ? "" : (r.delta_vs_raw >= 0 ? "+" : "") + fmt(r.delta_vs_raw, 1)}</tspan></text>`;
  });
  // whiskers are clipped to the plot so CIs past 100 cannot run into the value column; a › marks the clipped end
  s += `<defs><clipPath id="wf-plot"><rect x="${L}" y="0" width="${W - L - R}" height="${H}"/></clipPath></defs><g class="wsk" clip-path="url(#wf-plot)">${wsk}</g>${chev}</svg>`;
  $("#waterfall").innerHTML = s;
  const sw = [["raw", KIND_COLOR.base], ["URIAL", KIND_COLOR.prompt], ["CAA", KIND_COLOR.steer], ...Object.entries(GROUP_COLOR).slice(1)];
  $("#wf-note").innerHTML = `<span class="wf-key">${sw.map(([l, c]) => `<span><i style="background:${c}"></i>${esc(l)}</span>`).join("")}</span>${esc(ab.model)}: bar = Composite relative to the raw base (faded = below it); whisker = 95% paired-bootstrap CI of the difference (CIs that cross the dashed line are not resolved; › = CI runs past 100). ${esc(ab.note)}`;
}

function renderAblationTable(ab) {
  const cols = [["label", "Condition", "l"], ["group", "Phase", "l"], ["composite", "Composite"], ["delta_vs_raw", "Δ raw [95% CI]"], ["turn_cross", "Turn"], ["constraint", "Constr."], ["clean_stop", "Stop"], ["fluency", "Flu."], ["natural_eos", "EOS"],
    ...ab.categories.map((c) => [`cat:${c}`, CAT_LABEL[c] || c]), ["composite_512", "512 tok"], ["composite_sdpa", "SDPA"], ["avg_new_tokens", "tok/ans"], ["tok_per_s", "tok/s"]];
  const st = S.eco.tableSort || { k: null, asc: false };
  let rows = [...ab.rows];
  const val = (r, k) => (k.startsWith("cat:") ? r.categories[k.slice(4)] : r[k]);
  if (st.k) rows.sort((a, b) => { const va = val(a, st.k), vb = val(b, st.k); const c = typeof va === "string" ? va.localeCompare(vb) : (va ?? -1) - (vb ?? -1); return st.asc ? c : -c; });
  const heat = (v) => v == null ? "–" : `<span class="heat" style="background:rgba(226,232,240,${(0.04 + 0.28 * Math.min(1, Math.max(0, (v - 60) / 40))).toFixed(2)})">${fmt(v, 0)}</span>`;
  const head = `<tr>${cols.map(([k, l, cls]) => `<th data-k="${k}" class="${cls || ""}${st.k === k ? ` sorted${st.asc ? " asc" : ""}` : ""}"${st.k === k ? ` aria-sort="${st.asc ? "ascending" : "descending"}"` : ""}><button type="button" class="th-sort">${l}</button></th>`).join("")}</tr>`;
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
    $(`th[data-k="${k}"] .th-sort`, t).focus();
  }));
}

/* ============================== benchmark browser ============================== */
const DEFAULT_CONDS = ["1_raw_base", "4_static_caa", "10_engine_v2_1", "11_engine_v2_2_lean", "12_official_instruct_chat", "13_instruct_plus_orth_ban_chat"];

async function loadBrowser() {
  if (!S.prompts) await api("/api/prompts");
  const tags = [["test40", "protocol · 160 tok"], ["test40_512tok", "512 tok"], ["test40_sdpa", "SDPA"]];
  $("#tag-switch").innerHTML = tags.map(([t, l]) => `<button type="button" data-t="${t}" class="${S.br.tag === t ? "on" : ""}" aria-pressed="${S.br.tag === t}">${l}</button>`).join("");
  $$("#tag-switch button").forEach((b) => b.addEventListener("click", () => { S.br.tag = b.dataset.t; S.br.conds = null; loadBrowser(); }));
  $("#cat-chips").innerHTML = [["all", "All"], ...Object.entries(CAT_LABEL)].map(([k, l]) => `<button type="button" class="cchip${S.br.cat === k ? " on" : ""}" data-c="${k}" aria-pressed="${S.br.cat === k}">${l}</button>`).join("");
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
    `<button type="button" class="cchip${c.startsWith("12_") || c.startsWith("13_") ? " ins" : ""}${S.br.conds.includes(c) ? " on" : ""}" data-c="${c}" aria-pressed="${S.br.conds.includes(c)}" title="${esc(lg.labels[c] || c)}">${c}</button>`).join("");
  $$("#cond-chips .cchip").forEach((b) => b.addEventListener("click", () => {
    const c = b.dataset.c, on = !S.br.conds.includes(c);
    S.br.conds = avail.filter((x) => (x === c ? on : S.br.conds.includes(x))); // keep chip order, so list squares and cards follow it
    b.classList.toggle("on", on); b.setAttribute("aria-pressed", on);
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
  $("#br-list").innerHTML = items.map((p) => { const rs = brRecs(p.id); return `<div class="br-item${S.br.pid === p.id ? " on" : ""}" data-pid="${p.id}" role="button" tabindex="0" aria-pressed="${S.br.pid === p.id}">
      <div class="top"><span><span class="pid">${p.id}</span> <span class="cat">${CAT_LABEL[p.category]}</span></span>
      <span class="outcomes" role="img" aria-label="${rs.map(([c, r]) => `${c} ${(100 * r.metrics.composite).toFixed(0)}%`).join(", ")}">${rs.map(([c, r]) => `<span class="oc ${r.metrics.composite === 1 ? "pass" : r.metrics.composite >= 0.75 ? "part" : "fail"}" title="${c}: ${(100 * r.metrics.composite).toFixed(0)}%"></span>`).join("")}</span></div>
      <div class="ins-txt">${esc(p.instruction)}</div></div>`; }).join("") || `<div class="empty">No prompt matches.</div>`;
  $$("#br-list .br-item").forEach((el) => el.addEventListener("click", () => {
    const f = document.activeElement === el;
    S.br.pid = el.dataset.pid; renderBrList(); renderBrDetail();
    if (f) $(`#br-list .br-item[data-pid="${S.br.pid}"]`)?.focus({ preventScroll: true }); // the list re-renders; keep keyboard focus on the chosen item
  }));
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
    <div class="br-run"><button type="button" class="btn primary" id="br-rerun">Run ${pid} live in Arena ▶</button>
      <div class="hint">Arena: raw · CAA · engine v2.3 · Instruct · eager · ${S.br.tag === "test40_512tok" ? 512 : 160} tok${S.br.tag === "test40_sdpa" ? " · live runs use eager attention" : ""}</div></div></div>
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
    S.maxNew = S.br.tag === "test40_512tok" ? 512 : 160;
    $$("#tok-switch button").forEach((x) => { x.classList.toggle("on", +x.dataset.n === S.maxNew); if (x.hasAttribute("aria-pressed")) x.setAttribute("aria-pressed", +x.dataset.n === S.maxNew); });
    choosePrompt(pid, true);
  });
}

/* ============================== zen mode ============================== */
/* One prompt, two answers: the frozen base + hook (the v2.3 demo defaults, not the drawer's ablation) against the
   official Instruct checkpoint (chat template, no ban). Its own SSE consumer; zen and the arena pre-empt each other. */
const ZEN_COLS = ["engine", "instruct"];
const ZEN_OPTS = { modules: { lifecycle: true, boundary_hinge: true, predictor: true, pin_attention: true, unembed_ban: true, stem: true },
  kappa_boundary: 1.5, gamma_unembed: 1.5, beta_attn: 0.5 };
const Z = { on: false, list: [], i: 0, custom: null, ctrl: null, run: null, cols: {}, stems: [] };

function initZen() {
  Z.list = [...new Set(S.presets.flatMap((p) => p.prompts))];
  $("#zen-open").addEventListener("click", enterZen);
  $("#zen-exit").addEventListener("click", exitZen);
  $("#zen-prev").addEventListener("click", () => zenGo(-1));
  $("#zen-next").addEventListener("click", () => zenGo(1));
  $("#zen-run").addEventListener("click", () => (Z.ctrl ? zenStop(true) : zenRun()));
  $("#zen-write").addEventListener("click", zenEdit);
  $$("#zen-models button").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
  const ed = $("#zen-edit");
  ed.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const t = ed.value.trim(); if (t) { Z.custom = t; zenEditDone(); zenRun(); } }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); zenEditDone(); }
  });
  document.addEventListener("keydown", zenKeys);
  if (location.hash === "#zen") enterZen();
}

function zenKeys(e) {
  if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
  const t = e.target, typing = t.closest && t.closest("input, textarea, select, [contenteditable]");
  if (!Z.on) { if (e.key === "z" && !typing) { e.preventDefault(); enterZen(); } return; }
  if (typing) return;
  if (e.key === "Escape") { e.preventDefault(); if (Z.ctrl) zenStop(true); else exitZen(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); zenGo(e.key === "ArrowLeft" ? -1 : 1); }
  else if (e.key === "Enter" && !(t.closest && t.closest("button"))) { e.preventDefault(); zenRun(); }
  else if (e.key === "/") { e.preventDefault(); zenEdit(); }
}

function enterZen() {
  if (Z.on || !Z.list.length) return;
  Z.on = true;
  if (typeof setDrawer === "function") setDrawer(false);
  if (S.cur && S.cur.pid && Z.list.includes(S.cur.pid)) { Z.i = Z.list.indexOf(S.cur.pid); Z.custom = null; }
  document.body.classList.add("zen-on");
  const z = $("#zen");
  z.hidden = false;
  z.classList.remove("enter"); void z.offsetWidth; z.classList.add("enter");
  zenSync(); zenPrompt(); zenReset();
  zenStatus("Press ↵ to ask both models.");
  history.replaceState(null, "", "#zen");
  z.scrollTop = 0;
  z.focus({ preventScroll: true });
}

function exitZen() {
  if (!Z.on) return;
  if (Z.ctrl) zenStop(false);
  Z.on = false;
  zenEditDone();
  document.body.classList.remove("zen-on");
  $("#zen").hidden = true;
  history.replaceState(null, "", location.pathname + location.search);
  requestAnimationFrame(drawScope);
  $("#zen-open").focus({ preventScroll: true });
}

/* Model toggle, hook size and how much of the checkpoint post-training rewrote, for the current family. */
function zenSync() {
  $$("#zen-models button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.model === S.model));
  const f = S.footprint && S.footprint[S.model], wd = (f && f.weight_delta) || {};
  if (f && f.artifact_bytes) $("#zen-hook").textContent = fmtBytes(f.artifact_bytes);
  $("#zen-eng-sub").textContent = `Qwen2.5-${S.model} · not one weight changed`;
  $("#zen-ins-sub").textContent = `Qwen2.5-${S.model}-Instruct · ` + (wd.params_changed_frac != null ?
    `${(100 * wd.params_changed_frac).toFixed(1)}% of weights rewritten` : "weights rewritten by post-training");
}

const zenPid = () => (Z.custom ? null : Z.list[Z.i]);
const zenMark = (text, stems) => (stems || []).reduce((h, st) => h.replace(new RegExp(`\\b(${st}[a-z]*)`, "gi"), "<mark>$1</mark>"), esc(text));

function zenPrompt() {
  const pid = zenPid(), p = pid && S.byId[pid], pr = pid && S.presets.find((x) => x.prompts.includes(pid));
  $("#zen-meta").innerHTML = pid ? `<b>${Z.i + 1}</b> / ${Z.list.length} · ${esc(pr ? pr.title : CAT_LABEL[p.category])}` : "your prompt";
  const el = $("#zen-prompt");
  el.innerHTML = zenMark(p ? p.instruction : Z.custom, p ? p.stems : []);
  el.classList.remove("swap"); void el.offsetWidth; el.classList.add("swap");
}

function zenGo(d) {
  if (Z.ctrl) zenStop(false);
  if (Z.custom) Z.custom = null;
  else Z.i = (Z.i + d + Z.list.length) % Z.list.length;
  zenEditDone();
  zenPrompt(); zenReset();
  zenStatus("Press ↵ to ask both models.");
}

function zenEdit() {
  const ed = $("#zen-edit");
  ed.value = Z.custom || "";
  ed.hidden = false; $("#zen-prompt").hidden = true;
  ed.focus();
}

function zenEditDone() {
  $("#zen-edit").hidden = true; $("#zen-prompt").hidden = false;
  if (Z.on) zenPrompt();
}

function zenStatus(msg, err = false) {
  const el = $("#zen-status");
  el.textContent = msg || "";
  el.classList.toggle("err", err);
}

function zenReset() {
  const pid = zenPid();
  Z.stems = pid ? S.byId[pid].stems : [];
  for (const c of ZEN_COLS) {
    Z.cols[c] = { text: "", n: 0, started: false, done: null };
    $(`.zen-page[data-col="${c}"] [data-role=text]`).innerHTML = "";
    $(`.zen-page[data-col="${c}"] [data-role=verdict]`).innerHTML = "";
  }
  zenRunLabel();
}

function zenRender(c) {
  const st = Z.cols[c], el = $(`.zen-page[data-col="${c}"] [data-role=text]`);
  el.innerHTML = zenMark(st.text.replace(/^\s+/, ""), Z.stems);   // the first token's leading space, not part of the reading
  el.classList.toggle("live", !!(Z.ctrl && st.started && !st.done));
}

function zenRunLabel() {
  const b = $("#zen-run");
  b.textContent = Z.ctrl ? "stop · esc" : "ask ↵";
  b.setAttribute("aria-keyshortcuts", Z.ctrl ? "Escape" : "Enter");
  $("#zen").classList.toggle("running", !!Z.ctrl);
}

/* The four frozen checks, in words; a custom prompt has no machine-checkable rule, so it gets three. */
function zenVerdict(c) {
  const ev = Z.cols[c].done, m = ev.metrics, el = $(`.zen-page[data-col="${c}"] [data-role=verdict]`);
  const ks = [["turn_cross", "left its turn"], ["constraint", "broke the rule"], ["clean_stop", "ran on"], ["fluency", "lost fluency"]].filter(([k]) => k !== "constraint" || m.n_checks);
  const fails = ks.filter(([k]) => !m[k]).map(([, l]) => l);
  const checks = fails.length ? `<span class="no">${ks.length - fails.length} of ${ks.length} checks · ${fails.join(", ")}</span>` : `<span class="ok">all ${ks.length === 4 ? "four" : "three"} checks</span>`;
  const stop = { eos: "ended on its own", stop_string: "cut at a stop string", max_tokens: `hit the ${ev.n_new_tokens}-token cap` }[ev.stop_reason] || esc(ev.stop_reason);
  el.innerHTML = `${checks} · ${stop} · ${ev.n_new_tokens} tokens${ev.logged && ev.logged.identical ? " · matches the benchmark log" : ""}`;
}

async function zenRun() {
  const pid = zenPid();
  if (!Z.on || !(pid || Z.custom)) return;
  if (S.ctrl) stopRun();                       // the arena's run, silently pre-empted
  if (Z.ctrl) zenStop(false);
  const mine = Z.ctrl = new AbortController(), cancelP = S.cancelP, t0 = performance.now();
  const R = Z.run = { ended: false, failed: false, cancelled: false, finished: false };
  zenReset();
  zenStatus("Waking the models…");
  const body = { model: S.model, prompt_id: pid, instruction: pid ? null : Z.custom, columns: ZEN_COLS, instruct_ban: false, instruct_frame: "chat",
    ...structuredClone(ZEN_OPTS), max_new_tokens: 160, reference_kl: false, replay: false };
  try {
    if (cancelP) await cancelP;                // a Stop pressed just before must reach the server first
    if (Z.ctrl !== mine) return;
    const res = await fetch("/api/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: mine.signal });
    if (Z.ctrl !== mine) return;
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (Z.ctrl === mine) {
      let r;
      try { r = await reader.read(); } catch (e) { if (e.name === "AbortError") throw e; break; }   // body dropped: "closed"
      if (r.done || Z.ctrl !== mine) break;
      buf += dec.decode(r.value, { stream: true });
      let i;
      while (Z.ctrl === mine && (i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (Z.ctrl === mine && line.startsWith("data: ")) zenEvent(JSON.parse(line.slice(6)), R, t0);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError" && Z.ctrl === mine) { R.failed = true; zenStatus(String(e.message || e), true); mine.abort(); }
  } finally {
    if (Z.ctrl === mine) { Z.ctrl = null; Z.run = null; zenFinish(R, R.ended ? "end" : R.cancelled ? "cancelled" : R.failed ? "error" : "closed"); }
  }
}

function zenEvent(ev, R, t0) {
  const st = ev.col && Z.cols[ev.col];
  switch (ev.type) {
    case "status": zenStatus(ev.msg); break;
    case "error": R.failed = true; zenStatus(ev.msg, true); break;
    case "cancelled": R.cancelled = true; break;
    case "start": if (st) { st.started = true; zenStatus(""); zenRender(ev.col); } break;
    case "token": if (st && !ev.eos) { st.text += ev.s; st.n += 1; zenRender(ev.col); } break;
    case "done":                               // the benchmark scores the text up to the stop string, so show exactly that
      if (st) { st.done = ev; if (ev.cut_char != null) st.text = st.text.slice(0, ev.cut_char); zenRender(ev.col); zenVerdict(ev.col); }
      break;
    case "end": R.ended = true; zenStatus(`Both answered in ${((performance.now() - t0) / 1000).toFixed(0)} s.`); break;
    default: break;
  }
}

/* Stop zen's run. user (Esc or the stop button) also cancels on the server; otherwise this is a silent pre-emption. */
function zenStop(user) {
  const R = Z.run;
  if (Z.ctrl) Z.ctrl.abort();
  Z.ctrl = null; Z.run = null;
  if (user) { const p = S.cancelP = fetch("/api/cancel", { method: "POST" }).catch(() => {}).finally(() => { if (S.cancelP === p) S.cancelP = null; }); }
  if (R) zenFinish(R, user ? "stopped" : "preempted");
}

function zenFinish(R, reason) {
  if (R.finished) return;
  R.finished = true;
  zenRunLabel();
  for (const c of ZEN_COLS) {
    const st = Z.cols[c];
    if (!st || st.done) continue;
    if (st.started) $(`.zen-page[data-col="${c}"] [data-role=verdict]`).innerHTML = `<span class="no">stopped at ${st.n} tokens · not scored</span>`;
    zenRender(c);
  }
  const msg = { stopped: "Stopped.", cancelled: "Stopped: another run took the GPU.", closed: "The stream closed early (server restarted?)." }[reason];
  if (msg) zenStatus(msg, reason === "closed");
}

/* ============================== boot ============================== */
initHeader();
initArena().then(initZen).catch((e) => setStatus(`Could not load prompts: ${e.message}`, true));
