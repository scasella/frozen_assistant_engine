"""End-to-end check of the studio (demo/server.py) through FastAPI's TestClient, in-process.

    .venv/bin/python scripts/verify_demo.py            # needs ~13 GB free: loads Qwen2.5-3B and -3B-Instruct
    .venv/bin/python scripts/verify_demo.py --quick    # data endpoints only (no model)

Checks:
1. /api/summary, /api/prompts, /api/ablation_data and /api/logs return complete Phase 1-4 data for 3B and 1.5B.
2. Reproducibility: with the engine set to exactly condition 11, every arena column (raw, CAA, engine, Instruct; decoded
   interleaved) reproduces its cached benchmark log on t_qa_04, t_ng_01 and t_ng_04, and the engine alone does too.
3. Telemetry is read-only: the v2.3 demo engine streamed with telemetry and step-KL produces exactly the tokens that
   ``engine.generate`` produces without telemetry.
4. Live 4-way streams of the v2.3 demo on t_qa_04, t_ng_01, t_ng_04 carry drift_z / hinge_threshold / init_norm /
   protected_siblings telemetry, and on t_ng_04 (ban_morphology="stem") the engine explains how to ride a bicycle with
   no "balance", "balancing" or "balanc" and passes all four programmatic checks.
5. Report: the v2.3 demo engine on all 8 negation prompts vs. condition 11's logs (informational, not a selection).
Writes results/demo_verification.json and exits non-zero on any failed assertion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from demo import data, server  # noqa: E402
from engine import AssistantEngine, generate  # noqa: E402
from engine.prompts import EVAL_STOP_STRINGS  # noqa: E402
from eval.conditions import encode_for  # noqa: E402
from eval.verifiers import score_response  # noqa: E402

FAILS: list[str] = []
REPORT: dict = {"checks": {}}


def check(name: str, ok: bool, detail: str = "") -> None:
    REPORT["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def stream(client: TestClient, body: dict) -> dict:
    """POST /api/stream; returns {col: {"tokens": [...], "done": {...}, "start": {...}}} plus errors."""
    cols: dict[str, dict] = {}
    errors = []
    with client.stream("POST", "/api/stream", json=body) as r:
        assert r.status_code == 200, r.status_code
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            c = ev.get("col")
            if ev["type"] == "error":
                errors.append(ev["msg"])
            elif ev["type"] == "start":
                cols.setdefault(c, {"tokens": []})["start"] = ev
            elif ev["type"] == "token":
                cols.setdefault(c, {"tokens": []})["tokens"].append(ev)
            elif ev["type"] == "done":
                cols.setdefault(c, {"tokens": []})["done"] = ev
    cols["_errors"] = errors
    return cols


def check_data(client: TestClient) -> None:
    p = client.get("/api/prompts").json()
    check("prompts: 40 test prompts and 7 presets", len(p["prompts"]) == 40 and len(p["presets"]) == 7,
          f"{len(p['prompts'])} prompts, {len(p['presets'])} presets")
    for mk in data.MODELS:
        s = client.get("/api/summary", params={"model": mk}).json()
        conds = {r["condition"] for r in s["rows"]}
        need = {"1_raw_base", "3_urial_3shot_prefix", "4_static_caa", "10_engine_v2_1", "11_engine_v2_2_lean",
                "12_official_instruct_chat", "12_official_instruct_bare", "13_instruct_plus_orth_ban_chat"}
        kl_ok = all(r["kl"].get("heldout_no_negation") for r in s["rows"] if r["condition"] not in ("1_raw_base", "13_instruct_plus_orth_ban_chat"))
        has512 = all(r["composite_512"] is not None for r in s["rows"] if r["condition"] in
                     ("1_raw_base", "10_engine_v2_1", "11_engine_v2_2_lean", "12_official_instruct_chat"))
        check(f"summary {mk}: unit economics, KL pools, 512-token composites, recovery",
              need <= conds and kl_ok and has512 and "11_engine_v2_2_lean@512" in s["recovery"] and s["weight_delta"],
              f"{len(s['rows'])} approaches; KL held-out no-neg engine {s['rows'][4]['kl']['heldout_no_negation']['mean_kl']:.4f}")
        a = client.get("/api/ablation_data", params={"model": mk}).json()
        rows = {r["condition"]: r for r in a["rows"]}
        n_base = sum(not r["instruct"] for r in a["rows"])
        complete = all(len(r["categories"]) == 5 and len(r["ci_vs_raw"]) == 2 for r in a["rows"])
        check(f"ablation {mk}: 20 base conditions + 4 post-trained, CIs and categories", n_base == 20 and len(rows) == 24 and complete,
              f"{n_base} + {len(rows) - n_base}; cond 11 composite {rows['11_engine_v2_2_lean']['composite']:.1f}")
        for tag in data.TAGS:
            lg = client.get("/api/logs", params={"model": mk, "tag": tag}).json()
            sizes = {c: len(v) for c, v in lg["conditions"].items()}
            check(f"logs {mk}/{tag}", bool(sizes) and all(n == 40 for n in sizes.values()), f"{len(sizes)} conditions x 40")


EXACT11 = {"lifecycle": True, "boundary_hinge": True, "predictor": True, "pin_attention": True, "unembed_ban": True, "stem": False}


def check_reproduction(client: TestClient) -> None:
    for pid in ("t_qa_04", "t_ng_01", "t_ng_04"):
        res = stream(client, {"model": "3B", "prompt_id": pid, "modules": EXACT11, "reference_kl": True})
        same = {c: v["done"].get("logged", {}).get("identical") for c, v in res.items() if c != "_errors"}
        check(f"interleaved 4-way = cached logs ({pid})", not res["_errors"] and all(same.values()) and len(same) == 4,
              ", ".join(f"{c}={'=' if s else '≠'}" for c, s in same.items()) + (f" errors {res['_errors']}" if res["_errors"] else ""))
    res = stream(client, {"model": "3B", "prompt_id": "t_ng_04", "columns": ["engine"], "modules": EXACT11, "reference_kl": False})
    check("engine alone = cached condition-11 log (t_ng_04)", res["engine"]["done"]["logged"]["identical"])


def check_telemetry_is_read_only() -> None:
    lm = server.MANAGER.get("Qwen/Qwen2.5-3B")
    art = server.MANAGER.artifact("Qwen/Qwen2.5-3B")
    cfg = replace(server.COND11, ban_morphology="stem")
    p = data.test_prompts()["t_ng_04"]
    enc = encode_for(lm.tokenizer, p["instruction"], False)
    eng = AssistantEngine(lm, art, cfg).attach()
    try:
        ref = generate(lm, enc, 160, EVAL_STOP_STRINGS, eng, lm.eos_ids or None)
    finally:
        eng.detach()
    client = TestClient(server.app)
    res = stream(client, {"model": "3B", "prompt_id": "t_ng_04", "columns": ["engine"], "reference_kl": True})
    ok = res["engine"]["done"]["text"] == ref.text and res["engine"]["done"]["n_new_tokens"] == ref.n_new_tokens
    check("telemetry + step-KL do not change the v2.3 output (stream vs. generate)", ok,
          f"{ref.n_new_tokens} tokens, stop {ref.stop_reason}")


REQUIRED = ("drift_z", "hinge_threshold", "init_norm", "sustain_norm", "eos_cos", "in_boundary_window", "step_kl_nats")


def check_live(client: TestClient) -> None:
    for pid in ("t_qa_04", "t_ng_01", "t_ng_04"):
        t0 = time.time()
        res = stream(client, {"model": "3B", "prompt_id": pid, "reference_kl": True})
        eng = res["engine"]
        tels = [t.get("tel") or {} for t in eng["tokens"]]
        keys_ok = all(all(k in tel for k in REQUIRED) for tel in tels)
        drift_ok = all(tel["drift_z"] is not None and tel["hinge_threshold"] is not None for tel in tels[1:])
        init_ok = tels[0]["init_norm"] > tels[1]["init_norm"] > tels[3]["init_norm"] > 0 and tels[4]["init_norm"] == 0
        negation = bool(data.negation_info(data.test_prompts()[pid]["instruction"])["targets"])
        sib_ok = (any(tel.get("protected_siblings") for tel in tels) and
                  all(len(tel["protected_siblings"]) == 4 for tel in tels if "protected_siblings" in tel)) if negation else True
        check(f"telemetry stream ({pid})", not res["_errors"] and keys_ok and drift_ok and init_ok and sib_ok,
              f"{len(tels)} engine events; init pulse {tels[0]['init_norm']:.1f} → {tels[3]['init_norm']:.2f} → {tels[4]['init_norm']:.0f}; "
              f"max z {max(t['drift_z'] for t in tels[1:]):.2f}; boundary steps {sum(t['in_boundary_window'] for t in tels)}"
              + (f"; sibling lists of 4 on {sum('protected_siblings' in t for t in tels)} steps" if negation else ""))
        d = eng["done"]
        m = d["metrics"]
        REPORT.setdefault("live", {})[pid] = {c: {"text": v["done"]["text"], "metrics": {k: v["done"]["metrics"][k] for k in
                                                ("turn_cross", "constraint", "clean_stop", "fluency", "composite")},
                                                "stop_reason": v["done"]["stop_reason"], "n_new_tokens": v["done"]["n_new_tokens"],
                                                "tok_s": v["done"]["tok_s"]} for c, v in res.items() if c != "_errors"}
        print(f"       {pid}: engine {m['composite']:.2f} ({d['stop_reason']}, {d['n_new_tokens']} tok) in {time.time() - t0:.0f}s; "
              + ", ".join(f"{c} {v['done']['metrics']['composite']:.2f}" for c, v in res.items() if c not in ("_errors", "engine")))
        if pid == "t_ng_04":
            text = d["text"]
            hits = re.findall(r"balanc\w*", text, flags=re.IGNORECASE)
            on_topic = bool(re.search(r"\b(?:pedal|handlebar|bicycle|bike)", text, flags=re.IGNORECASE))
            all4 = all(m[k] == 1.0 for k in ("turn_cross", "constraint", "clean_stop", "fluency"))
            check("t_ng_04 stem ban: no 'balance' / 'balancing' / 'balanc', on topic, all 4 checks pass",
                  not hits and on_topic and all4, f"hits {hits}; checks {[m[k] for k in ('turn_cross', 'constraint', 'clean_stop', 'fluency')]}; "
                  f"guard steps {d['engine_trace'].get('stem_guard_steps', 0)}")
            REPORT["t_ng_04_text"] = text


def negation_sweep() -> None:
    lm = server.MANAGER.get("Qwen/Qwen2.5-3B")
    art = server.MANAGER.artifact("Qwen/Qwen2.5-3B")
    cfg = replace(server.COND11, ban_morphology="stem")
    log11 = data.logs("3B")["11_engine_v2_2_lean"]
    rows = []
    eng = AssistantEngine(lm, art, cfg).attach()
    try:
        for pid, p in data.test_prompts().items():
            if p["category"] != "negative_constraints":
                continue
            res = generate(lm, encode_for(lm.tokenizer, p["instruction"], False), 160, EVAL_STOP_STRINGS, eng, lm.eos_ids or None)
            m = score_response(res.text, res.stop_reason, p)
            stems = data.negation_info(p["instruction"])["stems"]
            rows.append({"prompt_id": pid, "v23_composite": m["composite"], "cond11_composite": log11[pid]["metrics"]["composite"],
                         "v23_stem_strict": data.stem_strict(res.text, stems),
                         "cond11_stem_strict": data.stem_strict(log11[pid]["response"], stems),
                         "identical_to_cond11": res.text == log11[pid]["response"], "v23_text": res.text})
    finally:
        eng.detach()
    n = len(rows)
    v23 = 100 * sum(r["v23_composite"] for r in rows) / n
    c11 = 100 * sum(r["cond11_composite"] for r in rows) / n
    strict = (sum(bool(r["v23_stem_strict"]) for r in rows), sum(bool(r["cond11_stem_strict"]) for r in rows))
    REPORT["negation_sweep_3B"] = {"v23_composite": v23, "cond11_composite": c11, "stem_strict_v23": strict[0],
                                   "stem_strict_cond11": strict[1], "n": n, "rows": rows,
                                   "note": "informational; the stem rule was motivated by test prompt t_ng_04, so this is not a held-out estimate"}
    print(f"[INFO] v2.3 demo on the 8 negation prompts (3B): Composite {v23:.1f} vs condition 11 {c11:.1f}; "
          f"stem-strict {strict[0]}/{n} vs {strict[1]}/{n}; identical to condition 11 on {sum(r['identical_to_cond11'] for r in rows)}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="data endpoints only (no model loading)")
    args = ap.parse_args()
    client = TestClient(server.app)
    t0 = time.time()
    check_data(client)
    if not args.quick:
        check_reproduction(client)
        check_telemetry_is_read_only()
        check_live(client)
        negation_sweep()
    REPORT["seconds"] = round(time.time() - t0, 1)
    REPORT["failed"] = FAILS
    if not args.quick:
        out = ROOT / "results" / "demo_verification.json"
        out.write_text(json.dumps(REPORT, indent=2))
        print(f"wrote {out.relative_to(ROOT)}")
    print(f"{len(REPORT['checks']) - len(FAILS)}/{len(REPORT['checks'])} checks passed in {REPORT['seconds']:.0f}s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
