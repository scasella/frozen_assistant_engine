"""Studio API (demo/server.py) on the cached logs; no model is loaded."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from dataclasses import asdict, replace  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from demo import data, server  # noqa: E402
from eval.conditions import relevant_config  # noqa: E402

client = TestClient(server.app)


def test_prompts_and_presets():
    p = client.get("/api/prompts").json()
    ids = {x["id"] for x in p["prompts"]}
    assert len(ids) == 40 and all(pid in ids for pr in p["presets"] for pid in pr["prompts"])
    assert next(x for x in p["prompts"] if x["id"] == "t_ng_04")["stems"] == ["balanc"]


def test_ablation_rows_match_the_benchmark():
    a = client.get("/api/ablation_data", params={"model": "3B"}).json()
    rows = {r["condition"]: r for r in a["rows"]}
    assert len(rows) == 24 and round(rows["11_engine_v2_2_lean"]["composite"], 1) == 95.6
    lo, hi = rows["10_engine_v2_1"]["ci_vs_raw"]
    assert lo < rows["10_engine_v2_1"]["delta_vs_raw"] < hi


def test_summary_kl_pools():
    s = client.get("/api/summary", params={"model": "3B"}).json()
    by = {r["condition"]: r for r in s["rows"]}
    eng = by["11_engine_v2_2_lean"]["kl"]["heldout_no_negation"]["mean_kl"]
    ins = by["12_official_instruct_chat"]["kl"]["heldout_no_negation"]["mean_kl"]
    assert eng < 0.02 < 0.5 < ins


def test_demo_engine_defaults_are_condition_11_plus_stem():
    cfg = server.engine_column_config(server.StreamRequest())
    assert cfg.ban_morphology == "stem"
    assert relevant_config(asdict(replace(cfg, ban_morphology="exact"))) == relevant_config(asdict(server.COND11))


def test_custom_prompt_checks():
    checks = data.infer_checks("Describe the moon in fewer than 30 words without using the word 'light'.")
    assert {"type": "forbidden", "patterns": [r"\blight"]} in checks and {"type": "max_words", "n": 29} in checks
    assert data.stem_strict("It balanced.", ["balanc"]) is False and data.stem_strict("It balaced.", ["balanc"]) is True


def test_stream_replays_a_cached_log():
    """SSE plumbing (producer thread -> queue -> response) without a model: replay mode streams a logged answer."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in the local Hugging Face cache")
    import json
    events = []
    with client.stream("POST", "/api/stream", json={"model": "3B", "prompt_id": "t_ng_02", "columns": ["raw"],
                                                    "replay": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    done = next(e for e in events if e["type"] == "done")
    assert done["text"] == data.logs("3B")["1_raw_base"]["t_ng_02"]["response"] and done["source"] == "cached log"
    assert events[-1]["type"] == "end" and not server.GPU.locked()
