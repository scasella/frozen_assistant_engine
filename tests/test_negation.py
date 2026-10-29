"""Negation parser and deterministic banned-token ordering."""

import pytest

from engine.negation import forbidden_token_ids, parse_negations


def _targets(s: str) -> list[str]:
    return [t for n in parse_negations(s) for *_, t in n.targets]


def test_quoted_and_unquoted_targets():
    assert _targets("Explain gravity in one paragraph without using the word 'force'.") == ["force"]
    assert _targets("List 3 programming languages without mentioning Python.") == ["Python"]
    assert _targets("List three musical instruments without mentioning the piano or the guitar.") == ["piano", "guitar"]
    assert _targets("Describe Paris without mentioning the Eiffel Tower.") == []      # multi-word: span only


def test_quantity_limits_are_not_negations():
    assert parse_negations("Write the steps as a numbered list of no more than five steps.") == []


def test_echo_words():
    (span,) = parse_negations("List 3 programming languages without mentioning Python.")
    assert span.echo_words == ["without", "mentioning"]


def test_banned_ids_are_sorted():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in the local Hugging Face cache")
    ids = list(forbidden_token_ids(tok, ["piano", "guitar"]))
    assert ids == sorted(ids), "projection order must not depend on set iteration / hash seed"


# ---- ban_morphology="stem" (v2.3 demo) ----

from engine.negation import continuation_token_ids, extend_word, stem_completions, stem_of  # noqa: E402


def test_stem_of():
    assert [stem_of(w) for w in ("balance", "force", "water", "computer", "Python", "lions", "trees", "sand")] == \
        ["balanc", "forc", "water", "comput", "python", "lion", "tree", "sand"]


def test_partial_word_tracking():
    assert extend_word("", " bal") == "bal"
    assert extend_word("bal", "anc") == "balanc"
    assert extend_word("bal", "',") == ""
    assert extend_word("", " the word 'bal") == "bal"
    assert stem_completions("bal", ["balanc"]) == ["anc"]
    assert stem_completions("Bal", ["balanc", "forc"]) == ["anc"]
    assert stem_completions("balanc", ["balanc"]) == [] and stem_completions("", ["balanc"]) == []


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in the local Hugging Face cache")


def test_stem_ban_extends_the_exact_set():
    tok = _tokenizer()
    exact = forbidden_token_ids(tok, ["balance"])
    stem = forbidden_token_ids(tok, ["balance"], "stem")
    assert forbidden_token_ids(tok, ["balance"], "exact") == exact          # legacy default unchanged
    assert set(exact) < set(stem) and list(stem) == sorted(stem)
    assert {" balancing", " balanced", "Balancer"} <= set(stem.values())
    force = set(forbidden_token_ids(tok, ["force"], "stem").values())
    assert {" forcing", " forced", "FORCE"} <= force and " forCell" not in force  # no camelCase joins


def test_continuations_complete_a_stem():
    tok = _tokenizer()
    texts = {tok.decode([i]) for i in continuation_token_ids(tok, "anc")}
    assert {"anc", "ancing"} <= texts and not any(t.startswith(" ") for t in texts)
