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
