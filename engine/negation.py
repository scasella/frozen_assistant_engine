"""Deterministic negation-span parser for polarity-aware attention pinning and negated-target suppression.

``parse_negations("Explain gravity in one paragraph without using the word 'force'.")`` returns one span whose
clause covers ``without using the word 'force'`` and whose single target is ``force``. Quantity limits such as
"no more than five steps" are not lexical negations and are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TRIGGER = re.compile(r"\b(?:without|do not|don't|never|avoid(?:ing)?|excluding|except|not|no)\b", re.IGNORECASE)
_QUANTITY = re.compile(r"(?:no|not)\s+(?:more|fewer|less|longer|shorter|greater)\s+than\b", re.IGNORECASE)
# A quote opens a quoted target only at a word boundary, so apostrophes ("don't") are not quotes.
_QUOTED = re.compile(r"(?<![\w])(['\"‘“])(.+?)(['\"’”])(?![\w])")
_FILLER = {"using", "use", "mentioning", "mention", "including", "include", "saying", "say", "writing", "write",
           "naming", "name", "the", "word", "words", "term", "terms", "a", "an", "any", "of"}
_ARTICLES = {"the", "a", "an", "any"}


@dataclass
class NegationSpan:
    clause: tuple[int, int]                                    # char span: trigger .. end of clause
    targets: list[tuple[int, int, str]] = field(default_factory=list)  # char spans + surface text
    echo_words: list[str] = field(default_factory=list)        # trigger word (+ "mentioning") for the anti-echo guard


def _clause_end(text: str, start: int, quotes: list[tuple[int, int]]) -> int:
    """First clause-ending punctuation after ``start`` that is outside a quoted span."""
    for m in re.finditer(r"[.;!?,](?=\s|$)", text[start:]):
        pos = start + m.start()
        if not any(a <= pos < b for a, b in quotes):
            return pos
    return len(text)


def parse_negations(instruction: str) -> list[NegationSpan]:
    quotes = [(m.start(), m.end()) for m in _QUOTED.finditer(instruction)]
    spans: list[NegationSpan] = []
    for m in _TRIGGER.finditer(instruction):
        if _QUANTITY.match(instruction, m.start()):
            continue
        if spans and m.start() < spans[-1].clause[1]:
            continue                                            # already inside the previous clause
        end = _clause_end(instruction, m.end(), quotes)
        span = NegationSpan(clause=(m.start(), end), echo_words=[m.group().split()[-1]])
        if re.match(r"\s+mentioning\b", instruction[m.end():], re.IGNORECASE):
            span.echo_words.append("mentioning")
        quoted = [q for q in _QUOTED.finditer(instruction, m.end(), end)]
        if quoted:
            span.targets = [(q.start(2), q.end(2), q.group(2)) for q in quoted]
        else:
            # unquoted: content words after the filler verbs, split on "or" / "and" / commas;
            # only single-word pieces become lexical targets ("lions", "Python", "the piano" -> "piano")
            rest_start = m.end()
            for piece in re.finditer(r"[^,]+?(?=\s+(?:or|and|nor)\s+|,|$)", instruction[rest_start:end]):
                words = [(w.start(), w.end(), w.group()) for w in re.finditer(r"[\w+#-]+", piece.group())]
                while words and words[0][2].lower() in _FILLER:
                    words.pop(0)
                if words and words[0][2].lower() in {"or", "and", "nor"}:
                    words.pop(0)
                while words and words[0][2].lower() in _ARTICLES:
                    words.pop(0)
                if len(words) == 1 and words[0][2].isalpha():
                    a = rest_start + piece.start() + words[0][0]
                    span.targets.append((a, a + len(words[0][2]), words[0][2]))
        spans.append(span)
    return spans


def char_spans_to_tokens(offsets: list[tuple[int, int]], spans: list[tuple[int, int]], shift: int) -> list[int]:
    """Token indices whose character offsets overlap any span (span coordinates shifted by ``shift``)."""
    out = []
    for i, (a, b) in enumerate(offsets):
        if b > a and any(b > s + shift and a < e + shift for s, e in spans):
            out.append(i)
    return out


def word_variants(word: str) -> set[str]:
    """Surface, capitalised and singular/plural variants of a negated target word."""
    w = word.strip()
    forms = {w, w.lower(), w.capitalize()}
    low = w.lower()
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        forms |= {low[:-1], low[:-1].capitalize()}
    else:
        plural = low + ("es" if low.endswith(("s", "x", "ch", "sh")) else "s")
        forms |= {plural, plural.capitalize()}
    return forms


def forbidden_token_ids(tokenizer, targets: list[str]) -> dict[int, str]:
    """Single-token vocabulary entries (with and without a leading space) for every variant of every target."""
    out: dict[int, str] = {}
    for t in targets:
        for form in sorted(word_variants(t)):
            for prefix in ("", " "):
                ids = tokenizer.encode(prefix + form, add_special_tokens=False)
                if len(ids) == 1:
                    out.setdefault(ids[0], prefix + form)
    # Sorted by token id: projections are applied sequentially and do not commute, so a set/hash-dependent order
    # would make generations differ between processes.
    return dict(sorted(out.items()))
