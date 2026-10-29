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


def forbidden_token_ids(tokenizer, targets: list[str], morphology: str = "exact") -> dict[int, str]:
    """Single-token vocabulary entries (with and without a leading space) for every variant of every target.

    ``morphology="stem"`` adds every token that starts a word with the target's stem (``stem_token_ids``).
    """
    out: dict[int, str] = {}
    for t in targets:
        for form in sorted(word_variants(t)):
            for prefix in ("", " "):
                ids = tokenizer.encode(prefix + form, add_special_tokens=False)
                if len(ids) == 1:
                    out.setdefault(ids[0], prefix + form)
    if morphology == "stem":
        for i, s in stem_token_ids(tokenizer, [stem_of(t) for t in targets]).items():
            out.setdefault(i, s)
    elif morphology != "exact":
        raise ValueError(f"unknown ban_morphology {morphology!r}")
    # Sorted by token id: projections are applied sequentially and do not commute, so a set/hash-dependent order
    # would make generations differ between processes.
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------------------------
# ban_morphology="stem": inflected forms and sub-word spellings of a negated target
# ---------------------------------------------------------------------------------------------

_SUFFIXES = ("ing", "es", "ed", "er", "e", "s")   # longest first
_VOCAB_TEXT: dict[int, list[str]] = {}
_CONTINUATIONS: dict[tuple[int, str], list[int]] = {}


def stem_of(word: str) -> str:
    """Lowercase root of a negated target: a word of five or more letters loses one trailing 'ing', 'es', 'ed', 'er',
    'e' or 's' when at least four letters remain ('balance' -> 'balanc', 'computer' -> 'comput', 'water' -> 'water')."""
    w = word.strip().lower()
    if len(w) >= 5:
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: -len(suf)]
    return w


def _vocab_text(tokenizer) -> list[str]:
    key = id(tokenizer)
    if key not in _VOCAB_TEXT:
        _VOCAB_TEXT[key] = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    return _VOCAB_TEXT[key]


def _starts_word_with(text: str, stem: str) -> bool:
    """``text`` minus leading non-letters starts with ``stem`` (case-insensitively) without a camelCase join inside the
    stem: 'Force', ' forces' and 'FORCE' match 'forc'; ' forCell' does not."""
    head = re.sub(r"^[^A-Za-z]+", "", text)[: len(stem)]
    return head.lower() == stem and (head[1:] == head[1:].lower() or head.isupper())


def stem_token_ids(tokenizer, stems: list[str]) -> dict[int, str]:
    """Every vocabulary token that begins a word with one of ``stems`` (' balance', ' balancing', 'Balancer', ...)."""
    stems = [s for s in stems if s]
    return {i: t for i, t in enumerate(_vocab_text(tokenizer)) if any(_starts_word_with(t, s) for s in stems)}


def continuation_token_ids(tokenizer, remainder: str) -> list[int]:
    """Tokens that continue a word (first character is a letter) and start with ``remainder``. Appended to generated
    text whose last word spells the rest of a stem ('bal'), they complete it ('anc', 'ancing'). Sorted ids."""
    key = (id(tokenizer), remainder)
    if key not in _CONTINUATIONS:
        _CONTINUATIONS[key] = [i for i, t in enumerate(_vocab_text(tokenizer))
                               if t[:1].isascii() and t[:1].isalpha() and t.lower().startswith(remainder)]
    return _CONTINUATIONS[key]


def extend_word(partial: str, piece: str) -> str:
    """The trailing ASCII letter run of ``partial + piece``, given that ``partial`` is the trailing run of the text so far."""
    tail = re.search(r"[A-Za-z]*$", piece).group()
    return partial + piece if tail == piece else tail


def stem_completions(partial: str, stems: list[str]) -> list[str]:
    """Remainders that would complete a stem from the unfinished last word ``partial`` ('bal' -> ['anc'])."""
    p = partial.lower()
    return sorted({s[len(p):] for s in stems if p and len(p) < len(s) and s.startswith(p)})
