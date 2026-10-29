"""Deterministic, programmatic metrics. No LLM judge.

TurnCross   : the response starts answering (not another question, not an echo of the prompt, not empty/garbage)
Constraint  : every machine-checkable rule attached to the prompt passes (see ``CHECKS``)
CleanStop   : no hallucinated turn / FAQ / forum markers, and not cut off at max tokens mid-sentence or in a loop
Fluency     : distinct 4-gram ratio > 0.75 and length > 10 characters
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from collections.abc import Callable

# Line-start markers of a new turn / FAQ entry / transcript speaker ("Question 2:", "[Question]", "User:", "Q:").
_TURN_MARKERS = re.compile(
    r"^\s*(?:#+\s*)?(?:\*\*)?(?:\[(?:question|q|user|human|assistant|answer)\s*\d*\]"
    r"|(?:question|q|user|human|assistant)\s*\d*\s*(?:\*\*)?\s*[:.)]|a\s*:)",
    re.IGNORECASE | re.MULTILINE,
)
# Web-scrape / forum boilerplate, and instruction-dataset / exam templates leaking from pretraining data
# (FLAN-style "A single-select problem ... Available options: (a). yes", hallucinated system prompts).
_WEB_MARKERS = re.compile(
    r"posted by|posted on|\b\d+\s+comments?\b|^\s*share\s*(?:\||$)|related (?:links|questions|posts|articles)"
    r"|^\s*reply\b|\bupvote|\bdownvote|all rights reserved|click here|^\s*tags?\s*:|asked by|answered by"
    r"|^\s*best answer|^\s*report\s*(?:\||$)|^\s*see also\s*:|^\s*\d+\s+(?:views|likes|answers)\b"
    r"|single-select problem|single-choice question|multi-choice problem|multiple-choice question"
    r"|available options\s*:|options are\s*:|^\s*options\s*:|choose your answer|pick your answer|select from the following"
    r"|^\s*[(\[]?[a-d1-4][)\]]?[.)]\s*(?:yes|no|true|false)\b|you are a helpful assistant|you are an ai assistant",
    re.IGNORECASE | re.MULTILINE,
)
_QUESTION_START = re.compile(r"^(?:what|why|how|which|who|when|where|can|is|are|do|does|should|could|would)\b", re.I)
_TERMINAL = (".", "!", "?", ")", "]", "}", '"', "'", "`", ";", ":")


def words(text: str) -> list[str]:
    return text.split()


def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    w = text.lower().split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 if not grams else len(set(grams)) / len(grams)


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"()-[]{}`*#_/<>=+%$&|\\~^@" for ch in text) / len(text)


# ---------------------------------------------------------------------------------------------
# Lifecycle metrics
# ---------------------------------------------------------------------------------------------

def _non_ascii_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return 0.0 if not letters else sum(not c.isascii() for c in letters) / len(letters)


def turn_cross(response: str, instruction: str) -> bool:
    text = response.strip()
    if sum(ch.isalnum() for ch in text) < 3 or _alpha_ratio(text[:200]) < 0.85:
        return False
    if _non_ascii_letter_ratio(text[:200]) > 0.3:
        return False  # drifted into another language / script
    first = _first_line(text)
    if _TURN_MARKERS.match(first) or _WEB_MARKERS.search(first):
        return False
    if first.endswith("?") and _QUESTION_START.match(first) and len(text.splitlines()) <= 1:
        return False  # the whole "answer" is just another question
    if first.endswith("?") and _QUESTION_START.match(first) and len(words(first)) >= 3:
        nxt = text[len(first):].strip()
        if not nxt or _TURN_MARKERS.match(_first_line(nxt)) or _first_line(nxt).endswith("?"):
            return False  # a list of questions
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    resp, ins = norm(text).split(), norm(instruction).split()
    head, rest = resp[: len(ins)], resp[len(ins):]
    if ins and difflib.SequenceMatcher(None, head, ins).ratio() > 0.9 and (head[:1] == ins[:1] or len(rest) < 5):
        # re-typed the whole prompt, or only restated it. A paraphrased preamble followed by an answer
        # ("Here is the query that ...") and a math answer restating the premises both pass.
        return False
    return True


def contamination(response: str) -> list[str]:
    """Markers of a run-on turn / FAQ continuation / forum boilerplate inside the response."""
    found = []
    body = response.strip()
    for m in _TURN_MARKERS.finditer(body):
        found.append(m.group(0).strip())
    for m in _WEB_MARKERS.finditer(body):
        found.append(m.group(0).strip())
    return found


def clean_stop(response: str, stop_reason: str) -> bool:
    if contamination(response):
        return False
    if stop_reason == "max_tokens":
        text = response.rstrip()
        mid_sentence = not text.endswith(_TERMINAL)
        looping = distinct_ngram_ratio(text) <= 0.75 or _repeated_lines(text)
        if mid_sentence or looping:
            return False
    return True


def _repeated_lines(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 8]
    return len(lines) >= 4 and len(set(lines)) / len(lines) < 0.6


def fluency(response: str) -> bool:
    return len(response.strip()) > 10 and distinct_ngram_ratio(response) > 0.75


# ---------------------------------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------------------------------

def _code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        m = re.search(r"```[a-zA-Z0-9_+-]*\n?(.*)$", text, flags=re.DOTALL)  # unterminated fence
        if m:
            blocks = [m.group(1)]
    return blocks


def _python_source(text: str) -> str:
    blocks = _code_blocks(text)
    if blocks:
        return "\n".join(blocks)
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith(("def ", "import ", "from ", "class "))), None)
    return "\n".join(lines[start:]) if start is not None else text


def _parse_python(src: str) -> ast.Module | None:
    lines = src.splitlines()
    for cut in range(len(lines), 0, -1):  # tolerate trailing prose after the code
        try:
            return ast.parse("\n".join(lines[:cut]))
        except SyntaxError:
            continue
    return None


def _extract_json(text: str) -> object | None:
    candidates = _code_blocks(text) + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        for m in re.finditer(r"[\[{]", cand):
            try:
                obj, _ = dec.raw_decode(cand[m.start():])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?::\d\d)?", text)]


def _sentences(text: str) -> list[str]:
    t = re.sub(r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Dr|St)\.", "_", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p for p in parts if len(p.split()) >= 2]


def check_keywords_min(text: str, keywords: list[str], min: int) -> bool:  # noqa: A002
    low = text.lower()
    return sum(k.lower() in low for k in keywords) >= min


def check_bullet_count(text: str, n: int, marker: str = "- ") -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    bullets = [l for l in lines if l.lstrip().startswith(marker)]
    return len(bullets) == n


def check_numbered_count(text: str, n: int) -> bool:
    nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)[.)]\s+\S", text, flags=re.MULTILINE)]
    return len(nums) == n and nums == list(range(1, n + 1))


def check_json_keys(text: str, keys: list[str]) -> bool:
    obj = _extract_json(text)
    return isinstance(obj, dict) and all(k in obj for k in keys)


def check_sentence_count(text: str, n: int) -> bool:
    return len(_sentences(text)) == n


def check_max_words(text: str, n: int) -> bool:
    return len(words(text)) <= n


def check_min_words(text: str, n: int) -> bool:
    return len(words(text)) >= n


def check_all_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= 0.95


def check_contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def check_forbidden(text: str, patterns: list[str]) -> bool:
    return not any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def check_one_paragraph(text: str) -> bool:
    t = text.strip()
    return "\n\n" not in t and not re.search(r"^\s*(?:[-*•]|\d+[.)])\s", t, flags=re.MULTILINE)


def check_regex_count_min(text: str, patterns: list[str], min: int, case_sensitive: bool = False) -> bool:  # noqa: A002
    flags = 0 if case_sensitive else re.IGNORECASE
    return sum(bool(re.search(p, text, flags=flags)) for p in patterns) >= min


def check_math_answer(text: str, answer: str) -> bool:
    nums = _numbers(text)
    target = answer.replace(",", "")
    if not nums:
        return False
    if nums[-1] == target:
        return True
    try:
        if ":" not in target and float(nums[-1]) == float(target):
            return True
    except ValueError:
        pass
    pat = rf"(?:answer|result|total|therefore|thus|so|equals|=|is)\W[^\n]{{0,40}}?(?<![\d.:]){re.escape(target)}(?![\d:])"
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def check_code_keywords(text: str, patterns: list[str]) -> bool:
    src = "\n".join(_code_blocks(text)) or text
    return all(re.search(p, src, flags=re.IGNORECASE) for p in patterns)


def check_python_def(text: str, name: str) -> bool:
    tree = _parse_python(_python_source(text))
    if tree is None:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))


_PROSE_LINE = re.compile(r"^[A-Z][a-z']+(?:\s+[A-Za-z',()]+){4,}.*[.:!]$")


def check_only_code(text: str) -> bool:
    t = text.strip()
    if t.startswith("```"):
        return t.endswith("```") and t.count("```") == 2
    return "```" not in t and not any(_PROSE_LINE.match(l.strip()) for l in t.splitlines())


CHECKS: dict[str, Callable[..., bool]] = {
    "keywords_min": check_keywords_min,
    "bullet_count": check_bullet_count,
    "numbered_count": check_numbered_count,
    "json_keys": check_json_keys,
    "sentence_count": check_sentence_count,
    "max_words": check_max_words,
    "min_words": check_min_words,
    "all_upper": check_all_upper,
    "contains_any": check_contains_any,
    "forbidden": check_forbidden,
    "one_paragraph": check_one_paragraph,
    "regex_count_min": check_regex_count_min,
    "math_answer": check_math_answer,
    "code_keywords": check_code_keywords,
    "python_def": check_python_def,
    "only_code": check_only_code,
}


def constraint_score(response: str, checks: list[dict]) -> tuple[bool, dict[str, bool]]:
    detail = {}
    for i, c in enumerate(checks):
        params = {k: v for k, v in c.items() if k != "type"}
        detail[f"{i}:{c['type']}"] = bool(CHECKS[c["type"]](response, **params))
    return all(detail.values()), detail


def score_response(response: str, stop_reason: str, prompt: dict) -> dict:
    tc = turn_cross(response, prompt["instruction"])
    cs, detail = constraint_score(response, prompt["checks"])
    st = clean_stop(response, stop_reason)
    fl = fluency(response)
    return {
        "turn_cross": float(tc),
        "constraint": float(cs),
        "clean_stop": float(st),
        "fluency": float(fl),
        "composite": (tc + cs + st + fl) / 4.0,
        "natural_eos": float(stop_reason == "eos"),
        "constraint_detail": detail,
        "contamination": contamination(response)[:5],
        "distinct4": round(distinct_ngram_ratio(response), 3),
    }
