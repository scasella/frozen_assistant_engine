"""Prompt framings, the URIAL teacher prefix, stop strings and negative-trap templates."""

from __future__ import annotations

import random

# Stop strings enforced at the decoding-loop level for all "+stops" conditions.
EVAL_STOP_STRINGS: list[str] = ["\nQuestion:", "\nUser:"]
# Calibration teacher generation additionally cuts at markdown rules.
CALIB_STOP_STRINGS: list[str] = ["\nQuestion:", "\nUser:", "\n---"]

QUESTION_PREFIX = "Question: "
ANSWER_DELIM = "\nAnswer:"

URIAL_PREFIX = """The following is a record of questions answered by a helpful, knowledgeable, and honest AI assistant. Each user message is labeled "Question:" and each assistant reply is labeled "Answer:". The assistant answers the question directly and accurately, follows every instruction about format, length, and wording exactly, explains its reasoning when that helps the reader, and never adds unrelated content. When the answer is complete, the assistant stops.

Question: Explain how vaccines protect people from disease. Use exactly 3 bullet points, each starting with "- ".
Answer: - A vaccine safely exposes the immune system to a harmless piece or weakened form of a pathogen, such as an inactivated virus, a purified surface protein, or genetic instructions (mRNA) that let the body's own cells briefly make that protein. Because the real pathogen is absent or disabled, the person does not develop the disease itself.
- The immune system reacts as if a genuine infection were underway: B cells produce antibodies that bind to the specific shape of the pathogen's proteins, and T cells learn to recognize and destroy cells that have been infected. This first response takes one to two weeks to fully develop.
- Long-lived memory B cells and T cells remain in the body afterwards, sometimes for decades. If the person later meets the real pathogen, these memory cells recognize it within hours and mount a fast, strong response that clears the infection before it can cause serious illness. When enough people in a community are vaccinated, the pathogen also struggles to spread, which protects people who cannot be vaccinated.

Question: A school buys 9 boxes of markers. Each box holds 15 markers, and 27 markers are given to the art club. How many markers does the school have left? Show your reasoning.
Answer: Let's work through the problem one step at a time.

Step 1: Find the total number of markers the school bought. There are 9 boxes, and each box holds 15 markers, so we multiply:

9 × 15 = 135 markers.

Step 2: Subtract the markers that were given to the art club:

135 − 27 = 108 markers.

Step 3: Check the result. Adding the markers that were given away back to the markers that remain should give the original total: 108 + 27 = 135, which matches the 135 markers we started with.

The school has 108 markers left.

Question: Write a Python function called is_palindrome(s) that checks whether a string is a palindrome, ignoring capitalization and spaces.
Answer: Here is a simple and readable implementation:

```python
def is_palindrome(s: str) -> bool:
    cleaned = "".join(ch.lower() for ch in s if not ch.isspace())
    return cleaned == cleaned[::-1]
```

How it works:

1. The generator expression walks through every character of the input, skips whitespace, and converts the remaining characters to lowercase, so "Racecar" and "race car" are treated the same way.
2. `"".join(...)` combines those characters into a single cleaned string.
3. The slice `cleaned[::-1]` produces the cleaned string in reverse order. If the string and its reverse are identical, the text reads the same forwards and backwards, so the function returns `True`; otherwise it returns `False`.

Example usage:

```python
print(is_palindrome("Never odd or even"))  # True
print(is_palindrome("Python"))             # False
print(is_palindrome(""))                   # True (an empty string is trivially a palindrome)
```

The function runs in linear time, O(n), because it looks at each character a constant number of times.

"""


def minimal_prompt(instruction: str) -> str:
    """Minimal zero-shot frame used by every steered condition at inference time."""
    return f"{QUESTION_PREFIX}{instruction}{ANSWER_DELIM}"


# --- Negative trap templates (base-model failure modes) -------------------------------------

_USERNAMES = ["user_99", "throwaway_42", "jdoe1987", "mike_t", "curious_cat", "anon5521", "techfan", "lurker_01"]
_SITES = ["Related links:", "Related questions:", "See also:", "More from this thread:"]


def trap_question_continuation(other: list[str], rng: random.Random) -> str:
    """Trap A: the model keeps generating an exam / FAQ list instead of answering."""
    style = rng.randrange(4)
    if style == 0:
        return "".join(f"\nQuestion {k + 2}: {q}" for k, q in enumerate(other[:3]))
    if style == 1:
        return "".join(f"\n[Question]{q}\n[Answer]" for q in other[:2])
    if style == 2:
        return "".join(f"\nQ: {q}\nA:" for q in other[:2])
    return "".join(f"\n\nQuestion: {q}\nAnswer:" for q in other[:2])


def trap_forum_boilerplate(other: list[str], rng: random.Random) -> str:
    """Trap B: the model drifts into web-scrape / forum boilerplate."""
    user = rng.choice(_USERNAMES)
    n_comments = rng.randint(2, 48)
    related = rng.choice(_SITES)
    return (
        f"\n\nPosted by {user} | {n_comments} comments | Share | {related}\n"
        f"- {other[0]}\n- {other[1]}\n"
        f"Report | Reply | Upvote ({rng.randint(1, 300)})"
    )


def trap_multi_turn(other: list[str]) -> str:
    """Trap C: after answering, the model hallucinates a new user turn instead of stopping."""
    return f"\nUser: Thanks! Can you also help with this one? {other[0]}\nAssistant: Sure! Here is"


_EXAM_TEMPLATES = [
    "\n A single-select problem: Is the question answered in a satisfactory fashion?\n\nAvailable options:\n (A). yes.\n (B). no.",
    "\n\nChoose your answer from:\n (a). yes\n (b). no\nThe answer is: (a).",
    "\nOPTIONS:\n- yes\n- no\nAnswer: yes",
    "\n A single-choice question: {q}\nOptions are:\n (1). True\n (2). False",
    "\n\nMulti-choice problem: {q}\nPick your answer from:\n[a]. Yes.\n[b]. No.",
]


def trap_exam_template(other: list[str], rng: random.Random) -> str:
    """Trap D: pretraining-genre drift into instruction-dataset / exam templates (FLAN-style options)."""
    return rng.choice(_EXAM_TEMPLATES).format(q=other[0])

