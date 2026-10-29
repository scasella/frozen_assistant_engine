"""Deterministic metric checks (no model needed)."""

from eval.verifiers import (check_json_keys, check_math_answer, check_numbered_count, check_only_code,
                            clean_stop, contamination, score_response, turn_cross)

SQL = ("Write a SQL query to find the top 3 departments by average salary from a table employees(department, salary). "
       "Return only the SQL code block.")


def test_flan_template_and_bracket_turns_are_contamination():
    flan = ("Photosynthesis is how plants make food.\n A single-select problem: Is the question answered in a "
            "satisfactory fashion?\n\nAvailable options:\n (a). yes.\n (b). no.")
    assert contamination(flan)
    assert contamination("The answer is 168.\n[Question]A store sells 20 packets")
    assert contamination("It helps.\nYou are a helpful assistant, who always answers")
    assert not contamination("1. Java\n2. C\n3. Rust")
    assert not contamination("- Solar\n- Wind\n- Hydro")


def test_clean_stop_penalises_mid_sentence_cap_but_not_eos():
    assert clean_stop("A full answer.", "eos")
    assert not clean_stop("An answer that was cut off in the", "max_tokens")
    assert not clean_stop("Fine.\nQuestion: next one?", "eos")


def test_turn_cross_echo_rule():
    assert turn_cross("Here is the SQL query to find the top 3 departments by average salary from a table "
                      "employees(department, salary):\n```sql\nSELECT 1\n```", SQL)
    assert not turn_cross(SQL + "\nSELECT 1", SQL)                      # literal re-typing of the prompt
    assert turn_cross("Tom reads 25 pages per day. 300 / 25 = 12 days. The answer is 12.",
                      "Tom reads 25 pages per day. How many days will it take him to read a 300-page book?")
    assert not turn_cross("Question 2: What is light?\nQuestion 3: Why?", "x")
    assert not turn_cross("What is gravity?\nWhat is mass?", "x")


def test_constraint_checks():
    assert check_math_answer("So 3x = 36.", "36")
    assert check_math_answer("It ends at 10:15 PM.", "10:15")
    assert not check_math_answer("The answer is 37.", "36")
    assert check_json_keys('```json\n{"summary": "100 C", "confidence": 0.9}\n```', ["summary", "confidence"])
    assert check_numbered_count("1. A\n2. B\n3. C\n4. D\n5. E", 5)
    assert check_only_code("```sql\nSELECT 1;\n```") and not check_only_code("Here is the query:\n```sql\nSELECT 1;\n```")


def test_score_response_composite():
    prompt = {"instruction": "Explain gravity without using the word 'force'.",
              "checks": [{"type": "forbidden", "patterns": ["\\bforc(e|es|ed|ing)\\b"]}, {"type": "min_words", "n": 5}]}
    m = score_response("Gravity is the attraction between objects with mass.", "eos", prompt)
    assert m["composite"] == 1.0 and m["natural_eos"] == 1.0
    m = score_response("Gravity is a force between objects with mass.", "eos", prompt)
    assert m["constraint"] == 0.0 and m["composite"] == 0.75
