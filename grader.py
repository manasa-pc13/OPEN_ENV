from __future__ import annotations

import json
from typing import Any, Dict, List

from tasks import TaskSpec

# -----------------------------
# Grader requirements (MANDATORY)
#
# Score range: 0.0 .. 1.0
# Partial scoring:
# - nulls fixed → +0.3
# - duplicates removed → +0.3
# - format fixed → +0.2
# - normalized → +0.2
# -----------------------------

EPS = 1e-6


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "na", "n/a"):
        return True
    return False


def _count_nulls(rows: List[Dict[str, Any]]) -> int:
    c = 0
    for r in rows:
        for v in r.values():
            if _is_null(v):
                c += 1
    return c


def _duplicate_count(rows: List[Dict[str, Any]]) -> int:
    seen = set()
    dup = 0
    for r in rows:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup


def _looks_currency_or_thousands(v: Any) -> bool:
    return isinstance(v, str) and ("$" in v or ("," in v and any(ch.isdigit() for ch in v)))


def _looks_date_slash(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    parts = v.strip().split("/")
    return len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[2]) == 4


def _count_format_issues(rows: List[Dict[str, Any]]) -> int:
    issues = 0
    for r in rows:
        for v in r.values():
            if _looks_currency_or_thousands(v) or _looks_date_slash(v):
                issues += 1
    return issues


def _is_normalized_value(v: Any) -> bool:
    if not isinstance(v, (int, float)):
        return False
    x = float(v)
    return -EPS <= x <= 1.0 + EPS


def _column_is_normalized(rows: List[Dict[str, Any]], column: str) -> bool:
    vals: List[float] = []
    for r in rows:
        if column not in r:
            return False
        if not _is_normalized_value(r[column]):
            return False
        vals.append(float(r[column]))
    return (max(vals) - min(vals)) > EPS


def _rows_equal_with_tolerance(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """
    Simple deterministic comparison (order matters).
    Floats are compared with a tiny tolerance.
    """
    if len(a) != len(b):
        return False

    for ra, rb in zip(a, b):
        if set(ra.keys()) != set(rb.keys()):
            return False
        for k in ra.keys():
            va, vb = ra[k], rb[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if abs(float(va) - float(vb)) > 1e-6:
                    return False
            else:
                if va != vb:
                    return False
    return True


def grade_dataset(cleaned: List[Dict[str, Any]], task: TaskSpec) -> float:
    """
    Returns a score in [0, 1] based on the grading rubric.
    Partial credit is always allowed.
    """
    score = 0.0

    if _count_nulls(cleaned) == 0:
        score += 0.3

    if _duplicate_count(cleaned) == 0:
        score += 0.3

    if _count_format_issues(cleaned) == 0:
        score += 0.2

    if task.task_id == "hard":
        if _column_is_normalized(cleaned, "salary"):
            score += 0.2
    else:
        # For easy/medium we do not require normalization, so we don't penalize it.
        score += 0.2

    score = float(max(0.001, min(0.999, score)))

    # Perfect score requires matching the expected output.
    if score >= 0.99 and not _rows_equal_with_tolerance(cleaned, task.expected_clean):
        score = 0.95

    return float(max(0.001, min(0.999, score)))

