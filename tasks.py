from __future__ import annotations

from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """
    A task definition for the Data Cleaning environment.

    - messy_dataset: the starting (dirty) data
    - expected_clean: the target cleaned data
    - required_actions: actions that are truly needed to reach the clean target
      (used by the env for step-level "correct vs wrong" shaping)
    """

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    messy_dataset: List[Dict[str, Any]]
    expected_clean: List[Dict[str, Any]]
    required_actions: Set[str] = Field(default_factory=set)


# -----------------------------
# EASY: Only missing values
# -----------------------------
# Strategy that matches the expected_clean:
# - Use fill_mean (keeps all rows, fills numeric nulls)
EASY_MESSY = [
    {"id": 1, "age": 20, "salary": 1000.0},
    {"id": 2, "age": None, "salary": 1500.0},
    {"id": 3, "age": 30, "salary": None},
    {"id": 4, "age": 40, "salary": 2000.0},
]

# mean(age) ignoring null = (20+30+40)/3 = 30
# mean(salary) ignoring null = (1000+1500+2000)/3 = 1500
EASY_EXPECTED = [
    {"id": 1, "age": 20, "salary": 1000.0},
    {"id": 2, "age": 30.0, "salary": 1500.0},
    {"id": 3, "age": 30, "salary": 1500.0},
    {"id": 4, "age": 40, "salary": 2000.0},
]


# -----------------------------
# MEDIUM: Missing values + duplicates
# -----------------------------
# Strategy:
# - remove_duplicates
# - fill_mean
MEDIUM_MESSY = [
    {"id": 1, "age": 22, "salary": 1200.0},
    {"id": 2, "age": None, "salary": 1300.0},
    {"id": 2, "age": None, "salary": 1300.0},  # duplicate row
    {"id": 3, "age": 26, "salary": None},
    {"id": 4, "age": 30, "salary": 1600.0},
]

# mean(age) ignoring null = (22+26+30)/3 = 26
# mean(salary) ignoring null = (1200+1300+1600)/3 = 1366.666...
MEDIUM_EXPECTED = [
    {"id": 1, "age": 22, "salary": 1200.0},
    {"id": 2, "age": 26.0, "salary": 1300.0},
    {"id": 3, "age": 26, "salary": 1366.6666666666667},
    {"id": 4, "age": 30, "salary": 1600.0},
]


# -----------------------------
# HARD: Missing + duplicates + format issues + normalization
# -----------------------------
# Strategy:
# - fix_format
# - remove_duplicates
# - fill_mean
# - normalize_column (salary)
HARD_MESSY = [
    {"id": 10, "date": "01/05/2026", "age": 18, "salary": "$1,000"},
    {"id": 11, "date": "01/06/2026", "age": None, "salary": "$1,500"},
    {"id": 11, "date": "01/06/2026", "age": None, "salary": "$1,500"},  # duplicate
    {"id": 12, "date": "01/07/2026", "age": 22, "salary": None},
    {"id": 13, "date": "01/08/2026", "age": 30, "salary": "$2,500"},
]

# After fix_format:
# - salary becomes floats: 1000, 1500, 1500, None, 2500
# - date becomes "YYYY-MM-DD"
# After remove_duplicates: remove one id=11 row
# After fill_mean:
# - mean(age) = (18+22+30)/3 = 23.333333333333332
# - mean(salary) = (1000+1500+2500)/3 = 1666.6666666666667
# Then normalize salary with min=1000, max=2500:
# - 1000 -> 0.0
# - 1500 -> 0.3333333333333333
# - 1666.666... -> 0.4444444444444445
# - 2500 -> 1.0
HARD_EXPECTED = [
    {"id": 10, "date": "2026-01-05", "age": 18, "salary": 0.0},
    {"id": 11, "date": "2026-01-06", "age": 23.333333333333332, "salary": 0.3333333333333333},
    {"id": 12, "date": "2026-01-07", "age": 22, "salary": 0.4444444444444445},
    {"id": 13, "date": "2026-01-08", "age": 30, "salary": 1.0},
]


TASKS: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        task_id="easy",
        difficulty="easy",
        description="Fill missing numeric values using column means.",
        messy_dataset=EASY_MESSY,
        expected_clean=EASY_EXPECTED,
        required_actions={"fill_mean"},
    ),
    "medium": TaskSpec(
        task_id="medium",
        difficulty="medium",
        description="Remove duplicate rows and fill missing numeric values using column means.",
        messy_dataset=MEDIUM_MESSY,
        expected_clean=MEDIUM_EXPECTED,
        required_actions={"remove_duplicates", "fill_mean"},
    ),
    "hard": TaskSpec(
        task_id="hard",
        difficulty="hard",
        description="Fix currency/date formats, remove duplicates, fill missing values, then normalize salary.",
        messy_dataset=HARD_MESSY,
        expected_clean=HARD_EXPECTED,
        required_actions={"fix_format", "remove_duplicates", "fill_mean", "normalize_column"},
    ),
}

