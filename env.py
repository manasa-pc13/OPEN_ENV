from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from grader import grade_dataset
from tasks import TASKS, TaskSpec


# -----------------------------
# Typed models (Pydantic)
# -----------------------------

ActionType = Literal[
    "remove_nulls",
    "fill_mean",
    "remove_duplicates",
    "fix_format",
    "normalize_column",
]


class Action(BaseModel):
    """
    One action per step.

    Examples:
      {"type": "fill_mean"}
      {"type": "normalize_column", "column": "salary"}
    """

    type: ActionType
    column: Optional[str] = None


class Observation(BaseModel):
    """
    What the agent sees after reset() and each step().
    """

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    max_steps: int
    dataset_preview: List[Dict[str, Any]]
    null_counts: Dict[str, int]
    duplicate_count: int
    format_issues: Dict[str, List[str]]
    normalized_columns: List[str]


class StepInfo(BaseModel):
    """
    Extra info returned by step() to explain what happened.
    """

    action_received: Dict[str, Any]
    action_status: Literal["applied", "redundant", "wrong", "invalid"]
    message: str
    step_reward: float
    grader_score_if_done: Optional[float] = None
    total_reward_so_far: float


# -----------------------------
# OpenEnv-style environment
# -----------------------------


class DataCleaningEnv:
    """
    Data Cleaning Agent Environment (OpenEnv-style API).

    Required API:
    - reset()
    - step(action) -> (observation, reward, done, info)
    - state()

    Dataset is a list of dict rows.
    """

    def __init__(self, task_id: str = "easy", max_steps: int = 10):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id='{task_id}'. Valid: {list(TASKS.keys())}")

        self.task_id = task_id
        self.max_steps = int(max_steps)

        self._task: TaskSpec = TASKS[task_id]
        self._dataset: List[Dict[str, Any]] = copy.deepcopy(self._task.messy_dataset)

        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._normalized_columns: set[str] = set()

    def reset(self) -> Observation:
        self._task = TASKS[self.task_id]
        self._dataset = copy.deepcopy(self._task.messy_dataset)

        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._normalized_columns = set()

        obs = self._make_observation()
        print(f"[DEBUG] reset task_id={self.task_id} difficulty={self._task.difficulty}")
        return obs

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self._task.difficulty,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "total_reward": self._total_reward,
            "dataset": copy.deepcopy(self._dataset),
        }

    def step(self, action: Dict[str, Any] | Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply one action. Must return: (observation, reward, done, info)

        Reward rules (MANDATORY):
        - Correct action → +0.2
        - Wrong action → -0.1
        - Redundant action → -0.05
        Plus, when the episode ends, we also add the final grader score.

        Reward returned by step() is clamped to [0.0, 1.0] as required.
        """

        if self._done:
            obs = self._make_observation()
            info = StepInfo(
                action_received=self._to_action_dict(action),
                action_status="invalid",
                message="Episode already done. Call reset() to start again.",
                step_reward=0.0,
                grader_score_if_done=float(grade_dataset(self._dataset, self._task)),
                total_reward_so_far=self._total_reward,
            ).model_dump()
            return obs, 0.0, True, info

        self._step_count += 1

        # Validate action input (safeguard: never crash on bad input)
        try:
            a = action if isinstance(action, Action) else Action.model_validate(action)
        except Exception as e:
            raw_reward = -0.1
            reward = self._clamp_reward(raw_reward)
            self._total_reward += reward
            done = self._check_done_or_limit()
            obs = self._make_observation()
            info = StepInfo(
                action_received=self._to_action_dict(action),
                action_status="invalid",
                message=f"Invalid action format: {e}",
                step_reward=reward,
                grader_score_if_done=float(grade_dataset(self._dataset, self._task)) if done else None,
                total_reward_so_far=self._total_reward,
            ).model_dump()
            print(f"[DEBUG] step invalid action reward={reward:.3f} done={done}")
            return obs, reward, done, info

        before = copy.deepcopy(self._dataset)
        status, message = self._apply_action(a)
        changed = before != self._dataset

        # Step reward shaping
        raw_reward = 0.0
        if status == "applied":
            if changed and a.type in self._task.required_actions:
                raw_reward = 0.2
            elif not changed:
                status = "redundant"
                raw_reward = -0.05
            else:
                raw_reward = 0.0
        elif status == "redundant":
            raw_reward = -0.05
        else:
            raw_reward = -0.1

        done = self._check_done_or_limit()

        grader_score = None
        if done:
            grader_score = float(grade_dataset(self._dataset, self._task))
            raw_reward += grader_score

        reward = self._clamp_reward(raw_reward)
        self._total_reward += reward

        obs = self._make_observation()
        info = StepInfo(
            action_received=a.model_dump(),
            action_status=status,
            message=message,
            step_reward=reward,
            grader_score_if_done=grader_score,
            total_reward_so_far=self._total_reward,
        ).model_dump()

        print(
            f"[DEBUG] step {self._step_count}/{self.max_steps} action={a.type} "
            f"status={status} reward={reward:.3f} done={done}"
        )
        return obs, reward, done, info

    # -----------------------------
    # Observation helpers
    # -----------------------------

    def _clamp_reward(self, r: float) -> float:
        return float(max(0.0, min(1.0, r)))

    def _to_action_dict(self, action: Any) -> Dict[str, Any]:
        if isinstance(action, Action):
            return action.model_dump()
        if isinstance(action, dict):
            return action
        return {"raw": repr(action)}

    def _check_done_or_limit(self) -> bool:
        score = float(grade_dataset(self._dataset, self._task))
        if score >= 0.99:
            self._done = True
            return True
        if self._step_count >= self.max_steps:
            self._done = True
            return True
        return False

    def _make_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            difficulty=self._task.difficulty,
            step_count=self._step_count,
            max_steps=self.max_steps,
            dataset_preview=copy.deepcopy(self._dataset[: min(5, len(self._dataset))]),
            null_counts=self._count_nulls(self._dataset),
            duplicate_count=self._count_duplicates(self._dataset),
            format_issues=self._detect_format_issues(self._dataset),
            normalized_columns=sorted(list(self._normalized_columns)),
        )

    # -----------------------------
    # Dataset checks
    # -----------------------------

    def _is_null(self, v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "na", "n/a"):
            return True
        return False

    def _count_nulls(self, rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        cols: set[str] = set()
        for r in rows:
            cols.update(r.keys())
            for k, v in r.items():
                if self._is_null(v):
                    counts[k] = counts.get(k, 0) + 1
        for c in cols:
            counts.setdefault(c, 0)
        return dict(sorted(counts.items(), key=lambda x: x[0]))

    def _count_duplicates(self, rows: List[Dict[str, Any]]) -> int:
        seen = set()
        dup = 0
        for r in rows:
            key = json.dumps(r, sort_keys=True, default=str)
            if key in seen:
                dup += 1
            else:
                seen.add(key)
        return dup

    def _detect_format_issues(self, rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        issues: Dict[str, List[str]] = {}
        for r in rows:
            for k, v in r.items():
                if isinstance(v, str):
                    s = v.strip()
                    if "$" in s or ("," in s and any(ch.isdigit() for ch in s)):
                        issues.setdefault(k, []).append("currency_or_thousands")
                    if "/" in s and len(s.split("/")) == 3:
                        issues.setdefault(k, []).append("date_slash_format")
        for k in list(issues.keys()):
            issues[k] = sorted(list(set(issues[k])))
        return dict(sorted(issues.items(), key=lambda x: x[0]))

    # -----------------------------
    # Action implementations
    # -----------------------------

    def _apply_action(
        self, action: Action
    ) -> Tuple[Literal["applied", "redundant", "wrong", "invalid"], str]:
        t = action.type

        if t == "remove_nulls":
            before = copy.deepcopy(self._dataset)
            self._dataset = self._remove_rows_with_nulls(self._dataset)
            if self._dataset == before:
                return "redundant", "No rows removed (no null rows detected)."
            return "applied", "Removed rows that contained null/missing values."

        if t == "fill_mean":
            before = copy.deepcopy(self._dataset)
            self._dataset = self._fill_numeric_nulls_with_mean(self._dataset)
            if self._dataset == before:
                return "redundant", "No numeric nulls filled (nothing to do)."
            return "applied", "Filled numeric missing values with column mean."

        if t == "remove_duplicates":
            before = copy.deepcopy(self._dataset)
            self._dataset = self._remove_duplicate_rows(self._dataset)
            if self._dataset == before:
                return "redundant", "No duplicates removed (none detected)."
            return "applied", "Removed duplicate rows."

        if t == "fix_format":
            before = copy.deepcopy(self._dataset)
            self._dataset = self._fix_common_formats(self._dataset)
            if self._dataset == before:
                return "redundant", "No format issues fixed (nothing changed)."
            return "applied", "Fixed common formats (currency strings, date strings)."

        if t == "normalize_column":
            if not action.column:
                return "invalid", "normalize_column requires 'column' (example: {'type':'normalize_column','column':'salary'})."
            col = action.column
            before = copy.deepcopy(self._dataset)
            self._dataset, ok = self._minmax_normalize_column(self._dataset, col)
            if not ok:
                return "wrong", f"Could not normalize column '{col}'. It must be numeric with variation and no missing values."
            if self._dataset == before:
                return "redundant", f"Column '{col}' already normalized (or normalization had no effect)."
            self._normalized_columns.add(col)
            return "applied", f"Normalized column '{col}' into range [0, 1]."

        return "invalid", f"Unknown action type: {t}"

    # -----------------------------
    # Cleaning operations
    # -----------------------------

    def _remove_rows_with_nulls(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            if any(self._is_null(v) for v in r.values()):
                continue
            out.append(r)
        return out

    def _fill_numeric_nulls_with_mean(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def to_float(x: Any) -> Optional[float]:
            if isinstance(x, (int, float)) and x == x:
                return float(x)
            return None

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for r in rows:
            for k, v in r.items():
                fv = to_float(v)
                if fv is not None:
                    sums[k] = sums.get(k, 0.0) + fv
                    counts[k] = counts.get(k, 0) + 1

        means = {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}

        out = copy.deepcopy(rows)
        for r in out:
            for k, v in r.items():
                if self._is_null(v) and k in means:
                    r[k] = means[k]
        return out

    def _remove_duplicate_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for r in rows:
            key = json.dumps(r, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _fix_common_formats(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fixes:
        - "$1,200.50" -> 1200.50 (float)
        - "01/05/2026" -> "2026-01-05" (string)
        """
        out = copy.deepcopy(rows)

        def parse_currency_or_number(s: str) -> Optional[float]:
            t = s.strip().replace("$", "").replace(",", "")
            try:
                return float(t)
            except Exception:
                return None

        def fix_date_slash(s: str) -> Optional[str]:
            t = s.strip()
            parts = t.split("/")
            if len(parts) != 3:
                return None
            mm, dd, yyyy = parts
            if not (mm.isdigit() and dd.isdigit() and yyyy.isdigit() and len(yyyy) == 4):
                return None
            return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

        for r in out:
            for k, v in list(r.items()):
                if isinstance(v, str):
                    # currency/number first
                    fv = parse_currency_or_number(v)
                    if fv is not None and any(ch.isdigit() for ch in v):
                        r[k] = fv
                        continue
                    # then date
                    ds = fix_date_slash(v)
                    if ds is not None:
                        r[k] = ds
                        continue
        return out

    def _minmax_normalize_column(self, rows: List[Dict[str, Any]], column: str) -> Tuple[List[Dict[str, Any]], bool]:
        vals: List[float] = []
        for r in rows:
            if column not in r:
                return copy.deepcopy(rows), False
            v = r[column]
            if isinstance(v, (int, float)) and v == v:
                vals.append(float(v))
            else:
                return copy.deepcopy(rows), False

        if not vals:
            return copy.deepcopy(rows), False

        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return copy.deepcopy(rows), False

        out = copy.deepcopy(rows)
        for r in out:
            r[column] = (float(r[column]) - vmin) / (vmax - vmin)
        return out, True

