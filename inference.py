from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from env import DataCleaningEnv
from tasks import TASKS

# Optional OpenAI usage (fallback to rules if no API key).
# This project runs fine without OpenAI.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


def rule_based_plan(obs: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    """
    A simple "good enough" policy for demo purposes:
    - If format issues exist -> fix_format
    - If duplicates exist -> remove_duplicates
    - If any nulls exist -> fill_mean
    - If hard -> normalize salary
    """
    actions: List[Dict[str, Any]] = []

    if obs.get("format_issues"):
        actions.append({"type": "fix_format"})

    if obs.get("duplicate_count", 0) > 0:
        actions.append({"type": "remove_duplicates"})

    null_counts = obs.get("null_counts", {})
    if any(int(v) > 0 for v in null_counts.values()):
        actions.append({"type": "fill_mean"})

    if task_id == "hard":
        actions.append({"type": "normalize_column", "column": "salary"})

    # If we think there's nothing to do, try a harmless redundant action.
    if not actions:
        actions.append({"type": "remove_duplicates"})
    return actions


def openai_next_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Ask OpenAI for the single best next action.
    If anything fails, caller should fall back to rule-based.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not configured (missing OPENAI_API_KEY or openai package).")

    client = OpenAI(api_key=api_key)

    system = (
        "You are a data cleaning agent. "
        "Return ONLY valid JSON with this schema:\n"
        '{ "type": "remove_nulls|fill_mean|remove_duplicates|fix_format|normalize_column", "column": "optional" }\n'
        "Pick the best NEXT action (one step) to improve data quality."
    )

    payload = {
        "task_id": task_id,
        "difficulty": TASKS[task_id].difficulty,
        "observation": obs,
        "valid_actions": [
            {"type": "remove_nulls"},
            {"type": "fill_mean"},
            {"type": "remove_duplicates"},
            {"type": "fix_format"},
            {"type": "normalize_column", "column": "salary"},
        ],
    }

    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.0,
    )

    text = resp.output_text.strip()
    return json.loads(text)


def run_task(task_id: str) -> None:
    print("\n" + "=" * 70)
    print(f"Task: {task_id} ({TASKS[task_id].difficulty})")
    print(f"Goal: {TASKS[task_id].description}")

    env = DataCleaningEnv(task_id=task_id, max_steps=10)
    obs_model = env.reset()

    done = False
    while not done:
        obs = obs_model.model_dump()

        # Choose action
        action: Optional[Dict[str, Any]] = None
        try:
            action = openai_next_action(obs, task_id)
            print(f"[agent] OpenAI action: {action}")
        except Exception as e:
            plan = rule_based_plan(obs, task_id)
            action = plan[0]
            print(f"[agent] Fallback action: {action} (reason: {e})")

        obs_model, reward, done, info = env.step(action)

        print(f"[reward] {reward:.3f}")
        print(f"[info] {info['message']}")

        if done:
            print(f"[final] grader_score={info.get('grader_score_if_done')}")
            print(f"[final] total_reward={info.get('total_reward_so_far')}")
            print("[final] cleaned dataset:")
            print(env.state()["dataset"])


def main() -> None:
    for task_id in ["easy", "medium", "hard"]:
        run_task(task_id)


if __name__ == "__main__":
    main()

