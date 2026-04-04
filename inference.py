from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from env import DataCleaningEnv
from tasks import TASKS

BENCHMARK = os.getenv("BENCHMARK", "data_cleaning")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
    api_key = os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not configured (missing HF_TOKEN or openai package).")

    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

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

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content.strip()
    return json.loads(text)


def run_task(task_id: str) -> None:
    print(f"[DEBUG] \n" + "=" * 70, flush=True)
    print(f"[DEBUG] Task: {task_id} ({TASKS[task_id].difficulty})", flush=True)
    print(f"[DEBUG] Goal: {TASKS[task_id].description}", flush=True)

    env = DataCleaningEnv(task_id=task_id, max_steps=10)
    obs_model = env.reset()

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    done = False
    step_count = 0
    rewards: List[float] = []

    while not done:
        step_count += 1
        obs = obs_model.model_dump()

        # Choose action
        action: Optional[Dict[str, Any]] = None
        error_msg = None
        try:
            action = openai_next_action(obs, task_id)
            print(f"[DEBUG] OpenAI action: {action}", flush=True)
        except Exception as e:
            plan = rule_based_plan(obs, task_id)
            action = plan[0]
            print(f"[DEBUG] Fallback action: {action} (reason: {e})", flush=True)

        obs_model, reward, done, info = env.step(action)
        rewards.append(reward)

        log_step(step=step_count, action=str(action).replace('"', "'"), reward=reward, done=done, error=error_msg)

        if done:
            grader_score = float(info.get('grader_score_if_done', 0.0))
            score = max(0.0, min(1.0, grader_score))
            success = score >= 1.0
            
            print(f"[DEBUG] grader_score={grader_score}", flush=True)
            print(f"[DEBUG] total_reward={info.get('total_reward_so_far')}", flush=True)
            print(f"[DEBUG] cleaned dataset: {env.state()['dataset']}", flush=True)
            
            log_end(success=success, steps=step_count, score=score, rewards=rewards)


def main() -> None:
    for task_id in ["easy", "medium", "hard"]:
        run_task(task_id)


if __name__ == "__main__":
    main()

