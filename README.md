# Data Cleaning Agent Environment (OpenEnv-style)

This is a beginner-friendly, **hackathon-ready** project that simulates a **real-world data cleaning job**.

You get a messy dataset (a Python list of dictionaries). An AI agent (or a simple rule-based fallback) cleans the data step-by-step by calling an environment API.

## What you built

Project files:

```
project/
├── env.py
├── tasks.py
├── grader.py
├── inference.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## OpenEnv-style API (the “environment”)

The environment class is `DataCleaningEnv` in `env.py`.

It implements the required functions:

- `reset()` → returns an **Observation**
- `step(action)` → returns `(observation, reward, done, info)`
- `state()` → returns the full internal state (for debugging)

### Actions (what the agent can do)

Valid `action["type"]` values:

- `remove_nulls` (drops rows with any missing values)
- `fill_mean` (fills missing numeric values with the column mean)
- `remove_duplicates` (removes duplicate rows)
- `fix_format` (fixes currency strings like `"$1,000"` and dates like `"01/05/2026"`)
- `normalize_column` (min-max normalizes a numeric column to `[0, 1]`)

Example actions:

```python
{"type": "fill_mean"}
{"type": "remove_duplicates"}
{"type": "normalize_column", "column": "salary"}
```

### Observation (what the agent sees)

Observation includes:

- dataset preview (first few rows)
- missing-value counts per column
- duplicate row count
- simple format issue detection
- which columns are normalized

These are typed using **Pydantic** models (`Observation`, `Action`, `StepInfo`).

## Tasks (easy, medium, hard)

Tasks are defined in `tasks.py`:

1. **easy**: missing values only
2. **medium**: missing values + duplicates
3. **hard**: missing values + duplicates + format issues + normalization

Each task includes:

- `messy_dataset`
- `expected_clean`
- `required_actions` (used for step reward shaping)

## Grader (score 0..1 with partial credit)

The grader is in `grader.py`.

Rubric (exactly as required):

- nulls fixed → +0.3
- duplicates removed → +0.3
- format fixed → +0.2
- normalized → +0.2

Score is always clamped to `[0.0, 1.0]`.

## Reward function (per step)

In `env.py`, each `step()` returns a reward in `[0.0, 1.0]`:

- correct action → +0.2
- wrong action → -0.1
- redundant action → -0.05

When the episode ends (`done=True`), it also adds the **final grader score** to that last step’s reward (then clamps to `[0,1]`).

## Setup (Windows)

From your workspace root:

```powershell
cd "project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python inference.py
```

## Optional: Use OpenAI (instead of fallback rules)

If you want the agent to ask OpenAI for the next action:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
python inference.py
```

You can also change the model:

```powershell
$env:OPENAI_MODEL="gpt-4.1-mini"
python inference.py
```

## Docker run

From the workspace root:

```bash
docker build -t data-cleaning-env ./project
docker run --rm data-cleaning-env
```

## Example output (what you should see)

You’ll see logs like:

- which task is running
- each step’s action, reward, and info message
- final grader score and cleaned dataset

