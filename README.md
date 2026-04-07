# Data Cleaning Agent Environment (OpenEnv)

This is a complete, real-world OpenEnv environment that simulates a **data cleaning job**. It is designed to evaluate an AI agent's ability to take messy, unstructured datasets and correctly apply transformation operations to clean them, scoring the result against deterministic graders.

## Motivation & Real-World Task Simulation
Cleaning data is one of the most common and labor-intensive real-world tasks performed by data scientists and software engineers. This environment mirrors that reality, moving beyond typical "game" environments to present a structured but dynamic dataset. Agents must identify null values, duplicate records, non-standard formatting (like currencies or dates), and normalization targets, making it an excellent benchmark for LLM reasoning capabilities.

---

## Environment Spaces

### Observation Space
The observation space is returned at every `reset()` and `step()` call:
- `task_id` (string): Current task identifier.
- `difficulty` (string): Expected difficulty ('easy', 'medium', 'hard').
- `step_count` (int): Number of steps taken in the current episode.
- `max_steps` (int): The maximum allowed steps before truncation.
- `dataset_preview` (array of dicts): A JSON preview of the top rows.
- `null_counts` (dict): Feature-to-null-count mapping mapping.
- `duplicate_count` (int): Detected duplicate rows.
- `format_issues` (dict): Detected string formatting problems (e.g. currencies).
- `normalized_columns` (array): List of already normalized column names.

### Action Space
Discrete and parameterized actions to modify the dataset in-place:
- `{"type": "remove_nulls"}`: Drops any row containing missing values.
- `{"type": "fill_mean"}`: Imputes numeric missing values with column means.
- `{"type": "remove_duplicates"}`: Removes duplicate rows from the dataset.
- `{"type": "fix_format"}`: Parses standard dates and currency strings into valid formats (e.g. `$1,000` -> `1000.0`).
- `{"type": "normalize_column", "column": "salary"}`: Min-max normalizes a specific feature column onto `[0, 1]`.

---

## Grader Tasks (0.0 to 1.0)
The environment provides 3 built-in tasks with deterministic success/failure criteria.

| Task ID | Difficulty | Goal | Required Actions |
| --- | --- | --- | --- |
| **easy** | Easy | Fix missing numeric values. | `fill_mean` |
| **medium** | Medium | Fix missing values & remove duplicates. | `fill_mean`, `remove_duplicates` |
| **hard** | Hard | Fix nulls, duplicates, format issues, and normalize the salary column. | `fill_mean`, `remove_duplicates`, `fix_format`, `normalize_column` |

*The reward function provides dense partial signals:* Correct valid actions yield +0.2, wrong actions yield -0.1, and redundant actions yield -0.05. At termination, the environment adds the final grader score (`grade_dataset()`) which evaluates presence of nulls, duplicates, and correct ranges.

---

## Setup & Deployment Instructions

### Local Execution (FastAPI)
This project uses FastAPI to comply with the Hugging Face Space constraints.
```bash
python -m venv .venv
# Activate environment (Windows)
.\.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```
This will start your web server exposing `/reset`, `/step` and `/state` APIs!

### Running the Inference Baseline
The baseline agent uses a standard OpenAI API client. Ensure your credentials are set before running:
```powershell
$env:HF_TOKEN="YOUR_HUGGINGFACE_WRITE_TOKEN"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" # Or your preference

python inference.py
```
*(Note: If OpenAI credentials fail, it falls back to an automatic rule-based baseline.)*

### Baseline Scores (Reproducibility)
Using the baseline inference script, the agent accurately reproduces a perfect score across tasks:
- **Easy**: `0.999` Score
- **Medium**: `0.999` Score
- **Hard**: `0.999` Score

### Containerized Execution (Docker)
The Docker image encapsulates the FastAPI backend directly, mapping to the 7860 port Hugging Face Spaces expect.
```bash
docker build -t openenv-data-cleaning .
docker run -p 7860:7860 --rm openenv-data-cleaning
```
Once running, ping `http://localhost:7860/` for a 200 health check response.