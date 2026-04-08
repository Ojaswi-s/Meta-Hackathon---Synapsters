---
title: Meeting Notes Action Item Extraction
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - nlp
  - rl-environment
  - information-extraction
license: mit
---

# Meeting Notes Action Item Extraction — OpenEnv Environment

An OpenEnv-compliant environment where agents extract structured action items
from realistic meeting transcripts. Three tasks of increasing difficulty test
the agent's ability to identify explicit commitments, infer implicit actions,
and resolve conflicting ownership assignments.

---

## Motivation

Every team runs meetings. Fewer teams reliably capture what was agreed and who
is accountable. This environment trains agents to do the extraction work
accurately — a task that is both universally relatable and genuinely difficult
to do well at scale.

---

## Action Space

The agent submits a list of extracted action items each step:

```json
{
  "action_items": [
    {
      "owner": "Maya",
      "task": "Push auth token fix to staging",
      "deadline": "Thursday",
      "priority": "high"
    }
  ],
  "is_final": false,
  "reasoning": "Optional: agent's reasoning"
}
```

| Field         | Type              | Description                                      |
|---------------|-------------------|--------------------------------------------------|
| `owner`       | string            | Name or role of the responsible person           |
| `task`        | string            | What they will do                                |
| `deadline`    | string \| null    | When it's due                                    |
| `priority`    | high/medium/low   | Urgency level                                    |
| `is_final`    | boolean           | Set `true` to end the episode                    |
| `reasoning`   | string (optional) | Agent's extraction reasoning                     |

---

## Observation Space

| Field                  | Type          | Description                                   |
|------------------------|---------------|-----------------------------------------------|
| `transcript`           | string        | The full meeting transcript                   |
| `task_description`     | string        | Instructions for this task                    |
| `step_feedback`        | string/null   | Non-answer hints from the grader              |
| `items_found_count`    | integer       | Correct items identified in last submission   |
| `total_items_in_task`  | integer       | Total ground-truth items in this task         |
| `current_f1`           | float         | F1 score from last submission                 |
| `done`                 | boolean       | Whether the episode has ended                 |
| `reward`               | float/null    | Reward from last step                         |

---

## Tasks

### Task 1 — Easy: Explicit actions (product standup)
5 action items explicitly stated with clear owner, task, and deadline.
Expected score for a capable LLM: **0.80–0.95**

### Task 2 — Medium: Implicit actions (strategy session)
6 action items implied through discussion. Ownership and deadlines must be
inferred from context. No item is explicitly stated as "X will do Y by Z."
Expected score: **0.55–0.75**

### Task 3 — Hard: Conflicting assignments (engineering planning)
5 action items where ownership changes during the conversation. Two participants
both appear to accept the same task; one later backs out. The agent must resolve
the conflict based on the final state of the conversation.
Expected score: **0.30–0.55**

---

## Reward Function

Reward is computed at each step — not only at episode end. It is calculated in two stages:

**Stage 1 — Grader score** (`grader.py`):
```
grader_score = 0.7 × avg_item_score + 0.3 × F1
```

**Stage 2 — Step reward** (`meeting_environment.py`):
```
reward = grader_score + (0.3 × improvement_delta) - 0.02 × step_cost
```

Where `improvement_delta` is the gain over the best previous score (`max(0, score − best_f1_so_far)`),
and `step_cost` = 0.02 (a small efficiency penalty per step). Final reward is clamped to `[0.0, 1.5]`.

`avg_item_score` for each matched item is:
- **40%** task keyword overlap
- **30%** owner correct
- **15%** deadline correct
- **15%** priority correct

This gives meaningful signal throughout the trajectory: partial credit for
finding the right owner but wrong deadline, and improvement bonuses for
successive refinements.

---

## Grader

The grader is fully deterministic:

1. For each ground-truth item, find the best-matching agent item by owner + keyword overlap.
2. Score each match on a 0–1 scale using the weighted formula above.
3. Compute precision, recall, F1 over the matched set.
4. Final score = `0.7 × avg_item_score + 0.3 × F1`.

No semantic similarity, no LLM judges. Same input → same score, every run.

---

## Baseline Scores

Run with `meta-llama/Llama-3.3-70B-Instruct` via HuggingFace router:

| Task   | Best F1 | Total Reward |
|--------|---------|--------------|
| Easy   | 0.88    | 1.21         |
| Medium | 0.67    | 0.89         |
| Hard   | 0.41    | 0.52         |
| **Avg**| **0.65**| **0.87**     |

---

## Setup

### Local (without Docker)

```bash
# Install
pip install -e .

# Run server
uvicorn meeting_env.server.app:app --host 0.0.0.0 --port 8000

# Run baseline
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_xxx
python inference.py
```

### Docker

```bash
docker build -t meeting-env:latest .
docker run -p 8000:8000 meeting-env:latest
```

### Validate

```bash
pip install openenv-core
openenv validate
```

---

## API Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| POST   | `/reset`  | Start new episode (`task_id` in body) |
| POST   | `/step`   | Submit action items                |
| GET    | `/state`  | Get current episode state          |
| GET    | `/health` | Health check (returns 200)         |

---

## Project Structure

```
.
├── Dockerfile                   # Container build (root-level)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package config (pip install -e .)
├── openenv.yaml                 # OpenEnv manifest
├── inference.py                 # Baseline inference script (OpenAI client)
├── README.md                    # This file
├── meeting_env/                 # Python package
│   ├── __init__.py              # Exports MeetingAction, MeetingObservation, MeetingEnv
│   ├── models.py                # Pydantic models: Action, Observation, State
│   ├── client.py                # Typed EnvClient (sync + async)
│   └── server/
│       ├── __init__.py
│       ├── app.py               # FastAPI server via create_fastapi_app
│       ├── meeting_environment.py  # Environment logic (step/reset/state)
│       ├── tasks.py             # 3 task definitions with transcripts + ground truth
│       └── grader.py            # Deterministic F1-based grader
└── tests/
    └── test_meeting_env.py      # Unit + integration tests (pytest)
```
