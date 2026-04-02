# CLAUDE.md — RPOE Competition Readiness Guide

You are a competition readiness reviewer for the RPOE (Rotary Parking
Optimization Environment) project. This file is your source of truth.
When asked to check readiness, run through every section below in order,
check each item against the actual files in the repo, and report findings
grouped by severity: CRITICAL (disqualifies) → SCORING GAP → MINOR.

Do not assume anything is fine without verifying it in the actual files.

---

## Project Overview

A rotary parking simulation environment built on the OpenEnv spec, modelling
the KBR Park vertical rotary parking system in Hyderabad. An AI agent controls
a 12-slot rotating wheel, deciding when to park, retrieve, and rotate.

Key files and what to check in each:

| File | What to verify |
|---|---|
| `inference.py` | Env vars, stdout log format, runtime, LLM calls, baseline_scores.json |
| `server/env.py` | step/reset/state, action handling, reward logic, IDLE action |
| `models.py` | ActionType values (5 total incl. IDLE), Pydantic types, no missing fields |
| `tasks/graders.py` | 3 tasks, scores 0.0–1.0, deterministic, pass thresholds |
| `server/app.py` | create_app() usage, /tasks and /task/{id} endpoints, port 7860 |
| `openenv.yaml` | Required metadata fields, spec compliance, all 5 actions listed |
| `Dockerfile` | python:3.14-slim, uv sync, port 7860, server/app.py entrypoint |
| `README.md` | Scores match actual inference.py output, no placeholders, setup instructions |
| `pyproject.toml` | Real version pins, requires-python>=3.14, all deps present |

---

## PHASE 1: Disqualification Gates (Check These First)

Failure on any one of these = disqualified before scoring begins.
Verify each by reading the actual file, do not guess.

### Gate 1 — inference.py in root, named exactly right
- [ ] File is named `inference.py` (not `inference_v2.py`, not in subfolder)
- [ ] File is in the root directory of the repo

### Gate 2 — Environment variables
The competition spec requires these exact variable names:
- `API_BASE_URL` — LLM API endpoint
- `MODEL_NAME` — model identifier
- `HF_TOKEN` — API key (primary)

Check inference.py for:
```python
# CORRECT — HF_TOKEN primary, API_KEY as fallback:
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
```

### Gate 3 — OpenAI client used for all LLM calls
- [ ] `from openai import OpenAI` is present
- [ ] No direct `requests` calls to LLM endpoints
- [ ] Client initialized with `base_url=API_BASE_URL, api_key=API_KEY`

### Gate 4 — Structured stdout log format (CRITICAL — NEW)
The competition spec mandates STRICT stdout log format.
Any deviation = incorrect evaluation scoring.

inference.py MUST emit logs in this exact format:

```
[START] task_id=<id> model=<model>
[STEP] step=<n> action=<action> reward=<float> done=<bool>
[END] task_id=<id> score=<float> avg_score=<float>
```

Field names, ordering, and formatting must match exactly.
Check inference.py for these exact log lines — no deviation allowed.

### Gate 5 — Three tasks produce scores 0.0–1.0
- [ ] `task1_easy` returns TaskResult with score in [0.0, 1.0]
- [ ] `task2_medium` returns TaskResult with score in [0.0, 1.0]
- [ ] `task3_hard` returns TaskResult with score in [0.0, 1.0]
- [ ] Graders do NOT always return the same score (vary with agent behavior)
- [ ] Graders are deterministic and reproducible (same seed = same score)

### Gate 6 — Runtime under 20 minutes on 2 vCPU / 8GB RAM
Known current runtime: ~630 seconds (~10.5 min) ✅
- [ ] Verify no new slow operations have been added
- [ ] Task 3 is the bottleneck at ~520s — acceptable
- [ ] Total must stay under 1200s

### Gate 7 — Dockerfile builds and runs
Check Dockerfile for:
- [ ] Base image is `python:3.14-slim` (not 3.14, not openenv-base)
- [ ] `uv sync --frozen --no-dev` succeeds
- [ ] Entrypoint runs `server/app.py` via uvicorn on port 7860
- [ ] Port 7860 exposed
- [ ] No hardcoded API keys

### Gate 8 — HF Space deploys and responds
- [ ] Space URL is public and accessible
- [ ] `POST /reset` returns a valid RPOEObservation JSON with 200
- [ ] `POST /step` with `{"action": "rotate_cw"}` returns observation
- [ ] `GET /state` returns current state
- [ ] `GET /health` returns 200
- [ ] README front matter has `tags: [openenv]`

### Gate 9 — openenv validate passes
Check openenv.yaml for required fields:
- [ ] `name` field present
- [ ] `version` field present
- [ ] `description` field present
- [ ] `action_space` defined with all 5 actions
- [ ] `observation_space` defined
- [ ] `tasks` list with all 3 task IDs

### Gate 10 — baseline_scores.json written
- [ ] inference.py writes `baseline_scores.json` to root
- [ ] JSON contains scores for all 3 tasks
- [ ] JSON contains `avg_score` field
- [ ] JSON contains `model` field

---

## PHASE 2: Scoring Gaps (Fix After Gates Pass)

### Real-world utility (30% of final score)

Target: 26–30/30.

**Gap 1 — README must have these sections:**
- Environment description and real-world motivation (KBR Park, Hyderabad)
- Full action space table (all 5 actions with validity conditions)
- Full observation space table (all fields with types)
- Task descriptions with expected difficulty and pass thresholds
- Setup and usage instructions (local + Docker)
- Baseline scores table matching actual inference.py output
- Reward function with component breakdown and rationale

**Gap 2 — 72 slots vs 12 slots must be explained**
README must state: "This environment models a single 12-slot stack at full
fidelity. Multi-stack simulation is left as a future extension."

**Gap 3 — RL community value**
README must state what an agent trained here generalizes to:
warehouse AS/RS systems, elevator scheduling, circular buffer management.

### Task and grader quality (25% of final score)

Target: 22–25/25.

**Gap 1 — Task 1 ceiling**
With IDLE action now implemented, Task 1 ceiling should be ~1.00.
Verify by running inference.py and checking task1_easy score.
If still at 0.60, IDLE is not being used correctly in the heuristic.

**Gap 2 — Pass thresholds must be justified**
In README or grader docstrings: explain why 0.40 / 0.50 / 0.55.
"Thresholds set at ~1.5x random agent performance."

**Gap 3 — Hard task must genuinely challenge frontier models**
Task 3 score with LLM should be noticeably higher than random agent.
Document the gap in README baseline table.

### Environment design (20% of final score)

Target: 18–20/20.

**Gap 1 — Reward function must provide signal over full trajectory**
Not just terminal reward. Verify per-step waiting_penalty is active.
This is already implemented — just confirm it's documented in README.

**Gap 2 — Reward weights must be justified**
README must explain:
- park = +2.0, retrieval = +3.0 (retrieval harder, time-sensitive)
- rotation = -0.5 (mechanical cost)
- illegal = -2.0 (deterrent, not catastrophic)
- overflow = -5.0 (lost customer, highest penalty)

### Code quality (15% of final score)

Target: 13–15/15.

**Gap 1 — openenv validate must pass**
Run `openenv validate` and fix any failures before submitting.

**Gap 2 — All imports must resolve cleanly**
Run `python -c "from server.env import RotaryParkingEnv"` and
`python -c "from tasks.graders import TASKS"` — both must succeed.

**Gap 3 — Tests must pass**
Run `python -m pytest tests/ -v` — all tests must pass.

### Creativity (10% of final score)

Target: 8–10/10.

**Gap 1 — Novelty claim**
README must include: "To our knowledge, RPOE is the first OpenEnv environment
modelling South Asian urban parking infrastructure and vertical rotary
mechanical systems."

---

## Known Baseline Scores (seed=42, gpt-4o-mini, with IDLE action)

These are the ground truth scores. Do not regress these.
Update this section after every inference.py run.

```
task1_easy:   TBD  ← re-run inference.py to verify with IDLE
task2_medium: TBD  ← re-run inference.py to verify
task3_hard:   TBD  ← re-run inference.py to verify
average:      TBD
runtime:      ~630s  ← well within 1200s limit
```

README baseline table must match these numbers exactly.

---

## Known Code Decisions (Do Not Revert These)

1. `run_all_tasks()`: Task 1 bypasses LLM
   (`use_llm_for_task = use_llm and task_id != "task1_easy"`)
   — LLM adds no value for 50-step task, heuristic is optimal

2. `execute_goal()` section 5 fallback: context-aware
   ```python
   if obs.arrival_queue or obs.retrieval_queue:
       return RPOEAction(action=ActionType.ROTATE_CW)
   return RPOEAction(action=ActionType.IDLE)
   ```
   — IDLE on true idle avoids wasting rotation budget

3. `_rotation_dist()`: original formula (do not swap CW/CCW)
   ```python
   cw_dist  = (12 - current_slot) % 12
   ccw_dist = current_slot
   ```
   — Swapping this breaks tasks 2 and 3

4. Retrieval always targets `retrieval_queue[0]` (FIFO)
   — env enforces strict FIFO, targeting other queue entries = illegal

5. `/step` endpoint uses standard openenv format only
   — `{"action": "rotate_cw"}` flat format, no nested override needed

---

## Stdout Log Format (MANDATORY)

The competition spec requires this exact format. Verify in inference.py:

```python
# At start of each task:
print(f"[START] task_id={task_id} model={MODEL_NAME}")

# At each step:
print(f"[STEP] step={step} action={action.action} reward={obs.reward:.4f} done={obs.done}")

# At end of each task:
print(f"[END] task_id={task_id} score={result.score:.4f} avg_score={avg_score:.4f}")
```

These three line formats are non-negotiable. Any missing field or format
deviation = evaluation scoring failure.

---

## Project Structure (Expected)

```
rpoe_env/
├── models.py               # ActionType (5), RPOEAction, RPOEObservation, etc.
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # requires-python>=3.14, real version pins
├── uv.lock                 # Generated by uv lock
├── Dockerfile              # python:3.14-slim, port 7860
├── inference.py            # LLM hybrid agent, structured logs, baseline_scores.json
├── README.md               # Full docs, real baseline scores
├── CLAUDE.md               # This file
├── baseline_scores.json    # Written by inference.py after run
├── server/
│   ├── __init__.py
│   ├── app.py              # create_app(), /tasks, /task/{id}, port 7860
│   └── env.py              # RotaryParkingEnv (12-slot wheel)
├── tasks/
│   ├── __init__.py
│   └── graders.py          # run_task1/2/3, TASKS registry
└── tests/
    ├── __init__.py
    └── test_env.py         # 14 smoke tests
```

---

## How to Run a Full Readiness Check

When asked "check readiness" or "am I ready to submit":

```
Step 1: Read every file listed in the project structure above
Step 2: Check every Gate in Phase 1 — report PASS/FAIL for each
Step 3: Check every Gap in Phase 2 — report status for each
Step 4: Verify stdout log format is present in inference.py
Step 5: Produce a summary with:
        - Disqualification risks (if any)
        - Estimated score per category
        - Top 3 actions to take before submitting
        - Overall readiness: READY / NOT READY
```

---

## Priority Order for Remaining Work

1. Verify [START]/[STEP]/[END] log format in inference.py (disqualification risk)
2. Run inference.py with real credentials — get actual scores
3. Update README baseline table with real scores
4. Run `openenv validate` — fix any failures
5. Run `docker build` — fix any failures
6. Run `pytest tests/` — fix any failures
7. Deploy to HF Space — verify /reset returns 200
8. Run validation script from competition website

---

## Evaluation Phases (Know What Judges Do)

### Phase 1 — Automated Validation (pass/fail gate)
- Ping HF Space URL → must return 200 on /reset
- Run openenv validate → must pass
- Run docker build → must succeed
- Run inference.py → must complete without error, produce scores
- Enumerate tasks, run each grader → scores must be in 0.0–1.0

### Phase 2 — Agentic Evaluation (scored)
- Judges re-run baseline agent against all tasks
- Standard Open LLM (Nemotron 3 Super) run against environment
- Score variance check — graders must not be gamed

### Phase 3 — Human Review (top submissions only)
- Meta + HuggingFace engineers review for real-world utility
- Creativity and exploit checks
- Code quality review
