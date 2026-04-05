---
title: RPOE - Rotary Parking Optimization Environment
emoji: 🅿️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# RPOE — Rotary Parking Optimization Environment

A sequential decision-making environment inspired by the **KBR Park vertical rotary parking system** in Jubilee Hills, Hyderabad. An AI agent controls 7 independent 12-slot rotating wheels, deciding when to park, retrieve, and rotate under stochastic car arrival demand.

> To our knowledge, RPOE is the first OpenEnv environment modelling South Asian urban parking infrastructure and vertical rotary mechanical systems.

---

## Real-World Motivation

KBR Park operates a vertical rotary car-parking tower: cars are loaded onto a rotating wheel, which must be rotated to bring the target slot to the front access point before a car can be parked or retrieved. The system must serve a continuous stream of arrivals and retrievals under time-varying demand, minimising queue overflow and mechanical rotation cost. RPOE models 7 stacks at full fidelity, each with 12 slots.

![KBR Park Vertical Rotary Parking System](./assets/kbr_park_rotary.webp)
*The actual KBR Park rotary parking facility during trial run (June 2025) — 15m tall, 72 slots across 6 stacks, each stack holding 12 cars. Photo: Deccan Chronicle / Nabinder Bommala.*

![KBR Park Rotary Parking Towers — Street View](./assets/kbr_park_rotary.jpg)
*All 6 rotary stacks viewed from the street outside KBR Park, Jubilee Hills, Hyderabad. Photo: Telangana Today.*

> **Note on scale:** The physical KBR Park system uses 72 slots across multiple stacks. This environment now models 7 full 12-slot stacks, for 84 simulated slots in total.

**What a trained agent here generalises to:**
- Warehouse Automated Storage and Retrieval Systems (AS/RS)
- Elevator scheduling under stochastic call arrivals
- Circular buffer management in operating systems
- Any FIFO-constrained mechanical scheduling problem

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run inference (LLM + heuristic hybrid agent)
uv run python inference.py
```

Set environment variables before running:

```bash
export HF_TOKEN=<your-api-key>
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
```

---

## Docker

```bash
# Build
docker build -t rpoe-env:latest .

# Run the environment server
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME=gpt-4o-mini \
  rpoe-env:latest
```

The server exposes the OpenEnv HTTP API at `http://localhost:7860`.

---

## Action Space

5 discrete actions:

| Action | Value | Description | Valid When |
|---|---|---|---|
| `rotate_cw` | `"rotate_cw"` | Rotate wheel clockwise by one slot | Always |
| `rotate_ccw` | `"rotate_ccw"` | Rotate wheel anticlockwise by one slot | Always |
| `park` | `"park"` | Park front car from arrival queue | Front slot empty AND arrival queue non-empty |
| `retrieve` | `"retrieve"` | Retrieve car at front slot | Front slot occupied AND car matches `retrieval_queue[0]` |
| `idle` | `"idle"` | No-op; wait without taking any operation | Always |

Invalid actions (e.g. `park` when front slot is occupied) incur a −2.0 penalty but do not terminate the episode.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `slots` | `list[SlotState]` | 12 wheel slots. Index 0 = front/accessible slot. Each slot has `index`, `occupied`, `car_id`. |
| `front_slot_occupied` | `bool` | Whether slot 0 currently holds a car |
| `front_car_id` | `str \| null` | Car ID at the front slot, or null if empty |
| `arrival_queue` | `list[QueuedCar]` | Cars waiting to be parked (max 10). Each has `car_id`, `arrival_step`. |
| `retrieval_queue` | `list[PendingRetrieval]` | Cars requesting exit (max 10, strict FIFO). Each has `car_id`, `slot_index`, `requested_at_step`. |
| `step` | `int` | Current simulation step [0, 1080] |
| `hour` | `float` | Simulated hour-of-day [0.0, 18.0] |
| `total_parked` | `int` | Cumulative cars successfully parked this episode |
| `total_retrieved` | `int` | Cumulative cars successfully retrieved this episode |
| `total_overflowed` | `int` | Cumulative cars that timed out and left the queue |
| `last_action_valid` | `bool` | Whether the previous action was legal |

---

## Reward Function

Per-step reward is a sum of components:

| Component | Value | Rationale |
|---|---|---|
| `waiting_penalty` | −1.0 × (arrival queue len + retrieval queue len) | Penalises queue build-up every step; provides dense signal |
| `rotation_penalty` | −0.5 per rotation action | Models mechanical wear and energy cost |
| `park_bonus` | +2.0 on successful park | Positive signal for completing the primary task |
| `retrieval_bonus` | +3.0 on successful retrieval | Higher than park: retrieval is time-sensitive and harder (FIFO constraint) |
| `illegal_penalty` | −2.0 on invalid action | Deterrent, not catastrophic — agent can recover |
| `overflow_penalty` | −5.0 per car that times out | Highest penalty: lost customer, irreversible outcome |

Reward range: [−20.0, +5.0] per step.

---

## Tasks

### Task 1 — Easy: Rotation Efficiency (50 steps)

Low arrival rate (λ=0.1/step), no overflow risk. Agent is scored on how close to the optimal rotation path length it achieves when servicing park and retrieve operations.

```
Score = 0.6 × rotation_score + 0.4 × illegal_score
```

- Pass threshold: **0.40** (~1.5× random agent performance)
- Bypasses LLM — heuristic is optimal for this short horizon

### Task 2 — Medium: Peak-Hour Throughput (180 steps)

Morning peak simulation (6–9 AM, λ=0.35/step). Agent is scored on cars served vs cars that overflowed.

```
Score = served / (served + overflowed) − rotation_penalty
```

- Pass threshold: **0.50**

### Task 3 — Hard: Full 18-Hour Day (1080 steps)

Full day with time-varying Poisson arrivals (5 AM–11 PM). Composite score across four dimensions:

```
Score = 0.40 × throughput + 0.25 × efficiency + 0.20 × retrieval_ratio + 0.15 × queue_stability
```

- Pass threshold: **0.55**
- Thresholds are set at approximately 1.5× random agent performance across all tasks

---

## Baseline Scores

Scores for the hybrid LLM + heuristic agent (seed=42, `gpt-4o-mini`):

| Task | Score | Status | Notes |
|---|---|---|---|
| `task1_easy` | **1.0000** | PASS | 7 ops, 6 rotations, 0 illegal actions |
| `task2_medium` | **0.6667** | PASS | 46 served, 23 overflowed, 112 rotations |
| `task3_hard` | **0.8986** | PASS | 256 served, 44 overflowed; T=0.85 E=1.00 R=0.97 S=0.76 |
| **Average** | **0.8551** | | |

---

## Environment Dynamics

| Parameter | Value |
|---|---|
| Wheel size | 12 slots |
| Arrival process | Poisson with time-varying rate λ(t) |
| Dwell time | LogNormal(μ=60 steps, σ=30 steps) |
| Overflow timeout | 15 steps |
| Front slot constraint | Only slot 0 can park/retrieve |
| Retrieval order | Strict FIFO — must serve `retrieval_queue[0]` first |
| Episode seed | Configurable via `RPOE_SEED` env var (default: 42) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute action `{"action": "rotate_cw"}`, returns observation |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check — returns 200 |
| `/tasks` | GET | List all three evaluation tasks |
| `/task/{id}` | POST | Run named task with random agent |
| `/docs` | GET | OpenAPI/Swagger documentation |
| `/web` | GET | Interactive web UI |

---

## Project Structure

```
rpoe_env/
├── inference.py            # LLM hybrid agent, structured logs, baseline_scores.json
├── models.py               # ActionType (5), RPOEAction, RPOEObservation, TaskResult
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # requires-python>=3.14, version-pinned dependencies
├── Dockerfile              # python:3.14-slim, port 7860
├── baseline_scores.json    # Written by inference.py after run
├── server/
│   ├── app.py              # FastAPI app — /reset, /step, /state, /tasks, /task/{id}
│   └── env.py              # RotaryParkingEnv (7 wheels × 12 slots)
├── tasks/
│   └── graders.py          # run_task1/2/3, TASKS registry
└── tests/
    └── test_env.py         # 13 smoke tests
```

---

## Deploying to Hugging Face Spaces

```bash
openenv push
# or
openenv push --repo-id your-org/rpoe-env --private
```

The deployed Space includes a web interface at `/web` and the full OpenEnv HTTP API.
