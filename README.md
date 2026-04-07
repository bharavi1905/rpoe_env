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

A sequential decision-making environment modelling the **KBR Park vertical rotary parking system** in Jubilee Hills, Hyderabad. An AI agent controls 7 independent 12-slot rotating wheels, deciding when to park, retrieve, and rotate under stochastic time-varying car arrivals.

> To our knowledge, RPOE is the first OpenEnv environment modelling South Asian urban parking infrastructure and vertical rotary mechanical systems.

---

## Real-World Motivation

KBR Park operates vertical rotary parking towers: each tower stores 12 cars in the footprint of one bay. Only the **front slot** is accessible — the wheel must rotate to align the correct slot before any car can be parked or retrieved.

![KBR Park Vertical Rotary Parking System](./assets/kbr_park_rotary.webp)
*KBR Park rotary parking during trial run (June 2025) — 15m tall, 72 slots across 6 stacks. Photo: Deccan Chronicle / Nabinder Bommala.*

![KBR Park Rotary Parking Towers — Street View](./assets/kbr_park_rotary.jpg)
*All 6 stacks viewed from the street, Jubilee Hills, Hyderabad. Photo: Telangana Today.*

> **Note on scale:** The physical system has 72 slots across 6 stacks. RPOE models 7 stacks × 12 slots = 84 slots to add an extra scheduling dimension for multi-wheel coordination.

This is a hard real-time scheduling problem: arrivals are stochastic, retrieval is strict FIFO, waiting cars leave after 15 minutes, and 7 wheels compete for the agent's single action per step. A suboptimal operator loses customers to overflow and wastes energy on unnecessary rotations. An AI agent can learn to anticipate demand, minimise rotation cost, and sustain throughput across an 18-hour day.

**Generalises to:** warehouse AS/RS systems, elevator scheduling, circular buffer management, and any FIFO-constrained mechanical scheduling problem.

---

## Quick Start

```bash
uv sync

export HF_TOKEN=<your-api-key>
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini

uv run python inference.py
```

**Docker:**
```bash
docker build -t rpoe-env:latest .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN -e MODEL_NAME=gpt-4o-mini rpoe-env:latest
```

---

## Traffic Simulation

Each step = 1 minute. Arrivals follow a **time-varying Poisson process** scaled by `TRAFFIC_MULTIPLIER=1.5`:

| Period | Hours | Base λ | Effective λ (1.5×) |
|---|---|---|---|
| Early morning | 5–6 AM | 0.05 | 0.075 |
| **Morning peak** | 6–9 AM | **0.35** | **0.525** |
| Midday trickle | 9 AM–5 PM | 0.10 | 0.150 |
| **Evening peak** | 5–8 PM | **0.25** | **0.375** |
| Wind-down | 8–11 PM | 0.05 | 0.075 |

Cars park for a **LogNormal dwell time** (mean 60 steps ≈ 1 hr, std 30 steps), then join the retrieval queue. Any car waiting in the arrival queue for more than **15 steps** rage-quits — a −5.0 penalty with no recovery.

---

## Action Space

| Action | Value | What it does | Valid when |
|---|---|---|---|
| Rotate CW | `"rotate_cw"` | Shift wheel one slot clockwise (slot 11 → front) | wheel_index valid |
| Rotate CCW | `"rotate_ccw"` | Shift wheel one slot anticlockwise (slot 0 → back) | wheel_index valid |
| Park | `"park"` | Move first car from arrival queue into front slot | Front slot **empty** AND arrival queue non-empty |
| Retrieve | `"retrieve"` | Remove car at front slot | Front slot matches **retrieval_queue[0]** (strict FIFO) |
| Idle | `"idle"` | No-op | Always |

To bring slot *k* to the front: CW needs `(12 − k) mod 12` steps, CCW needs `k` steps — always take the shorter path. Invalid actions cost −2.0 but do not end the episode.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `wheels` | `list[WheelState]` | Per-wheel: `wheel_index`, `front_slot_occupied`, `front_car_id`, `slots` (12 × `SlotState`) |
| `slots` | `list[SlotState]` | Flattened 84-slot view. Index 0 per wheel = front (accessible) slot |
| `arrival_queue` | `list[QueuedCar]` | Cars waiting to park (max 10). Contains `car_id`, `arrival_step` |
| `retrieval_queue` | `list[PendingRetrieval]` | Cars requesting exit (max 10, strict FIFO). Contains `car_id`, `wheel_index`, `slot_index` |
| `step` | `int` | Current step [0, 1080] |
| `hour` | `float` | Simulated hour [0.0, 18.0] — 0 = 5 AM, 18 = 11 PM |
| `total_parked` | `int` | Cumulative successful parks |
| `total_retrieved` | `int` | Cumulative successful retrievals |
| `total_overflowed` | `int` | Cumulative cars that timed out |
| `reward` / `reward_breakdown` | `float` / `RewardBreakdown` | Step reward and per-component breakdown |
| `done` | `bool` | Episode ended |

---

## Reward Function

| Component | Value | Rationale |
|---|---|---|
| `waiting_penalty` | −1.0 × (arrival queue + retrieval queue length) | Dense per-step signal; pressures agent to keep queues short |
| `rotation_penalty` | −0.5 per rotation | Models mechanical wear and energy cost |
| `park_bonus` | +2.0 | Completing a park operation |
| `retrieval_bonus` | +3.0 | Higher than park — FIFO constraint makes retrieval mechanically harder |
| `illegal_penalty` | −2.0 | Deterrent; recoverable |
| `overflow_penalty` | −5.0 per car | Highest penalty — lost customer, irreversible |

Reward range per step: approximately **[−22.5, +3.0]**. Only one action fires per step, so bonuses cannot combine.

---

## Tasks and Graders

All tasks run with `seed=42` for full reproducibility.

---

### Task 1 — Easy: Rotation Efficiency (50 steps)

**Simulates:** Quiet opening hour · λ = 0.15/step · No overflow risk

Tests whether the agent takes the shortest rotation path when servicing operations, and avoids illegal actions.

```
rotation_score = max(0, 1 − max(0, actual_rotations − expected_rotations) / expected_rotations)
illegal_score  = max(0, 1 − illegal_count / 10)
score          = 0.6 × rotation_score + 0.4 × illegal_score
```

`expected_rotations = successful_ops × 3` (average 3 rotations per op for uniformly distributed slots).

**Pass threshold: 0.40** (≈1.5× random agent) · **LLM bypassed** — heuristic is optimal at this horizon.

---

### Task 2 — Medium: Peak-Hour Throughput (180 steps)

**Simulates:** Morning peak 6–9 AM · λ = 0.525/step · Overflow pressure

Tests sustained throughput under high arrival rate. With a 15-step overflow timeout, the agent must continuously drain the arrival queue.

```
throughput_score = served / (served + overflowed)       where served = parked + retrieved
rotation_penalty = min(0.30, max(0, (rotations − served × 4) / (served × 4)))
score            = throughput_score − rotation_penalty
```

Rotation penalty is capped at 0.30 — efficiency matters but throughput dominates.

**Pass threshold: 0.50** (≈1.5× random agent)

---

### Task 3 — Hard: Full 18-Hour Day (1080 steps)

**Simulates:** 5 AM–11 PM full operating day · λ = 0.075–0.525/step (time-varying)

Tests the agent across all demand bands. Rewards dynamic priority-switching: aggressive parking during peaks, aggressive retrieval during troughs to prevent the retrieval queue from saturating at its 10-car cap.

```
throughput = served / (served + overflowed)                           [40%]
efficiency = max(0, 1 − max(0, rotations / (served × 4) − 1))        [25%]
retrieval  = total_retrieved / total_parked                            [20%]
stability  = max(0, 1 − avg_queue_length / 10)                        [15%]
score      = 0.40 × throughput + 0.25 × efficiency + 0.20 × retrieval + 0.15 × stability
```

**Why retrieval matters:** the retrieval queue has a hard cap of 10. If it fills, new retrieval requests are silently dropped — the car is physically stranded. An agent that only parks will collapse on this dimension.

**Pass threshold: 0.55** (≈1.5× random agent)

---

## Baseline Scores

Hybrid LLM + heuristic agent · `seed=42` · `gpt-4o-mini` · `TRAFFIC_MULTIPLIER=1.5`:

| Task | Steps | Traffic | Score | Status |
|---|---|---|---|---|
| `task1_easy` | 50 | λ=0.15/step | **0.9990** | PASS |
| `task2_medium` | 180 | λ=0.525/step (peak) | **0.6796** | PASS |
| `task3_hard` | 1080 | λ=0.075–0.525/step | **0.8160** | PASS |
| **Average** | | | **0.8315** | runtime: 358.8s |

Task 3 breakdown: Throughput 0.80 · Efficiency 1.00 · Retrieval 0.90 · Stability 0.45 · avg queue 5.52

---

## Training an RL Policy

RPOE is designed as a drop-in RL training environment. The reward is dense (waiting penalty fires every step), episodes are configurable in length, and full determinism under a fixed seed makes evaluation reproducible.

**Recommended approach:** PPO with action masking — mask illegal actions to −∞ before sampling. Use the three tasks as a natural curriculum: Task 1 (mechanics) → Task 2 (peak pressure) → Task 3 (full-day generalisation).

---

## Environment Parameters

| Parameter | Value |
|---|---|
| Wheels | 7 independent rotary stacks |
| Slots per wheel | 12 (84 total) |
| Max arrival queue | 10 cars |
| Max retrieval queue | 10 cars (hard cap — overflow = stranded car) |
| Overflow timeout | 15 steps |
| Dwell time | LogNormal(μ=60, σ=30 steps) |
| Episode length | 1080 steps (18-hour day) |
| Traffic multiplier | 1.5× |
| Accessible slot | Index 0 per wheel (front only) |
| Retrieval order | Strict FIFO |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | `{"action": {"action": "rotate_cw", "wheel_index": 0}}` |
| `/state` | GET | Full serialisable environment state |
| `/health` | GET | 200 OK |
| `/tasks` | GET | List evaluation tasks |
| `/task/{id}` | POST | Run named task |
| `/docs` | GET | OpenAPI / Swagger |
| `/web` | GET | Interactive web UI |

---

## Project Structure

```
rpoe_env/
├── inference.py        # LLM hybrid agent, structured logs, baseline_scores.json
├── models.py           # ActionType (5), RPOEAction, RPOEObservation, TaskResult
├── openenv.yaml        # OpenEnv manifest
├── pyproject.toml      # requires-python>=3.14, pinned dependencies
├── Dockerfile          # python:3.14-slim, port 7860
├── server/
│   ├── app.py          # FastAPI — /reset /step /state /tasks /task/{id}
│   └── env.py          # RotaryParkingEnv (7 wheels × 12 slots)
├── tasks/
│   └── graders.py      # run_task1/2/3, TASKS registry
└── tests/
    └── test_env.py
```

---

## Deploy to Hugging Face Spaces

```bash
openenv push
```
