"""
inference.py — RPOE Hybrid Inference Script
============================================
Hybrid decision system: LLM sets high-level goals, deterministic executor
handles step-level rotation/park/retrieve actions.

LLM call reduction: ~1080 → ~50-100 calls for Task 3.

Required env vars:
  API_BASE_URL  — LLM API endpoint (default: OpenAI)
  MODEL_NAME    — model identifier  (default: gpt-4o-mini)
  HF_TOKEN      — API key for the LLM provider

Usage:
  python inference.py
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Bootstrap path so we can import without installing
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from server.env import RotaryParkingEnv
from models import RPOEAction, ActionType, RPOEObservation, TaskResult
from tasks.graders import TASKS

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "8"))
LLM_RETRIES = max(1, int(os.environ.get("LLM_RETRIES", "1")))
LLM_BACKOFF_SECONDS = float(os.environ.get("LLM_BACKOFF_SECONDS", "0.25"))

if not HF_TOKEN:
    print("[WARN] HF_TOKEN not set — will run heuristic baseline only.")

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# System prompt — goal-level decisions only
# ---------------------------------------------------------------------------

GOAL_SYSTEM_PROMPT = """You are a high-level planner for a rotary parking system.

A deterministic executor handles step-level rotation and action execution.
Your job is only to decide WHAT to do next, not HOW to rotate.

STATE (JSON):
- wheel_fronts: list of {wheel_index, front_occupied, front_car_id}
- arrival_queue_len: number of cars waiting to park
- retrieval_queue: list of {car_id, wheel_index, slot} — cars requesting exit
- empty_slots: number of empty slots on the wheel
- step, hour: current simulation time

GOALS — output exactly one:
    "retrieve"
    "park"
    "idle"

DECISION STRATEGY (follow in order):
1. If retrieval_queue[0] exists and its car is already at the front of its wheel -> choose "retrieve"
2. Else if arrival_queue_len > 0 and any wheel front is empty -> choose "park"
3. Else if retrieval_queue is non-empty -> choose "retrieve"
4. Else if arrival_queue_len > 0 -> choose "park"
5. Else -> choose "idle"

Respond with ONLY a JSON object: {"goal": "<goal>", "target_wheel": <int or null>, "target_slot": <int or null>}
- For "retrieve": set target_wheel and target_slot from retrieval_queue[0]
- For "park": set target_wheel to a wheel with capacity and target_slot to an empty slot index on that wheel
- For "idle": set both target_wheel and target_slot to null

No explanation. No markdown. Just the raw JSON.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SLOTS = 12


def _get_wheel(obs: RPOEObservation, wheel_index: int):
    return obs.wheels[wheel_index]


def _first_front_empty_wheel(obs: RPOEObservation) -> Optional[int]:
    for wheel in obs.wheels:
        if not wheel.front_slot_occupied:
            return wheel.wheel_index
    return None


def _best_parking_candidate(obs: RPOEObservation) -> tuple[Optional[int], Optional[int], Optional[ActionType]]:
    best_choice: tuple[int, int, int, ActionType] | None = None

    for wheel in obs.wheels:
        empty_slots = [slot.index for slot in wheel.slots if not slot.occupied]
        if not empty_slots:
            continue

        if not wheel.front_slot_occupied:
            return wheel.wheel_index, 0, None

        wheel_size = len(wheel.slots)
        for target_slot in empty_slots:
            cw_dist = (wheel_size - target_slot) % wheel_size
            ccw_dist = target_slot
            if cw_dist <= ccw_dist:
                direction = ActionType.ROTATE_CW
                distance = cw_dist
            else:
                direction = ActionType.ROTATE_CCW
                distance = ccw_dist

            candidate = (distance, target_slot, wheel.wheel_index, direction)
            if best_choice is None or candidate < best_choice:
                best_choice = candidate

    if best_choice is None:
        return None, None, None
    _, target_slot, wheel_index, direction = best_choice
    return wheel_index, target_slot, direction


# ---------------------------------------------------------------------------
# Executor — rotation-minimizing
# ---------------------------------------------------------------------------

def execute_goal(obs: RPOEObservation, goal: dict) -> RPOEAction:
    """
    Rotation-minimizing executor:
    - Act immediately if the front slot matches what we need.
    - Otherwise rotate toward the CLOSEST useful target (by rotation distance),
      not the lowest index.
    """

    # 1. RETRIEVE immediately if the right car is at front
    if obs.retrieval_queue:
        retrieval = obs.retrieval_queue[0]
        retrieval_wheel = _get_wheel(obs, retrieval.wheel_index)
        if retrieval_wheel.front_car_id == retrieval.car_id:
            return RPOEAction(action=ActionType.RETRIEVE, wheel_index=retrieval.wheel_index)

    # 2. PARK immediately if front is empty and cars are waiting
    if obs.arrival_queue:
        front_empty_wheel = _first_front_empty_wheel(obs)
        if front_empty_wheel is not None:
            return RPOEAction(action=ActionType.PARK, wheel_index=front_empty_wheel)

    # 3. HANDLE RETRIEVAL MOVEMENT
    if obs.retrieval_queue:
        retrieval = obs.retrieval_queue[0]
        current_slot = retrieval.slot_index
        wheel_index = retrieval.wheel_index

        cw_dist = (NUM_SLOTS - current_slot) % NUM_SLOTS
        ccw_dist = current_slot

        if cw_dist <= ccw_dist:
            return RPOEAction(action=ActionType.ROTATE_CW, wheel_index=wheel_index)
        return RPOEAction(action=ActionType.ROTATE_CCW, wheel_index=wheel_index)

    # 4. HANDLE PARKING MOVEMENT
    if obs.arrival_queue:
        wheel_index, _, direction = _best_parking_candidate(obs)
        if wheel_index is not None and direction is not None:
            return RPOEAction(action=direction, wheel_index=wheel_index)

    # =========================
    # 5. CONTEXT-AWARE FALLBACK
    # =========================
    # Active system: rotate if queues present
    # Truly idle: use IDLE action to avoid wasting budget and incurring illegal penalties
    if obs.arrival_queue or obs.retrieval_queue:
        fallback_wheel = goal.get("target_wheel_index")
        wheel_count = len(obs.wheels)
        if not isinstance(fallback_wheel, int) or fallback_wheel < 0 or fallback_wheel >= wheel_count:
            fallback_wheel = 0
        return RPOEAction(action=ActionType.ROTATE_CW, wheel_index=fallback_wheel)
    return RPOEAction(action=ActionType.IDLE, wheel_index=None)


# ---------------------------------------------------------------------------
# Goal validity and completion checks
# ---------------------------------------------------------------------------

def _is_goal_valid(obs: RPOEObservation, goal: dict) -> bool:
    """Return False if the goal can no longer be executed."""
    g = goal.get("goal", "idle")
    if g == "retrieve":
        return len(obs.retrieval_queue) > 0
    if g == "park":
        has_arrivals = len(obs.arrival_queue) > 0
        has_empty    = any(not s.occupied for s in obs.slots)
        return has_arrivals and has_empty
    return True  # idle is always valid


def _is_goal_complete(obs: RPOEObservation, goal: dict) -> bool:
    """Return True if the goal has been achieved and a new one is needed."""
    g = goal.get("goal", "idle")
    if g == "retrieve":
        initial_len = goal.get("initial_retrieval_len")
        if not obs.retrieval_queue:
            return True
        if isinstance(initial_len, int):
            return len(obs.retrieval_queue) < initial_len
        return False
    if g == "park":
        initial_len = goal.get("initial_arrival_len")
        if isinstance(initial_len, int):
            return len(obs.arrival_queue) < initial_len
        return False
    return True


def _apply_goal_metadata(obs: RPOEObservation, goal: dict) -> dict:
    """Attach execution metadata needed for stable goal execution."""
    g = dict(goal)
    goal_type = g.get("goal", "idle")

    if goal_type == "retrieve" and obs.retrieval_queue:
        g["target_wheel_index"] = obs.retrieval_queue[0].wheel_index
        g["target_slot"] = obs.retrieval_queue[0].slot_index
        g["target_car_id"] = obs.retrieval_queue[0].car_id

    if goal_type == "park":
        if not isinstance(g.get("target_wheel_index"), int) or not isinstance(g.get("target_slot"), int):
            target_wheel, target_slot, _ = _best_parking_candidate(obs)
            if target_wheel is not None:
                g["target_wheel_index"] = target_wheel
                g["target_slot"] = target_slot

    if goal_type == "idle":
        g["target_wheel_index"] = None
        g["target_slot"] = None
        g["target_car_id"] = None

    target_slot = g.get("target_slot")
    target_wheel = g.get("target_wheel_index")
    wheel_count = len(obs.wheels)
    if goal_type in {"retrieve", "park"} and isinstance(target_slot, int):
        target_slot = target_slot % NUM_SLOTS
        g["target_slot"] = target_slot
        if isinstance(target_wheel, int):
            if target_wheel < 0 or target_wheel >= wheel_count:
                target_wheel = None
            g["target_wheel_index"] = target_wheel
        g["direction"] = None
    else:
        g["direction"] = None

    g["initial_retrieval_len"] = len(obs.retrieval_queue)
    g["initial_arrival_len"] = len(obs.arrival_queue)
    return g


def _normalize_goal(parsed: Any) -> dict:
    """Normalize model JSON into canonical goal dict."""
    if not isinstance(parsed, dict):
        return {"goal": "idle", "target_wheel_index": None, "target_slot": None, "target_car_id": None}

    raw_goal = parsed.get("goal")
    if not isinstance(raw_goal, str):
        goal = "idle"
    else:
        g = raw_goal.strip().lower()
        alias_map = {
            "retrieve": "retrieve",
            "retrieval": "retrieve",
            "park": "park",
            "parking": "park",
            "idle": "idle",
            "wait": "idle",
            "none": "idle",
            "do_nothing": "idle",
        }
        goal = alias_map.get(g, "idle")

    raw_slot = parsed.get("target_slot")
    if isinstance(raw_slot, int):
        target_slot = raw_slot % NUM_SLOTS
    elif isinstance(raw_slot, str) and raw_slot.strip().lstrip("-").isdigit():
        target_slot = int(raw_slot.strip()) % NUM_SLOTS
    else:
        target_slot = None

    raw_wheel = parsed.get("target_wheel")
    if not isinstance(raw_wheel, int):
        raw_wheel = parsed.get("wheel_index")
    if isinstance(raw_wheel, int):
        target_wheel = raw_wheel
    elif isinstance(raw_wheel, str) and raw_wheel.strip().lstrip("-").isdigit():
        target_wheel = int(raw_wheel.strip())
    else:
        target_wheel = None

    # Clamp to valid range; None stays None (resolved later by _apply_goal_metadata)
    if isinstance(target_wheel, int) and target_wheel < 0:
        target_wheel = None

    return {"goal": goal, "target_wheel_index": target_wheel, "target_slot": target_slot, "target_car_id": None}


def _goal_completed_transition(
    prev_obs: RPOEObservation,
    obs: RPOEObservation,
    prev_action: RPOEAction,
    goal: dict,
) -> bool:
    """Completion check based on previous action and state transition."""
    g = goal.get("goal", "idle")

    if g == "retrieve" and prev_action.action == ActionType.RETRIEVE:
        return len(obs.retrieval_queue) < len(prev_obs.retrieval_queue)

    if g == "park" and prev_action.action == ActionType.PARK:
        return len(obs.arrival_queue) < len(prev_obs.arrival_queue)

    if g == "idle":
        return True

    return False


# ---------------------------------------------------------------------------
# LLM goal fetcher
# ---------------------------------------------------------------------------

def _obs_to_goal_prompt(obs: RPOEObservation) -> str:
    """Compress observation into a concise goal-planning prompt."""
    ret_q = [
        {"car_id": r.car_id, "wheel_index": r.wheel_index, "slot": r.slot_index}
        for r in obs.retrieval_queue[:3]
    ]
    empty_slots = [s for s in obs.slots if not s.occupied]
    data = {
        "step":              obs.step,
        "hour":              obs.hour,
        "wheel_fronts": [
            {
                "wheel_index": wheel.wheel_index,
                "front_occupied": wheel.front_slot_occupied,
                "front_car_id": wheel.front_car_id,
            }
            for wheel in obs.wheels
        ],
        "arrival_queue_len": len(obs.arrival_queue),
        "retrieval_queue":   ret_q,
        "empty_slots":       len(empty_slots),
        "empty_slot_indices": [
            {"wheel_index": slot.wheel_index, "slot": slot.index}
            for slot in empty_slots[:5]
        ],
    }
    return json.dumps(data, separators=(",", ":"))


def llm_agent(obs: RPOEObservation, retries: int = LLM_RETRIES) -> dict:
    """Call LLM for a high-level goal. Falls back to heuristic goal on failure."""
    prompt = _obs_to_goal_prompt(obs)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": GOAL_SYSTEM_PROMPT + "\n\nSTATE:\n" + prompt},
                    ],
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            raw = (resp.choices[0].message.content or "").strip()
            
            if not raw:
                raise ValueError("Empty response from model")
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            # Recover JSON object if extra text is present.
            raw = raw.strip()
            if not raw.startswith("{"):
                s = raw.find("{")
                e = raw.rfind("}")
                if s != -1 and e != -1 and e > s:
                    raw = raw[s:e + 1]

            parsed = json.loads(raw)
            goal = _normalize_goal(parsed)
            return _apply_goal_metadata(obs, goal)

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(LLM_BACKOFF_SECONDS * (attempt + 1))
            else:
                print(f"  [WARN] LLM goal failed ({e}), using heuristic goal.")
                return _heuristic_goal(obs)

    return _heuristic_goal(obs)


def _heuristic_goal(obs: RPOEObservation) -> dict:
    """Derive goal deterministically — used as fallback and when HF_TOKEN absent."""
    # Match the prompt policy: retrieve-now, then immediate park, then rotate goals.
    if (
        obs.retrieval_queue
        and _get_wheel(obs, obs.retrieval_queue[0].wheel_index).front_car_id == obs.retrieval_queue[0].car_id
    ):
        return _apply_goal_metadata(
            obs,
            {
                "goal": "retrieve",
                "target_wheel_index": obs.retrieval_queue[0].wheel_index,
                "target_slot": obs.retrieval_queue[0].slot_index,
                "target_car_id": obs.retrieval_queue[0].car_id,
            },
        )

    if obs.arrival_queue:
        front_empty_wheel = _first_front_empty_wheel(obs)
        if front_empty_wheel is not None:
            return _apply_goal_metadata(
                obs,
                {"goal": "park", "target_wheel_index": front_empty_wheel, "target_slot": 0, "target_car_id": None},
            )

    if obs.retrieval_queue:
        return _apply_goal_metadata(
            obs,
            {
                "goal": "retrieve",
                "target_wheel_index": obs.retrieval_queue[0].wheel_index,
                "target_slot": obs.retrieval_queue[0].slot_index,
                "target_car_id": obs.retrieval_queue[0].car_id,
            },
        )

    if obs.arrival_queue and any(not s.occupied for s in obs.slots):
        target_wheel, target_slot, _ = _best_parking_candidate(obs)
        return _apply_goal_metadata(
            obs,
            {
                "goal": "park",
                "target_wheel_index": target_wheel,
                "target_slot": target_slot,
                "target_car_id": None,
            },
        )

    return _apply_goal_metadata(
        obs,
        {"goal": "idle", "target_wheel_index": None, "target_slot": None, "target_car_id": None},
    )


# ---------------------------------------------------------------------------
# Hybrid agent — maintains goal state across steps
# ---------------------------------------------------------------------------

class HybridAgent:
    """
    Calls LLM only when a new goal is needed.
    Executes current goal deterministically step-by-step.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm       = use_llm
        self.current_goal: Optional[dict] = None
        self.goal_cache:   dict           = {}
        self.llm_calls:    int            = 0
        self.prev_obs:     Optional[RPOEObservation] = None
        self.prev_action:  Optional[RPOEAction] = None

    def _state_hash(self, obs: RPOEObservation) -> int:
        return hash(
            str([(slot.wheel_index, slot.index, slot.car_id) for slot in obs.slots])
            + str([(r.car_id, r.wheel_index, r.slot_index) for r in obs.retrieval_queue])
            + str(len(obs.arrival_queue))
        )

    def __call__(self, obs: RPOEObservation) -> RPOEAction:
        if (
            self.current_goal is not None
            and self.prev_obs is not None
            and self.prev_action is not None
            and _goal_completed_transition(self.prev_obs, obs, self.prev_action, self.current_goal)
        ):
            self.current_goal = None

        need_new_goal = (
            self.current_goal is None
            or not _is_goal_valid(obs, self.current_goal)
            or _is_goal_complete(obs, self.current_goal)
        )

        if need_new_goal:
            if self.use_llm:
                state_hash = self._state_hash(obs)
                if state_hash in self.goal_cache:
                    self.current_goal = _apply_goal_metadata(obs, self.goal_cache[state_hash])
                else:
                    self.current_goal = llm_agent(obs)
                    self.goal_cache[state_hash] = {
                        "goal": self.current_goal.get("goal", "idle"),
                        "target_wheel_index": self.current_goal.get("target_wheel_index"),
                        "target_slot": self.current_goal.get("target_slot"),
                        "target_car_id": self.current_goal.get("target_car_id"),
                    }
                    self.llm_calls += 1
            else:
                self.current_goal = _heuristic_goal(obs)

        action = execute_goal(obs, self.current_goal)
        self.prev_obs = obs
        self.prev_action = action
        return action


# ---------------------------------------------------------------------------
# Backward-compatible heuristic agent (for non-LLM runs)
# ---------------------------------------------------------------------------

def _heuristic_agent(obs: RPOEObservation) -> RPOEAction:
    """Stateless heuristic — used directly when use_llm=False."""
    return execute_goal(obs, _heuristic_goal(obs))


# ---------------------------------------------------------------------------
# Run all tasks
# ---------------------------------------------------------------------------

def run_all_tasks(use_llm: bool = True) -> Dict[str, TaskResult]:
    results = {}

    task_configs = [
        ("task1_easy",   ""
        ""),
        ("task2_medium", "Medium — Peak-hour throughput (180 steps)"),
        ("task3_hard",   "Hard   — Full day composite (1080 steps)"),
    ]

    print("\n" + "=" * 60)
    print("  RPOE Hybrid Inference")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  API    : {API_BASE_URL}")
    print(f"  Mode   : {'LLM + heuristic executor' if use_llm else 'heuristic only'}")
    print("=" * 60)

    total_start = time.time()
    completed_scores: list = []

    for task_id, label in task_configs:
        # Fresh agent per task — resets goal state and cache
        use_llm_for_task = use_llm and task_id != "task1_easy"
        agent = HybridAgent(use_llm=use_llm_for_task)

        print(f"\n[{label}]")
        print(f"[START] task_id={task_id} model={MODEL_NAME}")

        def _make_logged(inner):
            def _logged(obs):
                action = inner(obs)
                print(f"[STEP] step={obs.step} action={action.action} reward={obs.reward:.4f} done={obs.done}")
                return action
            return _logged

        t0      = time.time()
        result  = TASKS[task_id](agent_fn=_make_logged(agent), seed=42)
        elapsed = time.time() - t0

        results[task_id] = result
        completed_scores.append(result.score)
        running_avg = sum(completed_scores) / len(completed_scores)
        print(f"[END] task_id={task_id} score={result.score:.4f} avg_score={running_avg:.4f}")

        status = "PASS" if result.passed else "FAIL"
        print(f"  Score    : {result.score:.4f}  [{status}]")
        print(f"  Time     : {elapsed:.1f}s")
        print(f"  LLM calls: {agent.llm_calls}")
        print(f"  Notes    : {result.notes}")
        for k, v in result.metrics.items():
            print(f"    {k:<22}: {v}")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for task_id, result in results.items():
        bar_len = int(result.score * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {task_id:<18} {bar}  {result.score:.4f}")

    avg_score = sum(r.score for r in results.values()) / len(results)
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"  Total runtime: {total_elapsed:.1f}s  (limit: 1200s)")
    print("=" * 60 + "\n")

    output = {
        "scores":    {tid: r.score for tid, r in results.items()},
        "metrics":   {tid: r.metrics for tid, r in results.items()},
        "avg_score": avg_score,
        "model":     MODEL_NAME,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Scores written to baseline_scores.json")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_llm = bool(HF_TOKEN)
    if not use_llm:
        print("[INFO] No HF_TOKEN — running heuristic baseline (no LLM calls).")
    run_all_tasks(use_llm=use_llm)
