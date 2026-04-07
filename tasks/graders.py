"""
RPOE Task definitions and graders.
Each task returns a TaskResult with score 0.0–1.0.

Task 1 (Easy)   — Single-stack slot assignment, 50 steps, no queue pressure
Task 2 (Medium) — Morning peak hour, 180 steps, stochastic arrivals
Task 3 (Hard)   — Full 18-hour day, composite scoring with overflow penalty
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Any

from server.env import RotaryParkingEnv
from models import RPOEAction, ActionType, TaskResult


def _open_score(score: float) -> float:
    """Clamp to open interval (0, 1) — validator rejects exact 0.0 or 1.0."""
    return round(max(0.001, min(0.999, score)), 4)


# ---------------------------------------------------------------------------
# Task 1 — Easy: Optimal rotation planning
# ---------------------------------------------------------------------------

def run_task1(agent_fn: Callable, seed: int = 0) -> TaskResult:
    """
    Easy task: 50-step episode, low arrival rate (λ=0.1), no overflow risk.
    Agent is scored on rotation efficiency — how close to optimal path length
    it gets when servicing park and retrieve ops.

    Score = 1 - clamp(excess_rotations / budget, 0, 1)
    """
    env = RotaryParkingEnv(seed=seed, max_steps=50, wheel_size=12)
    # Override arrival rate to be gentle
    import server.env as emod
    _orig = emod._arrival_rate
    emod._arrival_rate = lambda h: 0.10

    obs = env.reset()
    total_reward       = 0.0
    rotation_count     = 0
    successful_ops     = 0
    illegal_count      = 0

    for _ in range(50):
        action = agent_fn(obs)
        obs = env.step(action)
        total_reward += obs.reward

        if action.action in (ActionType.ROTATE_CW, ActionType.ROTATE_CCW):
            rotation_count += 1
        if obs.reward_breakdown and (
            obs.reward_breakdown.park_bonus > 0
            or obs.reward_breakdown.retrieval_bonus > 0
        ):
            successful_ops += 1
        if obs.reward_breakdown and obs.reward_breakdown.illegal_penalty < 0:
            illegal_count += 1
        if obs.done:
            break

    emod._arrival_rate = _orig

    # Scoring: penalise excess rotations and illegal moves
    expected_rotations = max(1, successful_ops) * 3
    excess = max(0, rotation_count - expected_rotations)
    rotation_score = max(0.0, 1.0 - (excess / max(expected_rotations, 1)))
    illegal_score  = max(0.0, 1.0 - (illegal_count / 10))
    score = _open_score(0.6 * rotation_score + 0.4 * illegal_score)

    return TaskResult(
        task_id="task1_easy",
        score=score,
        metrics={
            "total_reward":     round(total_reward, 2),
            "rotation_count":   rotation_count,
            "successful_ops":   successful_ops,
            "illegal_actions":  illegal_count,
            "rotation_score":   round(rotation_score, 4),
            "illegal_score":    round(illegal_score, 4),
        },
        passed=score >= 0.4,
        notes=(
            f"Agent completed {successful_ops} ops in 50 steps with "
            f"{rotation_count} rotations and {illegal_count} illegal actions."
        ),
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium: Peak-hour throughput
# ---------------------------------------------------------------------------

def run_task2(agent_fn: Callable, seed: int = 0) -> TaskResult:
    """
    Medium task: 180-step morning peak simulation (6–9 AM, λ=0.35).
    Agent is scored on cars served vs cars that overflowed.

    Score = served / (served + overflowed), penalised for excess rotations.
    """
    env = RotaryParkingEnv(seed=seed, max_steps=180, wheel_size=12)
    # Force peak-hour arrivals
    import server.env as emod
    _orig = emod._arrival_rate
    emod._arrival_rate = lambda h: 0.35

    obs = env.reset()
    total_reward = 0.0
    rotation_count = 0

    for _ in range(180):
        action = agent_fn(obs)
        obs = env.step(action)
        total_reward += obs.reward
        if action.action in (ActionType.ROTATE_CW, ActionType.ROTATE_CCW):
            rotation_count += 1
        if obs.done:
            break

    emod._arrival_rate = _orig

    served     = obs.total_parked + obs.total_retrieved
    overflowed = obs.total_overflowed
    total_ops  = served + overflowed

    throughput_score = served / max(total_ops, 1)

    rot_budget  = max(1, served) * 4
    rot_penalty = max(0.0, (rotation_count - rot_budget) / max(rot_budget, 1))
    rot_penalty = min(rot_penalty, 0.3)

    score = _open_score(max(0.0, throughput_score - rot_penalty))

    return TaskResult(
        task_id="task2_medium",
        score=score,
        metrics={
            "total_reward":     round(total_reward, 2),
            "cars_parked":      obs.total_parked,
            "cars_retrieved":   obs.total_retrieved,
            "cars_overflowed":  overflowed,
            "throughput_score": round(throughput_score, 4),
            "rotation_count":   rotation_count,
        },
        passed=score >= 0.5,
        notes=(
            f"Served {served} cars, {overflowed} overflowed "
            f"({rotation_count} rotations in peak hour)."
        ),
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full-day composite score
# ---------------------------------------------------------------------------

def run_task3(agent_fn: Callable, seed: int = 0) -> TaskResult:
    """
    Hard task: Full 1080-step day (5 AM–11 PM), all demand bands active.
    Composite score across four dimensions:

      1. Throughput   (40%) — cars served / (served + overflowed)
      2. Efficiency   (25%) — 1 - rotation waste ratio
      3. Retrieval    (20%) — retrieved / parked (penalises stranded cars)
      4. Stability    (15%) — 1 - (avg queue length / 10)

    Each dimension clamped to [0, 1].
    """
    env = RotaryParkingEnv(seed=seed, max_steps=1080, wheel_size=12)
    obs = env.reset()

    total_reward    = 0.0
    rotation_count  = 0
    queue_lengths   = []

    for _ in range(1080):
        action = agent_fn(obs)
        obs = env.step(action)
        total_reward += obs.reward

        if action.action in (ActionType.ROTATE_CW, ActionType.ROTATE_CCW):
            rotation_count += 1

        queue_lengths.append(
            len(obs.arrival_queue) + len(obs.retrieval_queue)
        )
        if obs.done:
            break

    parked     = obs.total_parked
    retrieved  = obs.total_retrieved
    overflowed = obs.total_overflowed

    # 1. Throughput score
    served = parked + retrieved
    throughput = served / max(served + overflowed, 1)

    # 2. Efficiency score (rotation waste)
    rot_budget  = max(1, served) * 4
    rot_ratio   = min(rotation_count / max(rot_budget, 1), 2.0)
    efficiency  = max(0.0, 1.0 - (rot_ratio - 1.0)) if rot_ratio > 1 else 1.0

    # 3. Retrieval ratio
    retrieval_ratio = retrieved / max(parked, 1)
    retrieval_score = min(retrieval_ratio, 1.0)

    # 4. Queue stability
    avg_queue = sum(queue_lengths) / max(len(queue_lengths), 1)
    stability = max(0.0, 1.0 - avg_queue / 10.0)

    score = _open_score(
        0.40 * throughput
        + 0.25 * efficiency
        + 0.20 * retrieval_score
        + 0.15 * stability
    )

    return TaskResult(
        task_id="task3_hard",
        score=score,
        metrics={
            "total_reward":     round(total_reward, 2),
            "cars_parked":      parked,
            "cars_retrieved":   retrieved,
            "cars_overflowed":  overflowed,
            "rotation_count":   rotation_count,
            "throughput_score": round(throughput, 4),
            "efficiency_score": round(efficiency, 4),
            "retrieval_score":  round(retrieval_score, 4),
            "stability_score":  round(stability, 4),
            "avg_queue_len":    round(avg_queue, 2),
        },
        passed=score >= 0.55,
        notes=(
            f"Full-day: {parked} parked, {retrieved} retrieved, "
            f"{overflowed} overflowed. Score breakdown: "
            f"T={throughput:.2f} E={efficiency:.2f} "
            f"R={retrieval_score:.2f} S={stability:.2f}"
        ),
    )


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Callable] = {
    "task1_easy":   run_task1,
    "task2_medium": run_task2,
    "task3_hard":   run_task3,
}


# ---------------------------------------------------------------------------
# Manual test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    wheel_count = RotaryParkingEnv().wheel_count

    def random_agent(_):
        action = random.choice(list(ActionType))
        wheel_index = None if action == ActionType.IDLE else random.randrange(wheel_count)
        return RPOEAction(action=action, wheel_index=wheel_index)

    for task_id, fn in TASKS.items():
        result = fn(random_agent, seed=42)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {task_id}: score={result.score:.4f} — {result.notes}")
