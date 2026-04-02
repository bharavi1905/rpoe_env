"""
tests/test_env.py — RPOE smoke tests
Run: python -m pytest tests/ -v
"""

import pytest
from server.env import RotaryParkingEnv
from models import RPOEAction, ActionType, RPOEObservation, RPOEState, Reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return RotaryParkingEnv(seed=42, max_steps=100)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def test_reset_returns_observation(env):
    obs = env.reset()
    assert isinstance(obs, RPOEObservation)
    assert len(obs.slots) == 12
    assert obs.step == 0


def test_state_returns_env_state(env):
    env.reset()
    s = env.state
    assert isinstance(s, RPOEState)
    assert s.seed == 42
    assert s.step_count == 0
    assert not s.episode_done


def test_step_returns_correct_types(env):
    env.reset()
    obs = env.step(RPOEAction(action=ActionType.ROTATE_CW))
    assert isinstance(obs, RPOEObservation)
    assert isinstance(obs.reward, (int, float))
    assert isinstance(obs.done, bool)


# ---------------------------------------------------------------------------
# Action mechanics
# ---------------------------------------------------------------------------

def test_rotate_cw_moves_wheel(env):
    obs0 = env.reset()
    original_slots = [s.car_id for s in obs0.slots]

    env.step(RPOEAction(action=ActionType.ROTATE_CW))
    s = env.state
    new_slots = [sl.car_id for sl in s.slots]

    # After CW rotation: slot[0] gets old slot[11], slot[1] gets old slot[0]
    assert new_slots[1] == original_slots[0]


def test_rotate_ccw_is_inverse_of_cw(env):
    env.reset()
    # CW then CCW should return wheel to original
    obs0 = env.state
    orig = [s.car_id for s in obs0.slots]

    env.step(RPOEAction(action=ActionType.ROTATE_CW))
    env.step(RPOEAction(action=ActionType.ROTATE_CCW))

    s = env.state
    restored = [sl.car_id for sl in s.slots]
    assert orig == restored


def test_illegal_park_penalised(env):
    """Park with front slot occupied must return illegal penalty."""
    env.reset()
    # Force a car into slot 0 by direct manipulation
    env._slots[0] = "TEST-CAR"
    env._arrival_q.clear()
    # Arrival queue is empty AND slot occupied — park is illegal
    env.step(RPOEAction(action=ActionType.ROTATE_CW))  # just rotate
    # Now try park when queue is empty
    obs2 = env.step(RPOEAction(action=ActionType.PARK))
    assert obs2.reward_breakdown.illegal_penalty < 0


def test_episode_terminates_at_max_steps():
    env = RotaryParkingEnv(seed=0, max_steps=5)
    env.reset()
    obs = None
    for _ in range(5):
        obs = env.step(RPOEAction(action=ActionType.ROTATE_CW))
    assert obs.done is True


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

def test_reward_has_breakdown(env):
    env.reset()
    obs = env.step(RPOEAction(action=ActionType.ROTATE_CW))
    b = obs.reward_breakdown
    assert hasattr(b, "waiting_penalty")
    assert hasattr(b, "rotation_penalty")
    assert hasattr(b, "park_bonus")
    assert hasattr(b, "retrieval_bonus")
    assert hasattr(b, "illegal_penalty")
    assert hasattr(b, "overflow_penalty")


def test_reward_total_matches_breakdown(env):
    env.reset()
    obs = env.step(RPOEAction(action=ActionType.ROTATE_CW))
    b = obs.reward_breakdown
    expected = (
        b.waiting_penalty
        + b.rotation_penalty
        + b.park_bonus
        + b.retrieval_bonus
        + b.illegal_penalty
        + b.overflow_penalty
    )
    assert abs(obs.reward - expected) < 1e-3


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def test_task1_returns_valid_score():
    from tasks.graders import run_task1
    import random

    def rnd_agent(obs):
        return RPOEAction(action=random.choice(list(ActionType)))

    result = run_task1(rnd_agent, seed=42)
    assert 0.0 <= result.score <= 1.0
    assert result.task_id == "task1_easy"


def test_task2_returns_valid_score():
    from tasks.graders import run_task2
    import random

    def rnd_agent(obs):
        return RPOEAction(action=random.choice(list(ActionType)))

    result = run_task2(rnd_agent, seed=42)
    assert 0.0 <= result.score <= 1.0


def test_task3_returns_valid_score():
    from tasks.graders import run_task3
    import random

    def rnd_agent(obs):
        return RPOEAction(action=random.choice(list(ActionType)))

    result = run_task3(rnd_agent, seed=42)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_trajectory():
    """Two envs with same seed must produce identical reward sequences."""
    rewards_a, rewards_b = [], []

    for rewards in [rewards_a, rewards_b]:
        e = RotaryParkingEnv(seed=7, max_steps=20)
        e.reset()
        import random as rnd
        rnd.seed(99)
        for _ in range(20):
            act = RPOEAction(action=rnd.choice(list(ActionType)))
            obs = e.step(act)
            rewards.append(round(obs.reward, 4))
            if obs.done:
                break

    assert rewards_a == rewards_b
