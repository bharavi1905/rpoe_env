"""
Rotary Parking Optimization Environment (RPOE)
==============================================
Simulates the KBR Park vertical rotary parking system (Hyderabad).
Fully OpenEnv-compliant: step() / reset() / state with Pydantic models.
"""

from __future__ import annotations

import random
import math
import uuid
from typing import Optional, Dict, Any, List

import numpy as np

from openenv.core.env_server.interfaces import Environment


from models import (
    RPOEAction, ActionType, RPOEObservation, RPOEState,
    Reward, RewardBreakdown,
    SlotState, WheelState, QueuedCar, PendingRetrieval,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHEEL_SIZE       = 12          # slots per wheel (one rotary stack)
WHEEL_COUNT      = 7           # number of independent rotary stacks
MAX_ARRIVAL_Q    = 10          # max cars waiting outside
MAX_RETRIEVAL_Q  = 10          # max pending retrievals
MAX_STEPS        = 1080        # 18-hour day in 1-min steps (5 AM – 11 PM)
OVERFLOW_TIMEOUT = 15          # steps before queued car overflows (rage-leaves)
TRAFFIC_MULTIPLIER = 1.0       # scales the baseline arrival rates

# Poisson arrival rates (cars/step) by hour-of-day band
ARRIVAL_RATES = [
    # (start_hour, end_hour, lambda)
    (0.0,  1.0,  0.05),   # 5–6 AM  — very quiet
    (1.0,  4.0,  0.35),   # 6–9 AM  — morning peak (joggers/walkers)
    (4.0,  12.0, 0.10),   # 9 AM–5 PM — midday trickle
    (12.0, 15.0, 0.25),   # 5–8 PM  — evening peak
    (15.0, 18.0, 0.05),   # 8–11 PM — wind-down
]

# Mean dwell time (steps) and std for LogNormal distribution
DWELL_MU_STEPS  = 60
DWELL_STD_STEPS = 30


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _arrival_rate(hour: float) -> float:
    for start, end, lam in ARRIVAL_RATES:
        if start <= hour < end:
            return lam
    return 0.05


def _sample_dwell() -> int:
    """LogNormal dwell time — how long a car stays before requesting retrieval."""
    sigma = math.sqrt(math.log(1 + (DWELL_STD_STEPS / DWELL_MU_STEPS) ** 2))
    mu    = math.log(DWELL_MU_STEPS) - 0.5 * sigma ** 2
    return max(5, int(random.lognormvariate(mu, sigma)))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RotaryParkingEnv(Environment[RPOEAction, RPOEObservation, RPOEState]):
    """
    OpenEnv-compliant environment modelling one or more 12-slot vertical
    rotary parking wheels. Only the front-facing slot (index 0) of each
    wheel is accessible.

    The agent must rotate a target wheel (CW / CCW) to align the correct
    slot before parking or retrieving. Stochastic arrivals follow a Poisson
    process with time-of-day varying rate.

    Args:
        seed: RNG seed for reproducibility.
        max_steps: Episode length in simulation steps (default 1080 = 18 hrs).
        wheel_count: Number of independent rotary wheels (default 7).
        wheel_size: Number of slots per wheel (default 12).
        traffic_multiplier: Multiplier applied to baseline arrival rates.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = MAX_STEPS,
        wheel_count: int = WHEEL_COUNT,
        wheel_size: int = WHEEL_SIZE,
        traffic_multiplier: float = TRAFFIC_MULTIPLIER,
    ):
        super().__init__()

        self.seed_val   = seed
        self.max_steps  = max_steps
        self.wheel_count = wheel_count
        self.wheel_size = wheel_size
        self.traffic_multiplier = traffic_multiplier

        if self.wheel_count < 1:
            raise ValueError("wheel_count must be at least 1")
        if self.wheel_size < 1:
            raise ValueError("wheel_size must be at least 1")
        if self.traffic_multiplier <= 0:
            raise ValueError("traffic_multiplier must be positive")

        # Will be fully initialised in reset()
        self._slots: List[List[Optional[str]]]    = []   # [wheel][slot] -> car_id or None
        self._arrival_q: List[QueuedCar]          = []
        self._retrieval_q: List[PendingRetrieval] = []
        self._dwell_timers: Dict[str, int]        = {}   # car_id → step to request exit
        self._step: int                           = 0
        self._episode_id: str                     = str(uuid.uuid4())

        # Episode stats
        self._total_parked:     int = 0
        self._total_retrieved:  int = 0
        self._total_overflowed: int = 0
        self._last_valid:       bool = True
        self._last_action:      Optional[str] = None

        self.reset()

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RPOEObservation:
        """Reset to clean initial state. Returns initial observation."""
        if seed is not None:
            self.seed_val = seed
        self._episode_id = episode_id or str(uuid.uuid4())

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        self._slots          = [
            [None] * self.wheel_size
            for _ in range(self.wheel_count)
        ]
        self._arrival_q      = []
        self._retrieval_q    = []
        self._dwell_timers   = {}
        self._step           = 0
        self._total_parked   = 0
        self._total_retrieved= 0
        self._total_overflowed = 0
        self._last_valid     = True
        self._last_action    = None

        return self._make_obs()

    def step(
        self,
        action: RPOEAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RPOEObservation:
        """
        Advance the simulation by one step.

        Returns:
            RPOEObservation with done and reward set.
        """
        self._step += 1
        act = ActionType(action.action)
        wheel_index = action.wheel_index
        self._last_action = act.value

        valid_wheel = isinstance(wheel_index, int) and 0 <= wheel_index < self.wheel_count

        # ── 1. Stochastic arrivals ──────────────────────────────────────────
        hour   = self._current_hour()
        lam    = _arrival_rate(hour) * self.traffic_multiplier
        n_new  = np.random.poisson(lam)
        for _ in range(n_new):
            if len(self._arrival_q) < MAX_ARRIVAL_Q:
                car_id = f"CAR-{uuid.uuid4().hex[:6].upper()}"
                self._arrival_q.append(
                    QueuedCar(car_id=car_id, arrival_step=self._step)
                )

        # ── 2. Dwell timer expiry → retrieval requests ────────────────────
        for car_id, exit_step in list(self._dwell_timers.items()):
            if self._step >= exit_step:
                car_location = self._find_car(car_id)
                if car_location is not None and len(self._retrieval_q) < MAX_RETRIEVAL_Q:
                    wheel_idx, slot_idx = car_location
                    self._retrieval_q.append(
                        PendingRetrieval(
                            car_id=car_id,
                            wheel_index=wheel_idx,
                            slot_index=slot_idx,
                            requested_at_step=self._step,
                        )
                    )
                del self._dwell_timers[car_id]

        # ── 3. Overflow — cars that waited too long ───────────────────────
        overflow_penalty = 0.0
        fresh_arrival_q  = []
        for qcar in self._arrival_q:
            if self._step - qcar.arrival_step > OVERFLOW_TIMEOUT:
                self._total_overflowed += 1
                overflow_penalty += 5.0
            else:
                fresh_arrival_q.append(qcar)
        self._arrival_q = fresh_arrival_q

        # ── 4. Execute action ─────────────────────────────────────────────
        rotation_penalty = 0.0
        park_bonus       = 0.0
        retrieval_bonus  = 0.0
        illegal_penalty  = 0.0
        valid            = act == ActionType.IDLE or valid_wheel

        if act != ActionType.IDLE and not valid_wheel:
            illegal_penalty = 2.0

        elif act == ActionType.ROTATE_CW:
            self._slots[wheel_index] = [self._slots[wheel_index][-1]] + self._slots[wheel_index][:-1]
            rotation_penalty = 0.5

        elif act == ActionType.ROTATE_CCW:
            self._slots[wheel_index] = self._slots[wheel_index][1:] + [self._slots[wheel_index][0]]
            rotation_penalty = 0.5

        elif act == ActionType.PARK:
            if self._arrival_q and self._slots[wheel_index][0] is None:
                qcar = self._arrival_q.pop(0)
                self._slots[wheel_index][0] = qcar.car_id
                dwell = _sample_dwell()
                self._dwell_timers[qcar.car_id] = self._step + dwell
                self._total_parked += 1
                park_bonus = 2.0
            else:
                valid = False
                illegal_penalty = 2.0

        elif act == ActionType.RETRIEVE:
            front_car = self._slots[wheel_index][0]
            if (front_car is not None
                    and self._retrieval_q
                    and self._retrieval_q[0].car_id == front_car
                    and self._retrieval_q[0].wheel_index == wheel_index):
                self._retrieval_q.pop(0)
                self._slots[wheel_index][0] = None
                self._total_retrieved += 1
                retrieval_bonus = 3.0
            else:
                valid = False
                illegal_penalty = 2.0

        elif act == ActionType.IDLE:
            # No-op: wait without action (no rotation cost, no reward, no penalty)
            pass

        self._last_valid = valid

        # ── 5. Waiting penalty (per-step pressure) ────────────────────────
        waiting = len(self._arrival_q) + len(self._retrieval_q)
        waiting_penalty = -1.0 * waiting

        # ── 6. Compose reward ─────────────────────────────────────────────
        breakdown = RewardBreakdown(
            waiting_penalty  = waiting_penalty,
            rotation_penalty = -rotation_penalty,
            park_bonus       = park_bonus,
            retrieval_bonus  = retrieval_bonus,
            illegal_penalty  = -illegal_penalty,
            overflow_penalty = -overflow_penalty,
        )
        total_reward = (
            waiting_penalty
            - rotation_penalty
            + park_bonus
            + retrieval_bonus
            - illegal_penalty
            - overflow_penalty
        )
        reward = Reward(total=round(total_reward, 4), breakdown=breakdown)

        # ── 7. Done condition ─────────────────────────────────────────────
        done = self._step >= self.max_steps

        return self._make_obs(
            done=done,
            reward_total=reward.total,
            reward_breakdown=breakdown,
        )

    @property
    def state(self) -> RPOEState:
        """Return full serialisable environment state (for checkpointing)."""
        return RPOEState(
            episode_id=self._episode_id,
            step_count=self._step,
            wheels=self._make_wheels_state(),
            slots=self._make_flat_slots(),
            arrival_queue=list(self._arrival_q),
            retrieval_queue=list(self._retrieval_q),
            hour=self._current_hour(),
            episode_done=self._step >= self.max_steps,
            seed=self.seed_val,
            config={
                "wheel_count": self.wheel_count,
                "wheel_size": self.wheel_size,
                "max_steps":  self.max_steps,
                "overflow_timeout": OVERFLOW_TIMEOUT,
                "traffic_multiplier": self.traffic_multiplier,
            },
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _current_hour(self) -> float:
        """Map simulation step → hour of day (0 = 5 AM open, 18 = 11 PM close)."""
        return (self._step / self.max_steps) * 18.0

    def _find_car(self, car_id: str) -> Optional[tuple[int, int]]:
        """Return wheel and slot index of a car, or None if not found."""
        for wheel_index, wheel_slots in enumerate(self._slots):
            for slot_index, cid in enumerate(wheel_slots):
                if cid == car_id:
                    return wheel_index, slot_index
        return None

    def _make_flat_slots(self) -> List[SlotState]:
        return [
            SlotState(
                wheel_index=wheel_index,
                index=slot_index,
                occupied=wheel_slots[slot_index] is not None,
                car_id=wheel_slots[slot_index],
            )
            for wheel_index, wheel_slots in enumerate(self._slots)
            for slot_index in range(self.wheel_size)
        ]

    def _make_wheels_state(self) -> List[WheelState]:
        wheels = []
        for wheel_index, wheel_slots in enumerate(self._slots):
            slot_states = [
                SlotState(
                    wheel_index=wheel_index,
                    index=slot_index,
                    occupied=wheel_slots[slot_index] is not None,
                    car_id=wheel_slots[slot_index],
                )
                for slot_index in range(self.wheel_size)
            ]
            wheels.append(
                WheelState(
                    wheel_index=wheel_index,
                    front_slot_index=0,
                    front_slot_occupied=wheel_slots[0] is not None,
                    front_car_id=wheel_slots[0],
                    slots=slot_states,
                )
            )
        return wheels

    def _make_obs(
        self,
        done: bool = False,
        reward_total: float = 0.0,
        reward_breakdown: Optional[RewardBreakdown] = None,
    ) -> RPOEObservation:
        return RPOEObservation(
            wheel_count=self.wheel_count,
            wheels=self._make_wheels_state(),
            slots=self._make_flat_slots(),
            arrival_queue=list(self._arrival_q),
            retrieval_queue=list(self._retrieval_q),
            step=self._step,
            hour=round(self._current_hour(), 2),
            total_parked=self._total_parked,
            total_retrieved=self._total_retrieved,
            total_overflowed=self._total_overflowed,
            last_action_valid=self._last_valid,
            last_action=self._last_action,
            done=done,
            reward=reward_total,
            reward_breakdown=reward_breakdown,
        )

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self, mode: str = "human") -> str:
        hour_label = f"{int(5 + self._current_hour()):02d}:00"
        wheel_vis  = ""
        for wheel_index, wheel_slots in enumerate(self._slots):
            wheel_vis += f"  Wheel {wheel_index}\n"
            for slot_index, cid in enumerate(wheel_slots):
                marker = "►" if slot_index == 0 else " "
                slot   = f"[{cid[:8] if cid else '  EMPTY  '}]"
                wheel_vis += f"    {marker} {slot_index:2d} {slot}\n"

        out = (
            f"\n{'='*50}\n"
            f"  RPOE  Step {self._step:>4d}  |  {hour_label}\n"
            f"{'='*50}\n"
            f"{wheel_vis}"
            f"  Arrival queue : {len(self._arrival_q):>2d} cars\n"
            f"  Retrieval queue: {len(self._retrieval_q):>2d} cars\n"
            f"  Parked: {self._total_parked}  "
            f"Retrieved: {self._total_retrieved}  "
            f"Overflowed: {self._total_overflowed}\n"
        )
        if mode == "human":
            print(out)
        return out
