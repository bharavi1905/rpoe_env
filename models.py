"""
Pydantic typed models for the Rotary Parking Optimization Environment.
All OpenEnv spec types live here.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    ROTATE_CW  = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    PARK       = "park"
    RETRIEVE   = "retrieve"
    IDLE       = "idle"


class RPOEAction(Action):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    action: ActionType = Field(..., description="One of: rotate_cw, rotate_ccw, park, retrieve, idle")


# ---------------------------------------------------------------------------
# Observation sub-types
# ---------------------------------------------------------------------------

class SlotState(BaseModel):
    index: int = Field(..., description="Slot index 0–11; 0 is the front/accessible position")
    occupied: bool = Field(..., description="True if a car occupies this slot")
    car_id: Optional[str] = Field(default=None, description="Car identifier, or None if slot is empty")


class QueuedCar(BaseModel):
    car_id: str = Field(..., description="Unique car identifier")
    arrival_step: int = Field(..., description="Simulation step at which the car joined the arrival queue")


class PendingRetrieval(BaseModel):
    car_id: str = Field(..., description="Unique car identifier requesting retrieval")
    slot_index: int = Field(..., description="Wheel slot where the car currently sits")
    requested_at_step: int = Field(..., description="Simulation step at which retrieval was requested")


# ---------------------------------------------------------------------------
# Reward (internal — not an OpenEnv type)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    waiting_penalty: float = Field(..., description="-1.0 × number of cars waiting in arrival + retrieval queues")
    rotation_penalty: float = Field(..., description="-0.5 if a rotation action was taken, else 0")
    park_bonus: float = Field(..., description="+2.0 on a successful park action")
    retrieval_bonus: float = Field(..., description="+3.0 on a successful retrieval action")
    illegal_penalty: float = Field(..., description="-2.0 when an invalid action is attempted")
    overflow_penalty: float = Field(..., description="-5.0 per car that times out and overflows")


class Reward(BaseModel):
    total: float = Field(..., description="Sum of all reward components for this step")
    breakdown: RewardBreakdown = Field(..., description="Per-component reward detail")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class RPOEObservation(Observation):
    # Wheel state
    slots: List[SlotState] = Field(..., description="All 12 wheel slots; index 0 is the front/accessible slot")
    front_slot_occupied: bool = Field(..., description="True if the front slot currently holds a car")
    front_car_id: Optional[str] = Field(..., description="Car ID at the accessible slot, or None if empty")

    # Queues
    arrival_queue: List[QueuedCar] = Field(..., description="Cars waiting to be parked")
    retrieval_queue: List[PendingRetrieval] = Field(..., description="Cars that have requested exit")

    # Time
    step: int = Field(..., description="Current simulation step number")
    hour: float = Field(..., description="Simulated hour-of-day (0.0–18.0)")

    # Episode metadata
    total_parked: int = Field(..., description="Cumulative cars successfully parked this episode")
    total_retrieved: int = Field(..., description="Cumulative cars successfully retrieved this episode")
    total_overflowed: int = Field(..., description="Cumulative cars that timed out and overflowed this episode")
    last_action_valid: bool = Field(..., description="Whether the previous action was legal")
    last_action: Optional[str] = Field(default=None, description="String name of the last action taken")

    # Reward detail
    reward_breakdown: Optional[RewardBreakdown] = Field(default=None, description="Per-component reward for the last step")


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class RPOEState(State):
    # State provides: episode_id, step_count
    slots: List[SlotState] = Field(..., description="Current occupancy of all 12 wheel slots")
    front_slot_index: int = Field(..., description="Index of the front slot (always 0 after normalisation)")
    arrival_queue: List[QueuedCar] = Field(..., description="Cars waiting to be parked")
    retrieval_queue: List[PendingRetrieval] = Field(..., description="Cars that have requested exit")
    hour: float = Field(..., description="Simulated hour-of-day (0.0–18.0)")
    episode_done: bool = Field(..., description="True when the episode has ended")
    seed: int = Field(..., description="RNG seed used for this episode")
    config: Dict[str, Any] = Field(..., description="Environment configuration parameters")


# ---------------------------------------------------------------------------
# Task / grader types (internal — not OpenEnv types)
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    task_id: str = Field(..., description="Identifier for the task (e.g. 'task1')")
    score: float = Field(..., description="Normalised score in [0.0, 1.0]")
    metrics: Dict[str, float] = Field(..., description="Named sub-metrics contributing to the score")
    passed: bool = Field(..., description="True if the score meets the passing threshold")
    notes: str = Field(..., description="Human-readable summary of the result")
