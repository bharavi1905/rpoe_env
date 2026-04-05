"""
Pydantic typed models for the Rotary Parking Optimization Environment.
All OpenEnv spec types live here.
"""

from __future__ import annotations
from enum import Enum
from typing import ClassVar, List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    _wheel_count: ClassVar[int] = 7

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    action: ActionType = Field(..., description="One of: rotate_cw, rotate_ccw, park, retrieve, idle")
    wheel_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Target wheel index for the action; null is allowed only for idle.",
    )

    @classmethod
    def configure_wheel_count(cls, wheel_count: int) -> None:
        if wheel_count < 1:
            raise ValueError("wheel_count must be at least 1")
        cls._wheel_count = wheel_count

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        wheel_index = schema.get("properties", {}).get("wheel_index")
        if wheel_index is not None:
            wheel_index["minimum"] = 0
            wheel_index["maximum"] = cls._wheel_count - 1
            wheel_index["description"] = (
                "Target wheel index for the action. "
                f"Valid values are 0-{cls._wheel_count - 1}; null is allowed only for idle."
            )
        return schema

    @model_validator(mode="after")
    def validate_wheel_index(self) -> "RPOEAction":
        if self.action == ActionType.IDLE:
            return self
        if self.wheel_index is None:
            raise ValueError("wheel_index is required for non-idle actions")
        if self.wheel_index >= self._wheel_count:
            raise ValueError(f"wheel_index must be between 0 and {self._wheel_count - 1}")
        return self


# ---------------------------------------------------------------------------
# Observation sub-types
# ---------------------------------------------------------------------------

class SlotState(BaseModel):
    wheel_index: int = Field(..., description="Wheel containing this slot")
    index: int = Field(..., description="Slot index within the wheel; 0 is the front/accessible position")
    occupied: bool = Field(..., description="True if a car occupies this slot")
    car_id: Optional[str] = Field(default=None, description="Car identifier, or None if slot is empty")


class WheelState(BaseModel):
    wheel_index: int = Field(..., description="Wheel identifier")
    front_slot_index: int = Field(..., description="Front slot index for this wheel")
    front_slot_occupied: bool = Field(..., description="True if the front slot currently holds a car")
    front_car_id: Optional[str] = Field(default=None, description="Car ID at the accessible slot, or None if empty")
    slots: List[SlotState] = Field(..., description="All slots in this wheel")


class QueuedCar(BaseModel):
    car_id: str = Field(..., description="Unique car identifier")
    arrival_step: int = Field(..., description="Simulation step at which the car joined the arrival queue")


class PendingRetrieval(BaseModel):
    car_id: str = Field(..., description="Unique car identifier requesting retrieval")
    wheel_index: int = Field(..., description="Wheel containing the car requesting retrieval")
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
    wheel_count: int = Field(..., description="Number of rotary wheels in the installation")
    wheels: List[WheelState] = Field(..., description="All wheels in the installation")
    slots: List[SlotState] = Field(..., description="Flattened slot view across all wheels")

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
    wheels: List[WheelState] = Field(..., description="Current occupancy of all wheels")
    slots: List[SlotState] = Field(..., description="Flattened slot view across all wheels")
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
