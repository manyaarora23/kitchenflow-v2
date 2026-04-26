"""
KitchenFlow-v2: Models
Extends v1 models exactly — same field names, same types.
v2 additions are clearly marked so validators don't break.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


# Base classes (since openenv import is invalid)
class Action(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KitchenAction(Action):
    """
    dispatch_decisions maps order_id to action int:
        0 = wait
        1 = summon_driver
        2 = request_coordinator_priority  (v2)
        3 = requery_gps                   (v2 chaos recovery)
    """
    dispatch_decisions: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of order_id to action int. 0=wait, 1=summon_driver, 2=request_priority, 3=requery_gps",
    )


class KitchenObservation(Observation):
    """
    Full episode observation.
    All v1 fields preserved exactly.
    v2 fields added at bottom.
    """

    # v1 fields (preserved exactly)
    task_id: str = ""
    task_description: str = ""
    difficulty: str = ""
    time_min: int = 0
    max_time_min: int = 30
    traffic_index: float = 1.0
    orders: List[Dict[str, Any]] = Field(default_factory=list)
    orders_delivered: int = 0
    orders_failed: int = 0
    total_temp_penalty: float = 0.0
    total_waste_penalty: float = 0.0
    last_action_feedback: str = ""
    score: float = 0.0
    attempts: int = 0
    max_attempts: int = 30

    # v2 additions
    chaos_log: List[str] = Field(default_factory=list)
    coordinator_signals: Dict[str, str] = Field(default_factory=dict)
    curriculum_level: int = 1
    chaos_level: int = 0