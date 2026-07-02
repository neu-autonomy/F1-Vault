"""MuSHR jump controller — MPPI planning through the learned dynamics model.

Public API:
    MPPI, MPPIConfig            -- generic sampling-based MPC core (model-agnostic)
    GoalCost, GoalCostConfig    -- reach-a-goal-past-a-ramp task cost
    build_ramp_terrain, build_flat_terrain, make_get_patch, RampSpec, FLAT_Z
                                -- known synthetic terrain in the training convention

See controller/README.md for the design and controller/run_controller.py for the
closed-loop demo against JumpDynamics.
"""

from .cost import GoalCost, GoalCostConfig, GROUND_AIR_REF, quat_roll_pitch, wrap_pi
from .mppi import MPPI, MPPIConfig
from .terrain import (
    FLAT_Z, RampSpec, build_flat_terrain, build_ramp_terrain, make_get_patch,
)

__all__ = [
    "MPPI", "MPPIConfig",
    "GoalCost", "GoalCostConfig", "GROUND_AIR_REF", "quat_roll_pitch", "wrap_pi",
    "FLAT_Z", "RampSpec", "build_flat_terrain", "build_ramp_terrain",
    "make_get_patch",
]
