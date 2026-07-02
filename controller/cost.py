"""
Cost functions for the MuSHR jump-controller MPPI planner.

A cost function scores a batch of candidate rollouts (K, T+1, 22 core states)
plus the actions that produced them, and returns one scalar per rollout (lower =
better). MPPI turns these into softmax weights, so only *relative* cost matters.

`GoalCost` is the default task: drive to a goal XY position (typically placed
past a ramp) while staying upright, keeping the trajectory in the model's valid
regime, and using smooth forward throttle. See CONTROLLER_HANDOFF.md for the
regime caveats this encodes (forward-throttle bias, jumps top out ~10-15 cm, OOD
actions can fly the model several meters into the air).

Core-state indices used here (see dynamics_api.py for the full layout):
    [0:2]  x, y   world position
    [2]    z      world height   (flat ground root z ~ GROUND_AIR_REF)
    [3:7]  quat   orientation (w, x, y, z)  <- upright is read from HERE
    [16:18] world-frame vx, vy

NOTE: we take roll/pitch from the QUATERNION, not the euler channel [7:10]. The
model does not predict euler coherently (it swings 100-170 deg on straight flat
driving) — it was validated on position/velocity/yaw, not roll/pitch euler. The
quaternion stays unit-norm and upright, so it is the trustworthy tilt source.
"""

from dataclasses import dataclass

import numpy as np

# flat-ground root height; air = z - GROUND_AIR_REF (see data / handoff).
GROUND_AIR_REF = 0.19


def wrap_pi(a):
    """Wrap angle(s) stored in [0, 2pi) to [-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def quat_roll_pitch(quat):
    """Roll & pitch [rad] from quaternion(s) (..., 4) ordered (w, x, y, z).

    Returns (roll, pitch), each shaped like quat[..., 0]. atan2/asin already yield
    values in [-pi, pi] / [-pi/2, pi/2], so no wrapping is needed.
    """
    q = np.asarray(quat, np.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


@dataclass
class GoalCostConfig:
    goal_xy: np.ndarray            # (2,) target world position
    goal_radius: float = 0.25      # [m] within this of the goal counts as arrived
    w_goal: float = 1.0            # running weight on distance-to-goal
    w_goal_terminal: float = 12.0  # extra weight on final-step distance-to-goal
    w_upright: float = 4.0         # penalty on roll/pitch (rollover avoidance)
    upright_tol: float = 0.35      # [rad] free band before roll/pitch is penalised
    w_air: float = 8.0             # penalty on airborne height above air_max (OOD guard)
    air_max: float = 0.30          # [m] realistic jumps top out ~0.10-0.30 m
    w_ctrl: float = 0.02           # penalty on control magnitude (energy)
    w_smooth: float = 0.15         # penalty on control step-to-step change (jerk)
    w_reverse: float = 3.0         # penalty on throttle < 0 (out-of-distribution)
    w_speed: float = 0.30          # penalty for going SLOWER than v_target (keeps run-up)
    v_target: float = 2.8          # m/s desired approach speed (data run-up ~2.5-2.8)


class GoalCost:
    """Reach `goal_xy`, stay upright and in-distribution, drive smoothly."""

    def __init__(self, cfg: GoalCostConfig):
        self.cfg = cfg
        self.goal = np.asarray(cfg.goal_xy, np.float32)

    def __call__(self, traj, actions):
        c = self.cfg
        # traj: (K, T+1, D)   actions: (K, T, A)
        xy = traj[:, :, :2]                                  # (K,T+1,2)
        dist = np.linalg.norm(xy - self.goal[None, None], axis=-1)  # (K,T+1)

        # progress: running distance to goal + heavy terminal weight
        cost = c.w_goal * dist[:, 1:].sum(axis=1)            # skip t=0 (shared)
        cost += c.w_goal_terminal * dist[:, -1]

        # upright: penalise roll/pitch beyond a free band (hinge, squared).
        # roll/pitch come from the quaternion (the euler channel is unreliable).
        roll, pitch = quat_roll_pitch(traj[:, :, 3:7])
        tilt = np.maximum(np.abs(roll) - c.upright_tol, 0.0) ** 2 + \
            np.maximum(np.abs(pitch) - c.upright_tol, 0.0) ** 2
        cost += c.w_upright * tilt.sum(axis=1)

        # in-distribution guard: penalise flying above realistic jump height
        air = traj[:, :, 2] - GROUND_AIR_REF
        cost += c.w_air * (np.maximum(air - c.air_max, 0.0) ** 2).sum(axis=1)

        # keep the run-up: penalise being slower than v_target (one-sided, so the
        # car may still decelerate on the ramp crest as the real car does)
        speed = np.linalg.norm(traj[:, :, 16:18], axis=-1)          # (K,T+1)
        cost += c.w_speed * (np.maximum(c.v_target - speed, 0.0) ** 2).sum(axis=1)

        # control effort, smoothness, and reverse-throttle (OOD) penalties
        cost += c.w_ctrl * (actions ** 2).sum(axis=(1, 2))
        dv = np.diff(actions, axis=1)
        cost += c.w_smooth * (dv ** 2).sum(axis=(1, 2))
        throttle = actions[:, :, 0]
        cost += c.w_reverse * (np.maximum(-throttle, 0.0) ** 2).sum(axis=1)

        return cost.astype(np.float32)

    def arrived(self, core_state):
        """True once the plant is within goal_radius of the goal."""
        return float(np.linalg.norm(np.asarray(core_state)[:2] - self.goal)) <= \
            self.cfg.goal_radius
