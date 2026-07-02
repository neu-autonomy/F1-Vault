"""
Model Predictive Path Integral (MPPI) controller — generic, model-agnostic core.

MPPI is a sampling-based MPC: each control step it samples K noisy action
sequences around a nominal plan, rolls all K through a (black-box) dynamics
model in parallel, scores each rollout with a cost function, and updates the
nominal plan as a cost-weighted (softmax) average of the sampled noise. The
first action of the updated plan is executed; the plan is then shifted forward
one step (warm start) for the next call.

This maps directly onto the learned jump-dynamics model: a batched rollout of B
candidate action sequences is exactly `JumpDynamics.rollout`. This module knows
nothing about that model or about terrain — it takes:

    rollout_fn(core0 (K,D), actions (K,T,A)) -> traj (K, T+1, D)
    cost_fn(traj (K,T+1,D), actions (K,T,A))  -> cost (K,)

so it can be unit-tested against a toy dynamics and reused unchanged.

Reference: Williams et al., "Model Predictive Path Integral Control:
From Theory to Parallel Computation" (2017).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPPIConfig:
    horizon: int = 10                 # T: planning steps (control_dt=0.1s -> 1.0s).
    #   Kept short on purpose: the model's rollout error compounds over long horizons
    #   (CONTROLLER_HANDOFF.md), so a long plan optimises against a drifting prediction.
    num_samples: int = 1024           # K: rollouts per plan step (batched on GPU)
    temperature: float = 1.0          # lambda: softmax sharpness (lower = greedier)
    # per-dimension exploration noise std, in ACTION units ([-1,1] space). Steering
    # noise is smaller than throttle: excess steering wiggle at a ramp spoils the
    # launch (the model only launches cleanly when hit near-straight).
    noise_sigma: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.18]))
    # action box limits [low, high] per dimension. Throttle low defaults to 0.0
    # because the hardware cannot reverse and the model treats throttle<=0 as
    # out-of-distribution coast/brake (see CONTROLLER_HANDOFF.md caveats).
    u_low: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))
    u_high: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    # nominal plan the controller is initialised / re-seeded with each terminal step
    u_init: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.0]))
    # fraction of samples drawn as pure exploration around u_init instead of the
    # warm-started nominal (helps escape a bad plan); 0 disables.
    explore_fraction: float = 0.05
    seed: int = 0


class MPPI:
    """Receding-horizon MPPI planner over a black-box dynamics + cost.

    Usage:
        planner = MPPI(action_dim=2, cfg=MPPIConfig())
        u = planner.command(core_state, rollout_fn, cost_fn)   # (2,) action to apply
        # ... step the real plant with u, observe next state, repeat ...
    """

    def __init__(self, action_dim, cfg=None):
        self.cfg = cfg or MPPIConfig()
        self.A = action_dim
        self.T = self.cfg.horizon
        self.rng = np.random.default_rng(self.cfg.seed)
        # nominal plan U: (T, A), warm-started across calls
        self.U = np.tile(np.asarray(self.cfg.u_init, np.float32), (self.T, 1))
        # cache of the last planning pass, for logging / visualisation
        self.last = {}

    def reset(self):
        """Forget the warm-started plan (e.g. between episodes)."""
        self.U = np.tile(np.asarray(self.cfg.u_init, np.float32), (self.T, 1))
        self.last = {}

    def _clip(self, actions):
        return np.clip(actions, self.cfg.u_low, self.cfg.u_high)

    def command(self, core_state, rollout_fn, cost_fn):
        """Plan from `core_state` and return the first action to execute.

        core_state: (D,) current state of the plant.
        rollout_fn: (K,D),(K,T,A) -> (K,T+1,D)
        cost_fn:    (K,T+1,D),(K,T,A) -> (K,)
        returns:    (A,) action, already clipped to the action box.
        """
        K, T, A = self.cfg.num_samples, self.T, self.A
        sigma = np.asarray(self.cfg.noise_sigma, np.float32)

        # --- sample noisy control sequences around the nominal plan ---------
        noise = self.rng.normal(0.0, 1.0, size=(K, T, A)).astype(np.float32) * sigma
        V = self.U[None] + noise                                    # (K,T,A)

        # a slice of samples explore around u_init instead of the warm plan
        n_exp = int(self.cfg.explore_fraction * K)
        if n_exp > 0:
            u0 = np.asarray(self.cfg.u_init, np.float32)
            V[:n_exp] = u0[None, None] + noise[:n_exp]
        V = self._clip(V)
        # store the *clipped* deviation so the update stays inside the box
        eps = V - self.U[None]                                      # (K,T,A)

        # --- roll out all candidates and score them -------------------------
        core0 = np.tile(np.asarray(core_state, np.float32), (K, 1))  # (K,D)
        traj = np.asarray(rollout_fn(core0, V), np.float32)          # (K,T+1,D)
        costs = np.asarray(cost_fn(traj, V), np.float32)             # (K,)

        # --- softmax (path-integral) weights over rollout cost --------------
        beta = costs.min()
        w = np.exp(-(costs - beta) / max(self.cfg.temperature, 1e-6))
        w /= w.sum() + 1e-9                                          # (K,)

        # --- update nominal plan, then warm-start-shift it forward ----------
        self.U = self._clip(self.U + np.einsum("k,kta->ta", w, eps))
        u_exec = self.U[0].copy()

        self.last = {
            "costs": costs, "weights": w, "trajectories": traj,
            "actions": V, "best_idx": int(costs.argmin()),
            "planned_traj": None,  # filled below
        }
        # roll the executed action off the front; re-seed the tail with u_init
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = np.asarray(self.cfg.u_init, np.float32)

        return u_exec
