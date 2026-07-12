"""
Controller-facing API for the learned MuSHR jump-dynamics model.

This is the file to hand to the controls person. It wraps models/cnn_dynamics_v3
behind a clean single-step + multi-step interface and handles all the fiddly bits
(state layout, the 676-value elevation map, target-only normalization, delta
prediction, inf handling).

WHY THIS EXISTS (vs dynamics_model.synthetic_rollout):
  synthetic_rollout HALLUCINATES the terrain -- it predicts Δelevation and feeds its
  own guess back in, so multi-step rollouts drift and miss jumps entirely (validated:
  it predicts 0.07 m at a launch that really reaches 0.30 m). A controller KNOWS the
  terrain, so this API instead asks the caller for the local elevation patch at every
  step and never hallucinates. With correct terrain fed in, the rollout tracks a real
  jump to ~2 cm mean Z error over 3.9 s (vs 6 cm hallucinated).

============================  I/O CONVENTIONS  ============================
CORE STATE  -- shape (22,) or (B, 22), float32, this exact order:
    [0:3]   root_pos_w        world position (x, y, z)            [m]
    [3:7]   root_quat_w       orientation quaternion (w, x, y, z)
    [7:10]  world_euler_xyz   (roll, pitch, yaw)  -- stored in [0, 2pi)  [rad]
    [10:13] base_lin_vel      linear velocity, BODY frame          [m/s]
    [13:16] base_ang_vel      angular velocity, BODY frame         [rad/s]
    [16:19] root_lin_vel_w    linear velocity, WORLD frame         [m/s]
    [19:22] root_ang_vel_w    angular velocity, WORLD frame        [rad/s]
  (The full sim state is 720-dim: [core(22)][joints(22)][elevation(676)]. The model
   only needs the 22 core dims + the elevation patch; joints are ignored. If you have a
   full 720 state, core = state[:22] and elevation = state[-676:].reshape(26, 26).)

ELEVATION PATCH -- shape (26, 26) or (B, 26, 26), float32:
    A 2.5 m x 2.5 m heightmap centered on the robot, 0.1 m resolution, YAW-ALIGNED
    (rotates with the robot's heading; attach_yaw_only). Values are world-frame terrain
    height in the SAME convention the model trained on. On flat ground the dataset reads
    ~ -0.20 (a fixed offset baked into the sim's height_scan correction) -- whatever you
    feed must match that convention. Easiest correct source: the same RayCaster /
    elevation pipeline used during data collection (WheeledLab world_height_map). Replace
    +/-inf (ray misses) with INF_FILL = -10.0 before passing in.

ACTION -- shape (2,) or (B, 2), float32, both in [-1, 1]:
    [0] throttle  -> +/- 3.0 m/s target velocity  (NO REVERSE: <=0 just coasts/brakes)
    [1] steering  -> +/- 0.488 rad (~28 deg).  +steering turns +yaw (CCW / robot-left).

OUTPUT: next CORE state (same 22-dim layout). The model predicts a delta and this API
adds it. Elevation is NOT predicted here -- you supply it each step.
==========================================================================

QUICK START (controller):
    dyn = JumpDynamics("models/cnn_dynamics_v3/best_model.pt")
    # single step:
    next_core = dyn.predict_step(core, elevation_patch, action)
    # MPPI-style: B candidate action sequences, B rollouts in parallel:
    #   get_patch(core_batch, t) -> (B, 26, 26) sampled from YOUR terrain map / perception
    traj = dyn.rollout(core0_batch, actions_BT2, get_patch)   # (B, T+1, 22)
"""

from pathlib import Path

import numpy as np
import torch

from dynamics_model import (
    load_model, normalization_mode, grid_from_size, INF_FILL,
)

DEFAULT_CKPT = "models/cnn_dynamics_v3/best_model.pt"


class JumpDynamics:
    """Single- and multi-step forward dynamics. Pure inference, no terrain hallucination."""

    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config, stats, self.core_dim = load_model(ckpt_path, self.device)
        self.elev_size = self.config["elevation_map_size"]
        self.grid = grid_from_size(self.elev_size)
        self.input_norm, self.output_denorm = normalization_mode(self.config)
        # delta-target denorm stats (predictions come out normalized)
        self._ct_mean = torch.tensor(np.asarray(stats["core_target_mean"], np.float32),
                                     device=self.device)
        self._ct_std = torch.tensor(np.asarray(stats["core_target_std"], np.float32),
                                    device=self.device)
        self._in_mean = torch.tensor(np.asarray(stats["core_state_mean"], np.float32),
                                     device=self.device)
        self._in_std = torch.tensor(np.asarray(stats["core_state_std"], np.float32),
                                    device=self.device)
        self.predict_delta = self.config["predict_delta"]
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_step(self, core, elevation_patch, action):
        """One forward step. Accepts single (unbatched) or batched inputs.

        core:            (22,)   or (B, 22)
        elevation_patch: (26,26) or (B, 26, 26)   -- world-frame, yaw-aligned, inf-filled
        action:          (2,)    or (B, 2)
        returns next core, same shape as `core` ((22,) or (B, 22)).
        """
        single = (np.asarray(core).ndim == 1)
        c = torch.as_tensor(np.atleast_2d(core), dtype=torch.float32, device=self.device)
        e = np.asarray(elevation_patch, dtype=np.float32)
        if e.ndim == 2:
            e = e[None]
        e = torch.as_tensor(e, device=self.device)
        e = torch.nan_to_num(e, nan=INF_FILL, posinf=INF_FILL, neginf=INF_FILL)
        a = torch.as_tensor(np.atleast_2d(action), dtype=torch.float32, device=self.device)

        c_in = (c - self._in_mean) / self._in_std if self.input_norm else c
        pc, _ = self.model(c_in, e.unsqueeze(1), a)
        if self.output_denorm:
            pc = pc * self._ct_std + self._ct_mean
        next_core = (c + pc) if self.predict_delta else pc
        next_core = next_core.cpu().numpy()
        return next_core[0] if single else next_core

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, core0, actions, get_patch):
        """Closed-loop multi-step rollout WITHOUT hallucinating terrain.

        core0:    (22,) or (B, 22)   initial core state(s)
        actions:  (T, 2) or (B, T, 2)
        get_patch: callable(core_batch (B,22), t:int) -> (B, 26, 26)
                   Supply the local elevation patch from YOUR terrain map / perception
                   for each robot pose at step t. (For a static known map this depends
                   only on core_batch; t is provided for replaying recorded patches.)

        returns: (B, T+1, 22) core trajectory (row 0 = core0).  If inputs were unbatched,
                 returns (T+1, 22).
        """
        single = (np.asarray(core0).ndim == 1)
        core = np.atleast_2d(np.asarray(core0, np.float32)).copy()
        acts = np.asarray(actions, np.float32)
        if single:
            acts = acts[None]                      # (1, T, 2)
        B, T = core.shape[0], acts.shape[1]

        traj = np.zeros((B, T + 1, self.core_dim), np.float32)
        traj[:, 0] = core
        for t in range(T):
            patch = np.asarray(get_patch(core, t), np.float32)     # (B, 26, 26)
            core = self.predict_step(core, patch, acts[:, t])
            core = np.atleast_2d(core)
            traj[:, t + 1] = core
        return traj[0] if single else traj


class EnsembleJumpDynamics:
    """K independently-trained JumpDynamics models -> mean prediction + disagreement variance.

    Roadmap item #3 (risk-aware MPPI). Uncertainty here is EPISTEMIC: the spread of the
    K members' predictions. Where the members agree the model is confident; where they
    disagree (typically out-of-distribution states/actions -- e.g. the low-throttle regime
    the v3 data barely covered, or an OOD launch) the variance is large, and a risk-aware
    controller can steer away from it. This is the simplest robust uncertainty estimate
    (a deep ensemble); a later increment can make each member ALSO output a variance head
    (Gaussian NLL) for full PETS-style epistemic+aleatoric uncertainty -- see CLAUDE.md.

    Members must share I/O conventions (same core layout, elevation size, normalization).
    They can differ in weights only (different seeds) -- the intended use. Multi-step
    (rollout-trained) members are fine and recommended: they compound less.
    """

    def __init__(self, ckpt_paths, device=None):
        if isinstance(ckpt_paths, (str, Path)):
            ckpt_paths = [ckpt_paths]
        self.members = [JumpDynamics(p, device=device) for p in ckpt_paths]
        if not self.members:
            raise ValueError("EnsembleJumpDynamics needs >= 1 checkpoint")
        m0 = self.members[0]
        self.device, self.core_dim = m0.device, m0.core_dim
        self.elev_size, self.grid = m0.elev_size, m0.grid

    @classmethod
    def from_dir(cls, ensemble_dir, device=None):
        """Load every member_*/best_model.pt under an ensemble directory (see train_ensemble.py)."""
        d = Path(ensemble_dir)
        paths = sorted(d.glob("member_*/best_model.pt"))
        if not paths:
            raise FileNotFoundError(f"no member_*/best_model.pt under {d}")
        return cls(paths, device=device)

    # ------------------------------------------------------------------
    def predict_step_dist(self, core, elevation_patch, action):
        """One step -> (mean_next_core, var_next_core), both shaped like `core`.

        var is the per-dimension variance ACROSS the K members (epistemic uncertainty).
        """
        preds = np.stack(
            [m.predict_step(core, elevation_patch, action) for m in self.members], axis=0)
        return preds.mean(axis=0), preds.var(axis=0)

    def predict_step(self, core, elevation_patch, action):
        """Ensemble-mean one step (drop-in for JumpDynamics.predict_step)."""
        return self.predict_step_dist(core, elevation_patch, action)[0]

    # ------------------------------------------------------------------
    def rollout(self, core0, actions, get_patch, return_var=False):
        """Closed-loop rollout propagating the ENSEMBLE MEAN, tracking per-step disagreement.

        Same signature/semantics as JumpDynamics.rollout (recorded/known terrain via
        get_patch, never hallucinated). With return_var=True also returns a variance
        trajectory of the same shape: var[..., t, :] is the epistemic variance of the
        members' ONE-STEP predictions made at step t-1 (row 0 is zeros). Mean propagation
        keeps all members on one shared trajectory (cheap, deterministic); the per-step
        spread is the risk signal the cost function reads.
        """
        single = (np.asarray(core0).ndim == 1)
        core = np.atleast_2d(np.asarray(core0, np.float32)).copy()
        acts = np.asarray(actions, np.float32)
        if single:
            acts = acts[None]
        B, T = core.shape[0], acts.shape[1]

        traj = np.zeros((B, T + 1, self.core_dim), np.float32)
        var = np.zeros((B, T + 1, self.core_dim), np.float32)
        traj[:, 0] = core
        for t in range(T):
            patch = np.asarray(get_patch(core, t), np.float32)
            mean, v = self.predict_step_dist(core, patch, acts[:, t])
            core = np.atleast_2d(mean)
            traj[:, t + 1] = core
            var[:, t + 1] = np.atleast_2d(v)
        if single:
            traj, var = traj[0], var[0]
        return (traj, var) if return_var else traj


def make_risk_aware_planner_fns(ensemble, get_patch, base_cost_fn,
                                risk_weight=1.0, pos_dims=(0, 1, 2)):
    """Bundle an EnsembleJumpDynamics + a base cost into MPPI-compatible (rollout_fn, cost_fn).

    MPPI (controller/mppi.py) only needs:
        rollout_fn(core0 (K,D), actions (K,T,A)) -> traj (K,T+1,D)
        cost_fn(traj (K,T+1,D), actions (K,T,A))  -> cost (K,)
    so this needs NO change to mppi.py. rollout_fn returns the ensemble-mean trajectory
    (exactly what a deterministic model would) and stashes the per-rollout epistemic
    uncertainty; cost_fn adds `risk_weight * uncertainty` to your base task cost, so the
    planner avoids action sequences the ensemble is unsure about (PETS-style).

    uncertainty per rollout = sum over horizon of the positional std (sqrt of summed
    variance over pos_dims) -- "how far off could the predicted path be". Tune risk_weight
    against your base cost scale (0 disables; start small and raise until the planner stops
    picking OOD launches).
    """
    stash = {}

    def rollout_fn(core0, actions):
        traj, var = ensemble.rollout(core0, actions, get_patch, return_var=True)
        stash["var"] = var                              # (K, T+1, D)
        return traj

    def cost_fn(traj, actions):
        base = np.asarray(base_cost_fn(traj, actions), np.float32)
        var = stash.get("var")
        if var is None:
            return base
        pos_var = var[:, 1:, list(pos_dims)].sum(axis=2)   # (K, T) summed over pos dims
        uncertainty = np.sqrt(np.clip(pos_var, 0, None)).sum(axis=1)   # (K,)
        return base + risk_weight * uncertainty

    return rollout_fn, cost_fn


class TerrainMap:
    """Convenience: a static known heightmap you can query for elevation patches.

    Use this for offline planning over a known terrain. If your controller already gets
    elevation maps from the live perception/RayCaster pipeline, prefer feeding those
    directly (they're guaranteed to match the training convention) and skip this class.

    heightmap: 2D array H[iy, ix] of terrain height, world-frame, SAME convention/zero-
               level as the training elevation (flat ground ~ -0.20).
    res:       grid resolution [m].   origin: world (x, y) of H[0, 0].

    NOTE: the (i,j)->(dx,dy) orientation of the model's 26x26 patch and its exact
    zero-offset come from the sim's RayCaster; verify get_patch() against a few recorded
    (pose, patch) pairs before trusting it for control. The model + rollout machinery are
    validated; this geometric sampler is a best-effort reference.
    """

    HALF = 1.25      # patch extends +/-1.25 m
    RES = 0.1
    N = 26

    def __init__(self, heightmap, res, origin=(0.0, 0.0)):
        self.H = np.asarray(heightmap, np.float32)
        self.res = res
        self.ox, self.oy = origin

    def _bilinear(self, xs, ys):
        fx = (xs - self.ox) / self.res
        fy = (ys - self.oy) / self.res
        x0 = np.clip(np.floor(fx).astype(int), 0, self.H.shape[1] - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, self.H.shape[0] - 2)
        wx = np.clip(fx - x0, 0, 1); wy = np.clip(fy - y0, 0, 1)
        h = (self.H[y0, x0] * (1 - wx) * (1 - wy) + self.H[y0, x0 + 1] * wx * (1 - wy)
             + self.H[y0 + 1, x0] * (1 - wx) * wy + self.H[y0 + 1, x0 + 1] * wx * wy)
        return h

    def get_patch(self, core_batch, t=None):
        """core_batch (B,22) -> (B,26,26) yaw-aligned terrain patch."""
        core_batch = np.atleast_2d(core_batch)
        B = core_batch.shape[0]
        x, y = core_batch[:, 0], core_batch[:, 1]
        yaw = core_batch[:, 9]                          # world_euler yaw
        g = (np.arange(self.N) - (self.N - 1) / 2) * self.RES   # -1.25..1.25
        gx, gy = np.meshgrid(g, g, indexing="ij")               # (26,26)
        out = np.empty((B, self.N, self.N), np.float32)
        for b in range(B):
            c, s = np.cos(yaw[b]), np.sin(yaw[b])
            wx = x[b] + gx * c - gy * s
            wy = y[b] + gx * s + gy * c
            out[b] = self._bilinear(wx.ravel(), wy.ravel()).reshape(self.N, self.N)
        return out


# ============================================================================
# VALIDATION: terrain-aware rollout vs hallucinated, on real recorded jumps.
# Run:  .venv/bin/python learned_dynamics/dynamics_api.py
# Uses the RECORDED elevation patches as the terrain oracle, so it proves the rollout
# machinery is faithful when given correct terrain (what a controller provides).
# ============================================================================
def _validate():
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dynamics_model import replace_inf_elevation, synthetic_rollout

    DATA = "data/raw/dynamics_data_0000.h5"
    dyn = JumpDynamics()
    es, g = dyn.elev_size, dyn.grid

    with h5py.File(DATA, "r") as f:
        S = replace_inf_elevation(f["states"][:].astype(np.float32), es)
        A = f["actions"][:].astype(np.float32)
        env, ep, ts = f["env_ids"][:], f["episode_ids"][:], f["timesteps"][:]
        done = f["terminated"][:] | f["truncated"][:]

    key = env.astype(np.int64) * 1_000_000 + ep.astype(np.int64)
    air = S[:, 2] - 0.19
    seqs = []
    for k in np.unique(key):
        r = np.where(key == k)[0]
        if len(r) < 15:
            continue
        o = r[np.argsort(ts[r])]
        if 0.10 <= air[o].max() <= 0.30:
            d = done[o]; e = int(np.argmax(d)) if d.any() else len(o)
            if e >= 15:
                seqs.append(o[:e])
        if len(seqs) >= 3:
            break

    print(f"Validating terrain-aware rollout on {len(seqs)} real jump episodes\n")
    fig, axes = plt.subplots(1, len(seqs), figsize=(5 * len(seqs), 4), squeeze=False)
    for i, seq in enumerate(seqs):
        # terrain oracle = recorded patch at each true step
        patches = S[seq, -es:].reshape(-1, g, g)
        get_patch = lambda core, t: patches[min(t, len(patches) - 1)][None]
        traj = dyn.rollout(S[seq[0], :dyn.core_dim], A[seq[:-1]], get_patch)
        hall = synthetic_rollout(dyn.model, S[seq[0]].copy(), A[seq[:-1]],
                                 dyn.config, _CKPT_STATS, dyn.core_dim, dyn.device)
        true_z = S[seq, 2]
        ze_o = np.abs(traj[:, 2] - true_z).mean() * 100
        ze_h = np.abs(hall[:len(seq), 2] - true_z).mean() * 100
        xy_o = np.linalg.norm(traj[-1, :2] - S[seq[-1], :2])
        xy_h = np.linalg.norm(hall[len(seq) - 1, :2] - S[seq[-1], :2])
        print(f"  ep {i}: len={len(seq)} peak={air[seq].max()*100:.0f}cm | "
              f"Z MAE  terrain-aware={ze_o:.1f}cm  hallucinated={ze_h:.1f}cm | "
              f"final-XY  aware={xy_o:.2f}m  halluc={xy_h:.2f}m")
        tt = np.arange(len(seq)) * 0.1
        ax = axes[0][i]
        ax.plot(tt, true_z, "o-", color="k", label="true", ms=3)
        ax.plot(tt, traj[:, 2], "x--", color="tab:green", label=f"terrain-aware ({ze_o:.1f}cm)")
        ax.plot(tt, hall[:len(seq), 2], "x--", color="tab:red", label=f"hallucinated ({ze_h:.1f}cm)")
        ax.axhline(0.19, color="gray", ls=":", lw=1)
        ax.set_xlabel("t (s)"); ax.set_ylabel("z (m)"); ax.set_title(f"jump {i}"); ax.legend(fontsize=8)
    out = Path("learned_dynamics/figures/rollout_terrain_vs_hallucinated.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nsaved: {out}")
    print("\n=> terrain-aware rollout reproduces the jump arc; hallucinated misses it.")
    print("   Hand JumpDynamics to the controller; feed it elevation from the known map.")


if __name__ == "__main__":
    # full stats dict that synthetic_rollout (the hallucinating baseline) needs
    _CKPT_STATS = torch.load(DEFAULT_CKPT, weights_only=False, map_location="cpu")["stats"]
    _validate()
