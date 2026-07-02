# F1-Vault — project context for Claude

Robot–terrain dynamics prediction (NEU autonomy lab). The goal is a model
that predicts how an off-road robot moves over rough terrain, usable for
control (MPC). There are **two parallel workstreams owned by different
people** — keep them separate:

| Workstream | Owner | Code | Status |
|---|---|---|---|
| Physics-informed (ODE solver, "dphysics" from MonoForce) | Matt | `f1_vault/` package, `train_best.py`, `tests/`, `configs/`, `checkpoints/`, `best_model_exp0.pt`, `setup.py`, root `README.md` | **Do not modify unless explicitly asked** |
| Pure learned dynamics (CNN/RNN) | Emir (branch `emir`) | `learned_dynamics/`, `data/visual/` | Active; see findings below |
| MPPI controller (plans through the learned model) | Emir (branch `emir_controller`) | `controller/` | New; see `controller/README.md` |

Environment: use `.venv/bin/python` (torch + CUDA available). Run all
scripts from the repo root. `models/`, `figures/` dirs and `*.zip` are
gitignored; trained checkpoints live in `models/<run_name>/`.

## Data format (`data/raw/dynamics_data_0000.h5`)

One-step transition tuples collected from 512 parallel sim envs.
Rows are **interleaved across envs** (all envs at t, then t+1, ...), not
contiguous per episode — sort by `(env_ids, timesteps)` to reconstruct
episodes. Keys: `states`, `actions`, `next_states`, `terminated`,
`truncated`, `env_ids`, `episode_ids`, `timesteps`. `control_dt = 0.1s`.

State vector (720 dims) — verified against the file, documented in
`data/visual/play_h5_file.py`:

| dims | content |
|---|---|
| 0:3 | root position, world frame (x, y, z) |
| 3:7 | root quaternion (w, x, y, z) |
| 7:10 | euler roll/pitch/yaw, stored in [0, 2π] (redundant with quat) |
| 10:13 | linear velocity, **body** frame |
| 13:16 | angular velocity, body frame |
| 16:19 | linear velocity, world frame |
| 19:22 | angular velocity, world frame |
| 22:24 | last action (throttle, steering) |
| 24:44 | joint positions (24:34) + joint velocities (34:44), 20 dims — not used by the model |
| 44:720 | **elevation map: LAST 676 values = 26×26 grid** (may contain `inf` on ray-miss — replace before feeding a model) |

Actions: `[throttle, steering]`, both in [−1, 1]; throttle scales to
±3.0 m/s target velocity, steering to ±0.488 rad.

**⚠ Elevation map is 26×26 = 676, NOT 25×25 = 625 (corrected 2026-06-13).**
`GridPatternCfg(size=2.5, resolution=0.1)` produces `2.5/0.1 + 1 = 26`
samples per axis (endpoints included) → **676**. Verified three ways against
`data/raw/dynamics_data_0000.h5`: (1) the joint region's per-column std drops
to a uniform ~0.3 starting exactly at **column 44** and runs to 720 (720−44 =
676); (2) reshaping the last 676 as 26×26 is 2–5× spatially smoother than the
last 625 as 25×25; (3) the WheeledLab `grid_pattern` source. **Slice
`states[:, -676:]`, not `-625:`.** The "51 unidentified dims with `inf`" the
old note placed at 24:95 were simply the **first 51 elevation cells** being
misread as joints.

**Do NOT trust `f.attrs["elevation_map_size"]`** — it was hardcoded to a wrong
**625** in every file collected before 2026-06-13. WheeledLab now derives it
from the sensor config (writes 676), but old files still carry the wrong attr.

**⚠ Earlier conclusion was backwards:** prior notes claimed the file stored 625
and that the 676 (26×26) assumption was the bug. It is the other way around —
676 is correct. Any `models/cnn_dynamics*` checkpoint trained on a 625/25×25
slice was trained on a **misaligned (row-sheared) elevation map** and should be
retrained on the 676 layout.

**Boundary rows:** where `terminated|truncated` is True, `next_states[i]`
is the post-reset spawn position (a teleport, up to ~9 m), not physics.
Filter them in **both** training and eval — mixing filtered training with
unfiltered eval once produced 10× MAE disagreements.

## Model status — FIXED & working (2026-06-13)

The old failure (diagnosed 2026-06-10: rollouts ignored the action; root cause was the
DATA — episodes died in ≤2 steps, 45% boundary rows, spawn transients dominated, action
corr ≈ 0.03–0.16) is **resolved**. It was an upstream data-collection problem, fixed in
WheeledLab (`emir_jumping_data_collect`): a per-env steep "kicker" ramp, reachable spawn
speed (≤3 m/s hardware cap), live steering, rollover+stuck terminations so crashes reset
instead of polluting, and the corrected 676/26×26 elevation layout.

**Base model `models/cnn_dynamics_v3`** — trained on ALL 695k rows of run_20260613_115728
(5 batches, 508k train / 127k val), 100 epochs, ReduceLROnPlateau, val loss 0.170. It:
- predicts position to **0.6–1.2 cm** single-step, **7–14× better than predict-zero**;
  on the real-jump samples (Δz 5–15 cm) it's 14–20× better — it nails the vertical motion;
- predicts the accelerations too (world Δvz beats constant-velocity 3–7× incl. airborne);
- **PASSES** the action-conditioning test; steering sign is physically correct (+steer→+yaw);
- in closed-loop rollout **with correct terrain fed in**, tracks a real jump to ~1–3 cm Z MAE
  and reproduces the launch/landing arc (the hallucinating `synthetic_rollout` misses it).

`train_cnn.py` now loads a whole directory of batches (concatenates all `*.h5`); default
`data_file = "data/raw"`. Data quality verified with `inspect_dynamics_data.py` (WheeledLab)
and `learned_dynamics/jump_regime_eval.py` (per-regime accuracy + velocity/steering checks).

**Controller handoff is ready:** `learned_dynamics/dynamics_api.py` (`JumpDynamics`) +
`learned_dynamics/CONTROLLER_HANDOFF.md`. CRITICAL: the multi-step rollout must be fed the
local elevation patch from the known terrain each step — do NOT use `synthetic_rollout`,
which hallucinates terrain and misses jumps (e.g. flies an OOD reverse action 3.8 m up).

Old broken-data model dirs (v2, cnn_dynamics, _fixed, _new, baseline) and stale figures were
removed; only the v3 base + current figures remain. Remaining/optional: add throttle
diversity if the controller needs accurate braking/low-speed (current data is
forward-throttle-biased, so low/reverse-throttle prediction is extrapolation).

## learned_dynamics/ layout

`dynamics_model.py` is the single source of truth for the architecture
(`DynamicsCNN` = CNN elevation encoder/decoder + MLP over
[core_state, action, elevation latent]), checkpoint loading
(`load_model` — handles both 26×26-era and 25×25 checkpoints),
normalization flags (`normalization_mode`) and closed-loop
`synthetic_rollout`. Don't re-declare the model in scripts.

Scripts (see `learned_dynamics/README.md` for commands): `train_cnn.py`
(current trainer → `models/cnn_dynamics_v3/`), `train_rnn.py` (stale LSTM
experiment — needs the elevation + boundary-filter fixes before reuse),
`inspect_data.py` (H5 health report), `sanity_check.py` (numeric checks),
`action_conditioning_test.py` (**the go/no-go test before using a model
for control** — prints PASS/WEAK/FAIL), `visualize_trajectories.py`
(figures → `learned_dynamics/figures/`), `jump_regime_eval.py` (per-regime
single-step accuracy + velocity/steering checks + real-jump tracking plot),
`dynamics_api.py` (**controller-facing inference API** — `JumpDynamics`
single-step + terrain-aware batched rollout; see `CONTROLLER_HANDOFF.md`).

Training conventions: predict per-step **deltas**; boundary filter on;
**target-only normalization** (inputs fed raw, only targets normalized —
the checkpoint's `target_only_normalization` flag records this; eval
tools must respect it, feeding normalized inputs to a raw-input model
once sent rollouts to 4500 m).

## Checkpoint format

`torch.save` dict with `model_state_dict`, `config_dict` (architecture +
`elevation_map_size` + normalization flags) and `stats` (input/target
means & stds as numpy arrays). Load with
`dynamics_model.load_model(path, device)`; needs
`torch.load(..., weights_only=False)`.
