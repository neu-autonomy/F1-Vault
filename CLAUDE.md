# F1-Vault — project context for Claude

Robot–terrain dynamics prediction (NEU autonomy lab). The goal is a model
that predicts how an off-road robot moves over rough terrain, usable for
control (MPC). There are **two parallel workstreams owned by different
people** — keep them separate:

| Workstream | Owner | Code | Status |
|---|---|---|---|
| Physics-informed (ODE solver, "dphysics" from MonoForce) | Matt | `f1_vault/` package, `train_best.py`, `tests/`, `configs/`, `checkpoints/`, `best_model_exp0.pt`, `setup.py`, root `README.md` | **Do not modify unless explicitly asked** |
| Pure learned dynamics (CNN/RNN) | Emir (branch `emir`) | `learned_dynamics/`, `data/visual/` | Active; see findings below |

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
| 24:95 | joint positions/velocities + unidentified dims (some contain `inf`) |
| 95:720 | **elevation map: LAST 625 values = 25×25 grid** (contains `inf` — replace with −10 before feeding a model) |

Actions: `[throttle, steering]`, both in [−1, 1]; throttle scales to
±3.0 m/s target velocity, steering to ±0.488 rad. Always read
`f.attrs["elevation_map_size"]` rather than hard-coding.

**⚠ Historical bug:** `f1_vault/data/structures.py` and all pre-June-2026
CNN code assumed the elevation map was the last **676** values (26×26).
This file stores **625** (25×25) — the wrong slice pulls 51 non-elevation
dims into the map and misaligns every grid row. `learned_dynamics/` code
now reads the layout from the H5 attrs; checkpoints in
`models/cnn_dynamics*` (v2 and earlier) were trained on the scrambled map.

**Boundary rows:** where `terminated|truncated` is True, `next_states[i]`
is the post-reset spawn position (a teleport, up to ~9 m), not physics.
Filter them in **both** training and eval — mixing filtered training with
unfiltered eval once produced 10× MAE disagreements.

## Why the learned models have failed so far (diagnosed 2026-06-10)

The CNN looks good on single-step metrics (MAE 0.17–0.65× the
predict-zero baseline, scatter ρ ≈ 0.94–0.97) but **fails the action
conditioning test**: rollouts from the same start under full-throttle vs
reverse vs hard-left are nearly identical. Root cause is the **data, not
the architecture**:

1. **Episodes die instantly.** Median episode length is 2 steps; 81% are
   ≤2 steps; 45.6% of all rows are boundary rows. Something in the sim's
   termination condition kills nearly every episode at spawn.
2. **The data is spawn transients.** 43% of boundary-filtered rows are
   timestep 0 and 16% are timestep 1. At spawn the robot is launched
   upward (world vz ≈ +3 m/s at t=1; Z rises 0.25→0.78 m then falls). The
   model memorizes this ballistic arc — every rollout shows the same
   "Z up to ~0.8 then settle" shape regardless of action.
3. **No action signal to learn.** Over one 0.1 s step the action barely
   moves the state: max |corr(action, Δpos)| ≈ 0.16 (throttle↔ΔY);
   steering ≈ 0.03. Ballistic/spawn dynamics dominate, so ignoring the
   action costs the model almost nothing in training loss.

**Conclusion:** retraining or changing architecture won't fix action
conditioning. The fix is upstream data collection: stop the
instant-termination at spawn (and ideally drop the first few post-spawn
steps), then recollect with long episodes. `inspect_data.py` prints the
episode-structure stats to verify a new file before training on it.

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
(figures → `learned_dynamics/figures/`).

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
