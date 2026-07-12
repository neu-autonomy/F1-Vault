# learned_dynamics — pure-learning dynamics models

End-to-end learned one-step dynamics models for the robot, separate from the
physics-informed `f1_vault/` pipeline (Matt's — don't modify from here).
Full project context, data-format docs, and the current findings live in the
repo-root `CLAUDE.md`.

All commands run from the repo root:

```bash
# Data health report (no model needed) — run this first on any new H5 file
python learned_dynamics/inspect_data.py --data data/raw/dynamics_data_0000.h5

# Train the CNN (writes models/cnn_dynamics_v3/)
python learned_dynamics/train_cnn.py --epochs 50

# Numerical sanity check of a checkpoint (per-axis MAE vs predict-zero baseline)
python learned_dynamics/sanity_check.py \
    --model models/cnn_dynamics_v3/best_model.pt --data data/raw/dynamics_data_0000.h5

# THE go/no-go test: do rollouts actually respond to actions?
python learned_dynamics/action_conditioning_test.py \
    --model models/cnn_dynamics_v3/best_model.pt --data data/raw/dynamics_data_0000.h5

# Full diagnostic figures (scatter, histograms, synthetic rollouts) -> figures/
python learned_dynamics/visualize_trajectories.py \
    --model models/cnn_dynamics_v3/best_model.pt --data data/raw/dynamics_data_0000.h5
```

Files:

| File | Purpose |
|---|---|
| `dynamics_model.py` | Shared architecture, checkpoint loading, rollout helpers — single source of truth |
| `dynamics_api.py` | **Controller-facing inference API** (`JumpDynamics`): single-step + terrain-aware batched rollout. See `CONTROLLER_HANDOFF.md` |
| `jump_regime_eval.py` | Per-regime accuracy (flat/ramp/airborne) + velocity/steering checks + real-jump tracking |
| `train_cnn.py` | CNN training. `--horizon N` = multi-step rollout training (roadmap #1); `horizon=1` (default) = original one-step. Boundary filter + target-only normalization. `data_file` can be a dir |
| `train_ensemble.py` | Train a K-member deep ensemble via `train_cnn.py` (roadmap #3); passes `--horizon` through. Spreads members over GPUs |
| `rollout_eval.py` | **Closed-loop N-step** rollout error (the metric #1 targets) comparing checkpoints/ensembles on real jumps; ensemble uncertainty/error calibration |
| `ENHANCEMENTS.md` | Design + usage + incremental roadmap for enhancements #1–#3 |
| `train_rnn.py` | LSTM variant — **stale**, see its docstring before using |
| `inspect_data.py` | H5 health report: episode structure, action stats, integrity |
| `sanity_check.py` | Checkpoint vs data numerical checks |
| `action_conditioning_test.py` | Same start, 6 action sequences — PASS/WEAK/FAIL verdict |
| `visualize_trajectories.py` | Per-transition accuracy figures + closed-loop rollouts |
| `figures/` | Output figures (gitignored) |

## Improvement roadmap

The base (`cnn_dynamics_v3`, 1.2M params, 695K samples, 100 epochs) is solid for
single-step prediction and short-horizon control. Ordered by expected payoff:

**1–3 are IMPLEMENTED on branch `emir_model_controller_enhancements` — see
[`ENHANCEMENTS.md`](ENHANCEMENTS.md) for design, usage, and the incremental roadmap.**

**1. Multi-step / rollout training loss (highest impact for control). ✅ DONE.**
`train_cnn.py --horizon N` unrolls N steps (predict → feed own core state back, recorded
terrain each step, backprop through the window) with scheduled sampling. `horizon=1` is the
unchanged single-step default. Measure the win with `rollout_eval.py`.

**2. Throttle/action diversity in data collection. ✅ DONE.**
`collect_dynamics_data.py --throttle_dist {mixture(default),full,highbias}` covers the full
`[0,1]` throttle range (v3 was forward-biased, mean 0.72). A `mixture` dataset is in
`data/raw_mixture/`. Confirm with `inspect_data.py`.

**3. Probabilistic / ensemble dynamics (for risk-aware MPPI). ✅ DONE (deep ensemble).**
`train_ensemble.py` + `dynamics_api.EnsembleJumpDynamics` (mean + disagreement variance) +
`make_risk_aware_planner_fns` (MPPI-compatible, no change to `mppi.py`). Started with a
deep ensemble on purpose (simplest robust win); variance heads / full PETS are the next
increment — see `ENHANCEMENTS.md`.

**4. Trim the model and the inputs.**
- Drop the unused terrain-decoder head (the model predicts Δelevation but control feeds terrain
  in — the decoder is dead weight; removing it shrinks the model and speeds rollout).
- Drop unused/redundant state dims: the 20 joint dims (state[22:44]) are never used; the core
  stores both quaternion and Euler and both body- and world-frame velocity (redundant).

**5. Data augmentation (cheap accuracy).**
Left/right mirror symmetry (flip Y, yaw, steering) doubles the data for free and enforces a
physical symmetry. Optionally small terrain-map rotations.

**6. Bigger model / longer training / hyperparameter sweep.**
Val loss was still descending — more capacity (latent dim, MLP width) + more epochs + an LR
sweep will give incremental gains. Lower priority than 1–3.

**7. Longer-horizon + held-out-terrain evaluation.**
Current eval is single-step and short rollouts on the training terrain. Add a longer-horizon
rollout metric and evaluate on ramp geometries *not* seen in training to measure generalization.

**8. Sim-to-real (if it ever goes on hardware).**
Domain randomization (friction, mass, latency, sensor noise) during collection so the model
transfers to the real MuSHR.
