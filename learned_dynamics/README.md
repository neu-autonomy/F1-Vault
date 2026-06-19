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
| `train_cnn.py` | One-step CNN training (boundary filter + target-only normalization). `data_file` can be a dir — trains on all batches |
| `train_rnn.py` | LSTM variant — **stale**, see its docstring before using |
| `inspect_data.py` | H5 health report: episode structure, action stats, integrity |
| `sanity_check.py` | Checkpoint vs data numerical checks |
| `action_conditioning_test.py` | Same start, 6 action sequences — PASS/WEAK/FAIL verdict |
| `visualize_trajectories.py` | Per-transition accuracy figures + closed-loop rollouts |
| `figures/` | Output figures (gitignored) |

## Improvement roadmap (future reference — not yet implemented)

The current base (`cnn_dynamics_v3`, 1.2M params, 695K samples, 100 epochs) is solid for
single-step prediction and short-horizon control. Ordered by expected payoff:

**1. Multi-step / rollout training loss (highest impact for control).**
The model is trained on *single-step* error, but the controller uses *multi-step* rollouts,
so prediction errors compound (exposure bias). Train on N-step rollouts (predict, feed own
output back, backprop through the unrolled sequence) — or add scheduled sampling. This is the
single biggest lever for closed-loop fidelity over long horizons.

**2. Throttle/action diversity in data collection.**
Current data is forward-throttle-biased (mean 0.72, min ~0.06), so low/zero/reverse-throttle
prediction is out-of-distribution extrapolation. If the controller needs accurate braking or
speed modulation, recollect with throttle sampled across its full range (and accept it will
jump less). Re-run `inspect_dynamics_data.py` to confirm the new distribution.

**3. Probabilistic / ensemble dynamics (for risk-aware MPPI).**
Train a small ensemble (or a model that outputs variance) so the controller knows prediction
*uncertainty* — lets MPPI avoid regions the model is unsure about (PETS-style). Big robustness
win for sampling-based control.

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
