# learned_dynamics — pure-learning dynamics models (Emir)

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
| `train_cnn.py` | One-step CNN training (boundary filter + target-only normalization) |
| `train_rnn.py` | LSTM variant — **stale**, see its docstring before using |
| `inspect_data.py` | H5 health report: episode structure, action stats, integrity |
| `sanity_check.py` | Checkpoint vs data numerical checks |
| `action_conditioning_test.py` | Same start, 6 action sequences — PASS/WEAK/FAIL verdict |
| `visualize_trajectories.py` | Per-transition accuracy figures + closed-loop rollouts |
| `figures/` | Output figures (gitignored) |
