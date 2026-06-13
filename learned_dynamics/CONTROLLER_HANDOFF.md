# Learned MuSHR jump-dynamics model — controller handoff

A trained forward-dynamics model for the MuSHR car over ramps, with a clean inference
API for use in a controller (MPC / MPPI / whatever rolls out candidate action sequences).

## What you get

- **Model:** `models/cnn_dynamics_v3/best_model.pt` — one-step dynamics
  `(core_state, elevation_patch, action) -> next_core_state`.
- **API:** `learned_dynamics/dynamics_api.py` → `JumpDynamics` (single-step + batched
  rollout) and `TerrainMap` (optional offline terrain sampler).
- All conventions (state layout, units, elevation, action scaling) are documented at the
  top of `dynamics_api.py`. Read that first.

## How good is it

Trained on 695k transitions (508k train / 127k held-out val), 100 epochs.

Single-step (held-out val): position predicted to **0.6–1.2 cm, 7–14× better than
predict-zero**; on the real-jump samples (Δz 5–15 cm) it's **14–20× better** — it nails the
vertical motion, not just flat driving. It predicts the *accelerations* too (launch impulse,
gravity, landing), beating constant-velocity 3–7×. Steering sign is physically correct
(+steering → +yaw / left).

Multi-step closed-loop, **with correct terrain fed in**, tracks a real ~3.9 s jump to
~1–3 cm mean Z error and reproduces the launch/landing arc (see
`figures/rollout_terrain_vs_hallucinated.png`). Reminder: feed real terrain — the
hallucinating `synthetic_rollout` misses jumps and can fly an out-of-distribution action
several meters into the air.

## The one thing you must do right: terrain

The model does **not** know the terrain — you feed it the local elevation patch each step.
Do **not** use `dynamics_model.synthetic_rollout`; it hallucinates terrain and misses jumps
(red curve in the figure). Use `JumpDynamics.rollout` and supply elevation from the known
map / perception.

The elevation patch is a **26×26, 2.5 m × 2.5 m, 0.1 m-resolution, yaw-aligned** heightmap
centered on the robot, in the sim's `world_height_map` convention (flat ground ≈ −0.20;
replace ±inf with −10). **Easiest correct source: the same RayCaster/elevation pipeline
used in WheeledLab data collection** — if your controller runs in that sim or on the robot
with that perception, just pass those maps straight in. `TerrainMap` is a fallback for a
static known heightmap, but verify its orientation/zero-offset against a few recorded
`(pose, patch)` pairs before trusting it (the model is validated; that geometric sampler is
best-effort).

## Minimal usage

```python
from dynamics_api import JumpDynamics
dyn = JumpDynamics("models/cnn_dynamics_v3/best_model.pt")

# one step
next_core = dyn.predict_step(core, elevation_patch, action)   # core (22,), patch (26,26), action (2,)

# MPPI: B candidate action sequences rolled out in parallel
#   get_patch(core_batch (B,22), t) -> (B,26,26) sampled from your terrain map
traj = dyn.rollout(core0_batch, actions_BT2, get_patch)        # (B, T+1, 22)
```

`core` is `full_sim_state[:22]`; `elevation_patch` is `full_sim_state[-676:].reshape(26,26)`.

## Caveats

- Trained on **forward-throttle** data (the car was driven hard at ramps), so it's most
  accurate at high forward throttle. Low/zero/reverse-throttle predictions are
  extrapolation (reverse ≈ coast, since the hardware can't reverse). If you need accurate
  braking/low-speed behavior, collect data with more throttle diversity and retrain.
- Speeds are capped at the real 3 m/s hardware limit; jumps top out around 10–15 cm of air.
- Closed-loop error compounds over long horizons (fine for short MPC windows; ~2 cm Z over
  ~4 s in validation). Keep planning horizons short and re-plan.

## Reproduce the validation

```bash
cd F1-Vault
.venv/bin/python learned_dynamics/dynamics_api.py          # terrain-aware vs hallucinated rollout
.venv/bin/python learned_dynamics/jump_regime_eval.py      # per-regime single-step accuracy
```
