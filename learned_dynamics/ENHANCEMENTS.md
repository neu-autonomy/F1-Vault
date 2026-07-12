# Model & controller enhancements — branch `emir_model_controller_enhancements`

Buffing up the learned-dynamics model + MPPI controller beyond the `cnn_dynamics_v3`
base. This doc is the source of truth for **what changed, why, and how to use it**, plus
the ordered roadmap of what to build next. Read this before extending any of the three.

The three implemented here are roadmap items #1–#3 from `learned_dynamics/README.md`.
Guiding principle (per Emir): **start with the simplest thing that gives a real win,
verify it, then build up.** Don't jump to the fanciest method first.

---

## #1 — Multi-step / rollout training loss  ✅ implemented

**Problem (exposure bias).** The base model was trained on *single-step* error only, but
the controller rolls it out *many* steps, feeding predictions back. A model never trained
on its own (slightly-off) inputs accumulates error fast in closed loop.

**What we do.** `train_cnn.py --horizon N` unrolls the model N steps during training:
feed the model's OWN predicted core state back in, while feeding the **recorded elevation
patch** at each step (never hallucinated — exactly how the controller uses the model via
`JumpDynamics.rollout`), and backprop through the whole unrolled window. Exposure bias is
eased with **scheduled sampling**: teacher-forcing probability decays linearly `tf_start
→ tf_end` (default 1.0 → 0.0) over training, so early epochs are near single-step and late
epochs are fully self-fed.

- `horizon=1` is the original single-step trainer, **unchanged and default** — full
  backward compatibility.
- Episodes are reconstructed (sort by `env_id, episode_id, timestep`), windows slide over
  each episode's contiguous **non-boundary** run (boundary `next_state`s are teleports —
  see CLAUDE.md). Train/val split is **by episode** so no window leaks across the split.
- Loss is the per-step core error in normalized-delta units (so `N=1` reproduces the
  original loss scale exactly) + a small (0.1×) elevation-decoder aux loss.
- The checkpoint's `config_dict`/`stats` schema is **identical** to v3 (plus a
  `train_horizon` field), so a multi-step checkpoint is a **drop-in** for `JumpDynamics`
  and every eval tool — no downstream change needed.

**Horizon choice.** Episodes are short (median ~8 steps), so `N=5` (default via CLI) keeps
most episodes usable while still strongly cutting compounding. Raise N if you recollect
longer episodes.

**Usage.**
```bash
python learned_dynamics/train_cnn.py --data_file data/raw --horizon 5 --epochs 100 \
    --save_dir models/cnn_dynamics_ms5
```

---

## #2 — Throttle/action diversity in data collection  ✅ implemented

**Problem.** The v3 data was forward-throttle-biased (`Beta(4,1.5)`, mean 0.72, min ~0.02),
so low/zero-throttle prediction — braking, speed modulation — is out-of-distribution
extrapolation.

**What we do.** `collect_dynamics_data.py --throttle_dist` selects the throttle sampler
(reverse is disabled in the env, `no_reverse=True`, so the full range is `[0,1]`):
- `mixture` (**new default**): per-env coin-flip between `Uniform(0,1)` and `Beta(4,1.5)`
  → covers the full range (fixes the low-throttle gap) **and** keeps a large block of
  high-throttle jumps so the jump regime isn't lost. Recollected: throttle mean **0.58**,
  std **0.30**, full `[0.01,0.99]` coverage, throttle→ΔX corr up ~0.09→0.14.
- `full`: `Uniform(0,1)` — maximum low-throttle coverage, but far fewer jumps.
- `highbias`: the OLD `Beta(4,1.5)` (reproduces v3's distribution exactly).

Also fixed a pre-existing crash in the collector's startup print (`args.episode_length`
is `None` when not passed → `None/float`); it now uses the effective config value.

**Usage** (needs the WL conda env + IsaacLab; runs headless):
```bash
conda run -n WL python collect_dynamics_data.py --num_envs 512 --num_episodes 40 \
    --env kicker --throttle_dist mixture --headless --output_dir F1-Vault/data/raw_mixture
python learned_dynamics/inspect_data.py --data F1-Vault/data/raw_mixture/run_*/dynamics_data_0000.h5
```
A `mixture` dataset (~354k rows) lives in `data/raw_mixture/`.

---

## #3 — Ensemble dynamics for risk-aware MPPI  ✅ implemented (deep ensemble)

**Decision — read this before "improving" it.** We chose the **simplest** uncertainty
mechanism that gives a real robustness win: a **deep ensemble** (K copies of the SAME model,
different seeds; the spread of their predictions = epistemic uncertainty). We deliberately
did **NOT** start with a Gaussian variance head (mean+logvar, NLL loss) even though it's
cheaper at inference: a single net's variance is easily overconfident off-distribution —
exactly where the controller most needs honest uncertainty — and it's fiddlier to train.
The ensemble reuses the proven deterministic trainer unchanged and is hard to get wrong.

**This is the intended starting point, not the ceiling.** Incremental path (do in order,
verify each against `rollout_eval.py`'s uncertainty/error correlation before moving on):
1. deep ensemble ← **we are here**
2. give each member a variance head (Gaussian NLL) → full PETS (epistemic + aleatoric)
3. propagate uncertainty through the rollout (trajectory sampling / TS∞) instead of
   mean-propagation, if the cost needs distributional rollouts rather than a per-step
   disagreement scalar.

**What we do.**
- `train_ensemble.py` trains K members via the normal trainer (fresh process each, spread
  round-robin across GPUs), passing `--horizon` through — so the recommended artifact is an
  **ensemble whose members are each multi-step-trained** (#1 ∘ #3).
- `dynamics_api.EnsembleJumpDynamics` (`.from_dir("models/ens_ms5")`) — `predict_step`
  (ensemble mean, drop-in), `predict_step_dist` → (mean, per-dim variance), and `rollout(…,
  return_var=True)` → (mean_traj, var_traj). Uncertainty = variance ACROSS members; rollout
  propagates the mean and records per-step disagreement.
- `dynamics_api.make_risk_aware_planner_fns(ensemble, get_patch, base_cost, risk_weight)`
  bundles it into MPPI-compatible `(rollout_fn, cost_fn)` — **no change to `mppi.py`**. The
  cost adds `risk_weight × Σ_t positional-std`, so the planner avoids action sequences the
  ensemble is unsure about (PETS-style). Start `risk_weight` small; raise until the planner
  stops picking OOD launches.

**Usage.**
```bash
python learned_dynamics/train_ensemble.py --name ens_ms5 --members 5 --horizon 5 \
    --epochs 100 --data_file data/raw --gpus 0,1
```
```python
from dynamics_api import EnsembleJumpDynamics, make_risk_aware_planner_fns
ens = EnsembleJumpDynamics.from_dir("models/ens_ms5")
rollout_fn, cost_fn = make_risk_aware_planner_fns(ens, get_patch, goal_cost, risk_weight=0.5)
u = planner.command(core_state, rollout_fn, cost_fn)   # controller/mppi.py, unchanged
```

---

## Evaluating the wins

`learned_dynamics/rollout_eval.py` measures **closed-loop N-step** error (the thing #1
targets — single-step evals miss compounding), comparing any set of checkpoints and/or
ensembles on held-out real jumps with oracle terrain, and reports an ensemble
uncertainty/error correlation (positive = uncertainty is where the error is = useful).

```bash
python learned_dynamics/rollout_eval.py --horizon 10 --data data/raw/dynamics_data_0000.h5 \
    --models v3=models/cnn_dynamics_v3/best_model.pt ms5=models/ens_ms5/member_00/best_model.pt \
             ens=models/ens_ms5
```

### Results (2026-07-12, `models/ens_ms5` = 5 multi-step members, horizon 5, 100 ep, on `data/raw`)

10-step closed-loop rollout on 150 held-out real jumps (oracle terrain), vs the v3
single-step base:

| metric (10-step rollout)  | v3 (single-step) | one multi-step member | **ens_ms5 (5-member)** |
|---|---|---|---|
| horizon-mean Z MAE        | 3.39 cm | 1.62 cm | **1.31 cm**  (2.6× better) |
| final-step (t=10) Z MAE   | 5.45 cm | 1.90 cm | **1.57 cm**  (3.5× better) |
| final-step (t=10) XY MAE  | 31.2 cm | 7.45 cm | **6.23 cm**  (5.0× better) |

- **#1 works:** v3's position error COMPOUNDS (XY drifts to 31 cm over 10 steps); the
  multi-step model holds it to ~6-7 cm. At step 1 v3 is marginally better (0.26 vs 0.45 cm
  Z, it's tuned purely for single-step) but from step ~3 on the multi-step model dominates —
  and multi-step is what the controller uses.
- **#3 works:** the 5-member ensemble beats even a single multi-step model, and its
  member-disagreement pos-std (0.9-1.5 cm) is **positively correlated (+0.30) with actual
  error** — i.e. it's more unsure where it's more wrong, the signal risk-aware MPPI needs.
  Next increment (variance heads → PETS) should raise that correlation.
