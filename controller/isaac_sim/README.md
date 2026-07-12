# controller/isaac_sim/ — Milestone 2: the controller in Isaac Sim

The independent, model-vs-real check that [`controller/README.md`](../README.md) defers
to. Milestone 1 validated the MPPI controller with the learned model as **both** planner
and plant (it can't expose model error). Here the **plant is the full Isaac-Sim MuSHR car**
on a steep kicker ramp (WheeledLab's `KickerRampDataCollectionEnvCfg`); the learned model
is **only** the planner's internal dynamics. This is the trip-to-the-lab step.

## Runtime — use the WheeledLab conda env, NOT F1-Vault's `.venv`

Isaac Sim only imports under the `WL` conda env (it has `isaacsim` 4.5 + `isaaclab` +
`wheeledlab*`, plus torch/h5py/matplotlib). F1-Vault's own `.venv` is CPU/torch-only and
**cannot** launch the sim. The harness adds the F1-Vault controller + `learned_dynamics` to
`sys.path` itself, so running under `WL` python is all you need.

**The kicker env lives in the NUWheeledLab fork**, not the base WheeledLab that the `WL`
env pip-installs. The harness prepends `/home/emir/Lab/NUWheeledLab/source/wheeledlab_tasks`
to `sys.path` so `wheeledlab_tasks` resolves to the fork (the kicker cfg). If that checkout
moved, pass `--wheeledlab-src <path-to>/NUWheeledLab/source/wheeledlab_tasks`.

## Run it

```bash
cd /home/emir/Lab/F1-Vault

# headless, 3 jump attempts, MPPI driving (writes figures + prints per-episode summary)
/home/emir/miniforge3/envs/WL/bin/python controller/isaac_sim/run_isaac.py --headless

# watch it (spawns the Isaac viewport)     # or add --video to record to disk
/home/emir/miniforge3/envs/WL/bin/python controller/isaac_sim/run_isaac.py --enable_cameras

# reference: drive the data-collection RANDOM policy instead of MPPI
#   (sanity that the ramp launches the car at all, independent of the controller)
/home/emir/miniforge3/envs/WL/bin/python controller/isaac_sim/run_isaac.py --headless --replay-random
```

Useful knobs: `--episodes N`, `--horizon 10 --samples 1024 --temp 1.0`, `--reach 3.0`
(goal distance ahead through the ramp), `--episode-length 4.5` (sim seconds), `--seed`.

## What it does each 0.1 s control step

1. reads `obs["policy"]` → `core = state[:22]`, live 26×26 elevation = `state[-676:]`
2. splats the live patch into an **online global heightmap** ([`global_terrain.py`](global_terrain.py))
   so the planner has terrain at look-ahead poses (faithful — accumulated from the sim's
   own patches, never an analytic ramp)
3. **MPPI** plans through `JumpDynamics` over that map → `[throttle, steering]`
4. records the model's **1-step prediction vs. the sim's actual next state** — the
   model-vs-real faithfulness number this milestone exists to produce
5. `env.step(action)` advances the real sim; repeat

## Outputs

- `controller/isaac_sim/figures/isaac_mppi_ep<N>.png` — path over accumulated terrain,
  jump profile (air height), speed & tilt, executed actions.
- Per-episode + overall console summary: reached-goal, peak air, airborne time, max tilt,
  and **model 1-step Z / XY MAE** (planner-model vs sim plant).

## Reading the result

- **Controller works** ⇢ the car reaches the goal, stays upright (tilt well under the 80°
  rollover), and engages the ramp. Peak air on a maximal launch tops out ~10–15 cm (the
  regime the model was trained in); MPPI takes the ramp more conservatively than the raw
  random policy because reaching a goal past the ramp rewards a controlled traversal, not a
  maximal jump (same finding as Milestone 1).
- **Model is faithful** ⇢ low 1-step Z MAE (single-step val was 0.6–1.2 cm; a few cm in
  closed loop is expected). A large MAE that spikes at launch/landing is the real
  model-vs-sim gap to chase — the whole reason for running in-sim.

## Gotchas

- The kicker angle is randomized per reset (30–42°); some episodes are steeper and may flip
  or high-center — that's the env's terminations doing their job, not a controller bug. Run
  a few `--episodes`.
- If the planner behaves as if the ramp is in the wrong place, suspect the patch-axis
  convention first: the harness prints where the elevated terrain sits in the first live
  patch and the map round-trip error. **NOTE (2026-07-12): the "expected AHEAD near spawn"
  wording is misleading** — at spawn the ramp foot sits at the FAR edge of the ±1.25 m
  patch (only a low sliver is visible, mixed with flat-ground noise), so the diagnostic
  reads "under"/scattered even on the RECORDED training data. That is expected, not a bug.
  The large round-trip number is likewise dominated by ray-miss cells (clamped to +2 m)
  smearing under bilinear interp, not an orientation error.

## Why the car "didn't jump" — debugging log (2026-07-12)

Symptom: in Isaac the car tipped over the ramp / drove on without a real jump.
Root cause was the **controller cost, not the ramp**:

- **THE fix — decouple roll from pitch in `controller/cost.py`.** `GoalCost` penalized
  *pitch* like *roll* past ~20°, but climbing a 30–42° ramp IS 30–42° of nose-up pitch, so
  MPPI scored every launch as a near-rollover and refused to commit → it stalled at the
  ramp foot (original results: travelled only 0.06–0.45 m, stuck-terminated). Now roll stays
  strict (real rollover), pitch is tolerated to ~51°, and `air_max` 0.30→0.40 m. After the
  fix the car **commits, climbs the ramp (tilt 36–45°), and traverses the full ~3 m**, with
  small jumps (0–4 cm, occasional true launch).
- **Physics ceiling.** At the 3 m/s hardware cap the ramp only yields ~10–15 cm of air
  max (energy: `v_top² = v0² − 2g·height`). "More speed" is NOT available — faster spawns
  were the original bug that made jumps unreproducible by any legal action. Lowering the
  ramp angle BACKFIRES (a 14–24° ramp gives 0 cm air — the car just rolls over); the proven
  30–42° geometry is kept. So expect a *modest* hop, not big air.
- **Minor: terrain splat holes** (`global_terrain.py`). The 0.1 m patch was splatted into a
  0.05 m map with nearest-neighbour rounding, leaving holes that rotated queries interpolated
  against flat filler (~20 cm round-trip error at yaw≠0). Now each patch cell fills its
  footprint (round-trip 20→2 cm off-axis). Real improvement, but was NOT the main blocker.

To see it: `python controller/isaac_sim/run_isaac.py --episodes 8` (figures per episode),
or `--replay-random` for the committed-policy reference (which launches ~6 cm).
