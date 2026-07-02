# controller/ — MPPI controller over the learned jump dynamics

An **MPPI** (Model Predictive Path Integral) controller that plans through the
learned MuSHR jump-dynamics model (`learned_dynamics/`, see
[`CONTROLLER_HANDOFF.md`](../learned_dynamics/CONTROLLER_HANDOFF.md)). Each control
step it samples many candidate action sequences, rolls them all through
`JumpDynamics.rollout` in parallel on the GPU, scores them, and executes the
softmax-weighted best action — receding-horizon, re-planning every step.

This is **Milestone 1**: the controller is developed and validated entirely
against the learned model as the plant, over known terrain, **fully headless on
the lab GPU over SSH** (matplotlib PNGs, no Isaac Sim). It is *not* an independent
physics check — the model is both planner and plant here, so it cannot expose
model-vs-real gaps. That is the later Isaac Sim / WheeledLab step.

## Layout

| file | what |
|---|---|
| [`mppi.py`](mppi.py) | generic, model-agnostic MPPI core (`MPPI`, `MPPIConfig`) |
| [`cost.py`](cost.py) | jump-task cost (`GoalCost`): reach goal, stay upright, stay in-distribution |
| [`terrain.py`](terrain.py) | terrain sources: synthetic ramp (`build_ramp_terrain`) + faithful `RecordedTerrain` |
| [`episodes.py`](episodes.py) | load real jump episodes from the collected `.h5` data |
| [`run_controller.py`](run_controller.py) | closed-loop harness + figures (the entry point) |

`mppi.py` and `cost.py` know nothing about the dynamics model or terrain — MPPI
takes a `rollout_fn(core0, actions)->traj` and a `cost_fn(traj, actions)->cost`,
so the planner is reusable and unit-testable against any dynamics.

## Quickstart

Run from the repo root with the project venv:

```bash
# faithful terrain: replay REAL recorded elevation patches, MPPI drives to a goal
.venv/bin/python -m controller.run_controller --terrain recorded

# reference: drive the RECORDED actions instead of the controller (shows the real launch)
.venv/bin/python -m controller.run_controller --terrain recorded --replay

# synthetic kicker ramp / flat-ground sanity
.venv/bin/python -m controller.run_controller --terrain ramp
.venv/bin/python -m controller.run_controller --terrain flat

# knobs: --samples 2048 --horizon 12 --temp 0.5 --episode 1 --reach 3.5 --seed 0
```

Figures land in `controller/figures/mppi_run_<mode>.png` (top-down path over
terrain, jump profile, speed, executed actions).

## Terrain modes

The model does **not** know the terrain — you feed it the local 26×26 elevation
patch every step (the one thing the handoff says you must get right). Three
sources:

- **`recorded`** *(default)* — `RecordedTerrain` replays **real recorded patches**
  from a data episode via nearest-pose lookup. This is the handoff's blessed
  source (same RayCaster pipeline as data collection), so the model launches
  exactly as it did in the data. The selector (`episodes.load_jump_episode`)
  prefers a long, **straight** corridor so "drive to the goal" keeps the car on
  the recorded terrain.
- **`ramp`** — a synthetic kicker heightmap (`TerrainMap`) sized to the data
  (~28 cm rise over ~0.7 m). Convenient and fully controllable, but the
  handoff warns `TerrainMap`'s orientation/zero-offset is *best-effort* — it
  under-drives the model's jump vs real patches, so treat it as geometric sanity,
  not a faithful jump test.
- **`flat`** — featureless ground; a stability/goal-tracking baseline.

## Results (learned model as plant)

| mode | reaches goal | peak air | max tilt | note |
|---|---|---|---|---|
| flat | ✅ (1.7 s) | — | 2° | clean straight drive to goal |
| ramp (synthetic) | ✅ (1.7 s) | ~0 cm | 10° | climbs the ramp; synthetic terrain under-drives launch |
| recorded (MPPI) | ✅ (1.9 s) | ~2 cm | ~40° | drives straight through the ramp to the goal, engages the kicker |
| recorded (`--replay`) | — | **9.3 cm** | 65° | reference: the *recorded actions* launch the car (Z tracks to ~2 cm) |

The **replay** row is the key context: fed the exact recorded action sequence,
the model reproduces the real ~9 cm launch (Z-MAE ~2 cm vs the recording) — the
model + terrain pipeline is faithful. The **MPPI** controller reaches the goal and
engages the ramp but takes it conservatively (~2 cm air): the cost-optimal way to
reach a goal past a ramp is a controlled traversal, not a maximal jump, and the
ballistic launch is a knife-edge event that closed-loop control does not exactly
reproduce (see below).

## Findings that shaped the design (read before extending)

These came out of building this and are worth knowing:

1. **Use the quaternion, not the euler channel, for orientation.** The model does
   not predict euler `[7:10]` coherently (it swings 100–170° on straight flat
   driving). The quaternion `[3:7]` stays unit-norm and upright. `cost.py` reads
   roll/pitch from the quaternion (`quat_roll_pitch`).
2. **The z-sink on approach is real, not a bug.** Root z drops from the 0.19 spawn
   value to ~0 within ~2 steps — the same settling dip appears in the recorded
   data. So `air = z − 0.19` reads strongly negative during normal driving; a jump
   is a *rise above* that baseline. Don't "fix" it.
3. **The model is short-horizon.** There are ~zero long flat-driving episodes in
   the data (the car is always launched at a ramp in short episodes), and open-loop
   XY drifts ~1–3 m over ~4 s. So keep the MPPI horizon short (default 10 = 1.0 s)
   and re-plan — a long plan optimises against a drifting prediction.
4. **Forward-throttle regime only.** Reverse/low throttle is extrapolation
   (hardware can't reverse). MPPI clips throttle to `[0, 1]` and penalises reverse.
5. **A launch only fires on a near-straight, well-timed hit.** Steering wiggle at
   the ramp, or arriving at a different phase/speed, kills it — which is why
   closed-loop MPPI (which must also steer toward the goal and re-plans every step)
   under-launches relative to the pristine recorded action sequence.

## Limitations & next steps

- **Independent validation (Isaac Sim / WheeledLab).** The model is the plant
  here, so these numbers can't reveal model-vs-real error. Running the finished
  controller in the full sim is the real check — the trip-to-the-lab step.
- **Faithful 2-D terrain.** `RecordedTerrain` is a 1-D corridor oracle; it degrades
  when the controller strays laterally (the harness warns when the driven path
  leaves the corridor). A rasterised global heightmap from the RayCaster (or live
  perception) would support free planning and make synthetic ramps trustworthy.
- **Closing the jump gap.** If the goal is to *maximise* jumps, add an explicit
  clear-the-ramp reward and/or a terrain source that lets the controller commit to
  a straight, fast launch — bounded by the model's validity (jumps top out
  ~10–15 cm; OOD actions can fly it meters up, so keep the air ceiling in the cost).
- **Braking/low-speed.** The data is forward-throttle-biased; accurate
  braking/low-speed control needs more throttle diversity in the data + a retrain.
