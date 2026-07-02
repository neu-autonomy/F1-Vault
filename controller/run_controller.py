"""
Closed-loop MPPI demo: drive the MuSHR car over a ramp to a goal, using the
learned jump-dynamics model both as the planner's internal model AND as the
plant.

    python -m controller.run_controller                 # ramp demo, default knobs
    python -m controller.run_controller --terrain flat   # flat-ground sanity
    python -m controller.run_controller --samples 2048 --horizon 15 --goal-x 8

This is Milestone 1 from the plan: it validates the *controller* against the
learned model over known terrain, fully headless (matplotlib PNGs), no Isaac Sim.
It is NOT an independent physics check — the model is the plant here, so it can't
expose model-vs-real gaps. That comes later in WheeledLab/Isaac Sim.

Outputs figures to controller/figures/ and prints a run summary.
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "learned_dynamics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dynamics_api import JumpDynamics  # noqa: E402

from controller.cost import (  # noqa: E402
    GROUND_AIR_REF, GoalCost, GoalCostConfig, quat_roll_pitch,
)
from controller.mppi import MPPI, MPPIConfig  # noqa: E402
from controller.episodes import load_jump_episode  # noqa: E402
from controller.terrain import (  # noqa: E402
    FLAT_Z, RampSpec, RecordedTerrain, build_flat_terrain, build_ramp_terrain,
    make_get_patch,
)

# --- data-derived constants for the flat approach (see controller/README.md) ---
ROOT_Z_FLAT = 0.19       # world root z on flat ground (data median)
APPROACH_SPEED = 2.8     # m/s forward speed to start the car at (data run-up ~2.5-2.8)
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def make_initial_core(x=0.0, y=0.0, yaw=0.0, speed=APPROACH_SPEED, z=ROOT_Z_FLAT):
    """Build a plausible 22-dim core state: on flat ground, driving forward at `speed`.

    Layout matches dynamics_api.py: pos(3) quat(4) euler(3) body-vel(3) body-ang(3)
    world-vel(3) world-ang(3). Zero tilt / spin; velocity aligned with heading.
    """
    core = np.zeros(22, np.float32)
    core[0:3] = [x, y, z]
    core[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]   # quat (w,x,y,z)
    core[7:10] = [0.0, 0.0, yaw % (2 * np.pi)]                  # euler, yaw in [0,2pi)
    core[10:13] = [speed, 0.0, 0.0]                            # body-frame lin vel
    core[16:19] = [speed * np.cos(yaw), speed * np.sin(yaw), 0.0]  # world lin vel
    return core


def run(args):
    dyn = JumpDynamics()
    print(f"[setup] model on {dyn.device}, elev {dyn.elev_size} -> {dyn.grid}x{dyn.grid}")

    if args.terrain == "recorded":
        # faithful terrain: replay real recorded elevation patches (handoff's blessed
        # source). Start from the episode's spawn; goal = its final pose, past the ramp.
        ep = load_jump_episode(args.data, index=args.episode)
        tm = RecordedTerrain(ep["poses"], ep["yaws"], ep["patches"])
        core = ep["core0"].copy()
        # goal STRAIGHT AHEAD through the ramp: the launch is a knife-edge event that
        # only fires on a near-straight hit, so aiming down the heading (not at the
        # corridor's slightly-offset endpoint) lets the car take the ramp cleanly.
        # The landing zone past the ramp is flat, so straying there is harmless.
        heading = np.array([np.cos(core[9]), np.sin(core[9])], np.float32)
        goal_xy = (core[:2] + heading * args.reach).astype(np.float32)
        meta = {"kind": "recorded", "poses": ep["poses"],
                "centers": ep["patches"][:, 13, 13]}
        replay_actions = ep["actions"] if args.replay else None
        print(f"[terrain] recorded jump episode {args.episode}: peak air "
              f"{ep['peak_air']*100:.0f} cm, corridor span "
              f"{np.linalg.norm(ep['poses'][-1]-ep['poses'][0]):.1f} m"
              + ("  [REPLAY: driving recorded actions]" if args.replay else ""))
    elif args.terrain == "ramp":
        ramp = RampSpec(x_start=args.ramp_x, rise=args.ramp_rise)
        tm, meta = build_ramp_terrain(ramp)
        meta["kind"] = "ramp"
        core = make_initial_core(speed=args.speed)
        goal_xy = np.array([args.goal_x, args.goal_y], np.float32)
        replay_actions = None
        print(f"[terrain] synthetic kicker ramp: rise {ramp.rise*100:.0f} cm over "
              f"{ramp.up_len:.1f} m, lip at x={meta['lip_x']:.2f} m")
    else:
        tm, meta = build_flat_terrain()
        meta["kind"] = "flat"
        core = make_initial_core(speed=args.speed)
        goal_xy = np.array([args.goal_x, args.goal_y], np.float32)
        replay_actions = None
        print("[terrain] flat ground")
    get_patch = make_get_patch(tm)

    cost = GoalCost(GoalCostConfig(goal_xy=goal_xy))
    planner = MPPI(action_dim=2, cfg=MPPIConfig(
        horizon=args.horizon, num_samples=args.samples,
        temperature=args.temp, seed=args.seed))

    def rollout_fn(core0, actions):                 # (K,22),(K,T,2) -> (K,T+1,22)
        return dyn.rollout(core0, actions, get_patch)

    print(f"[start]  x={core[0]:.2f} y={core[1]:.2f} yaw={core[9]:.2f} "
          f"v={np.linalg.norm(core[16:18]):.2f} m/s   goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})")

    states, actions = [core.copy()], []
    n_replay = len(replay_actions) if replay_actions is not None else 0
    for step in range(args.steps):
        if replay_actions is not None:                  # reference: recorded actions
            u = replay_actions[min(step, n_replay - 1)]
        else:                                           # the MPPI controller
            u = planner.command(core, rollout_fn, cost)
        patch = tm.get_patch(core[None], step)[0]
        core = dyn.predict_step(core, patch, u)
        states.append(core.copy())
        actions.append(u.copy())
        if replay_actions is not None and step >= n_replay - 1:
            print(f"[done]   replayed {n_replay} recorded actions")
            break
        if replay_actions is None and cost.arrived(core):
            print(f"[done]   reached goal at step {step+1} "
                  f"({(step+1)*0.1:.1f} s)")
            break
    else:
        print(f"[done]   ran full {args.steps} steps without reaching goal")

    S = np.asarray(states)          # (n+1,22)
    A = np.asarray(actions)         # (n,2)

    if args.terrain == "recorded":
        # how far the ACTUAL driven path strayed from the recorded corridor (the
        # planner's hypothetical rollouts roam further; only the driven path matters)
        stray = max(float(np.linalg.norm(meta["poses"] - s[:2], axis=1).min())
                    for s in S)
        if stray > 0.6:
            print(f"[warn]   driven path strayed up to {stray:.2f} m from the recorded "
                  f"corridor — replayed terrain is less trustworthy that far out")
    summarize(S, A, goal_xy, tm, cost)
    if not args.no_plots:
        os.makedirs(FIG_DIR, exist_ok=True)
        plot_run(S, A, goal_xy, tm, meta, replay=(replay_actions is not None))


def summarize(S, A, goal_xy, tm, cost):
    air = S[:, 2] - GROUND_AIR_REF
    speed = np.linalg.norm(S[:, 16:18], axis=1)
    roll, pitch = quat_roll_pitch(S[:, 3:7])
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    final_dist = float(np.linalg.norm(S[-1, :2] - goal_xy))
    print("\n=== run summary ===")
    print(f"  steps executed     : {len(A)}  ({len(A)*0.1:.1f} s)")
    print(f"  reached goal        : {cost.arrived(S[-1])}  (final dist {final_dist:.2f} m)")
    print(f"  distance travelled  : {np.linalg.norm(S[-1,:2]-S[0,:2]):.2f} m")
    print(f"  peak air height     : {air.max()*100:.1f} cm  (launch detected: {air.max()>0.03})")
    print(f"  speed  min/mean/max : {speed.min():.2f}/{speed.mean():.2f}/{speed.max():.2f} m/s")
    print(f"  max tilt (roll/pit) : {np.degrees(tilt.max()):.1f} deg")
    print(f"  mean throttle/steer : {A[:,0].mean():.2f} / {A[:,1].mean():.2f}")


def plot_run(S, A, goal_xy, tm, meta, replay=False):
    terrain_kind = meta["kind"]
    t = np.arange(len(S)) * 0.1
    air = S[:, 2] - GROUND_AIR_REF
    speed = np.linalg.norm(S[:, 16:18], axis=1)
    # terrain rise directly under the robot (center cell of the yaw-aligned patch)
    ter_rise = np.array([tm.get_patch(s[None])[0, 13, 13] - FLAT_Z for s in S])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) top-down path over the terrain
    a = ax[0, 0]
    if "H" in meta:                                    # synthetic ramp / flat
        xr, yr = meta["x_range"], meta["y_range"]
        im = a.imshow(meta["H"], origin="lower",
                      extent=[xr[0], xr[1], yr[0], yr[1]],
                      aspect="auto", cmap="terrain", alpha=0.9)
        fig.colorbar(im, ax=a, label="terrain height (m)", fraction=0.046)
        if terrain_kind == "ramp":
            a.axvline(meta["lip_x"], color="w", ls=":", lw=1.5, label="ramp lip")
    else:                                              # recorded corridor
        p, ctr = meta["poses"], meta["centers"]
        sc = a.scatter(p[:, 0], p[:, 1], c=ctr, cmap="terrain", s=45,
                       label="recorded corridor")
        fig.colorbar(sc, ax=a, label="recorded elev (m)", fraction=0.046)
    a.plot(S[:, 0], S[:, 1], "-", color="tab:blue", lw=2, label="driven path")
    a.scatter([S[0, 0]], [S[0, 1]], c="k", s=40, zorder=5, label="start")
    a.scatter([goal_xy[0]], [goal_xy[1]], marker="*", c="tab:red", s=200,
              zorder=5, label="goal")
    a.set_xlabel("world x (m)"); a.set_ylabel("world y (m)"); a.axis("equal")
    a.set_title("top-down path over terrain"); a.legend(loc="best", fontsize=8)

    # (0,1) height vs time: robot air height and terrain rise under it
    a = ax[0, 1]
    a.plot(t, air * 100, "-o", ms=3, color="tab:blue", label="robot air height")
    a.plot(t, ter_rise * 100, "--", color="tab:brown", label="terrain under robot")
    a.axhline(0, color="gray", ls=":", lw=1)
    a.set_xlabel("t (s)"); a.set_ylabel("height above flat (cm)")
    a.set_title("jump profile"); a.legend(fontsize=8)

    # (1,0) speed vs time
    a = ax[1, 0]
    a.plot(t, speed, "-", color="tab:green", label="world speed")
    a.axhline(3.0, color="gray", ls=":", lw=1, label="3 m/s hw cap")
    a.set_xlabel("t (s)"); a.set_ylabel("speed (m/s)")
    a.set_title("speed"); a.legend(fontsize=8)

    # (1,1) executed actions vs time
    a = ax[1, 1]
    a.plot(t[:-1], A[:, 0], "-", color="tab:purple", label="throttle")
    a.plot(t[:-1], A[:, 1], "-", color="tab:orange", label="steering")
    a.axhline(0, color="gray", ls=":", lw=1)
    a.set_ylim(-1.05, 1.05)
    a.set_xlabel("t (s)"); a.set_ylabel("action [-1,1]")
    a.set_title("executed actions (MPPI)"); a.legend(fontsize=8)

    who = "recorded-action replay (reference)" if replay else "MPPI closed-loop"
    fig.suptitle(f"{who} over learned dynamics — {terrain_kind} terrain", fontsize=13)
    fig.tight_layout()
    tag = f"{terrain_kind}_replay" if replay else terrain_kind
    out = os.path.join(FIG_DIR, f"mppi_run_{tag}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nsaved figure: {os.path.relpath(out, _ROOT)}")


def build_argparser():
    p = argparse.ArgumentParser(description="MPPI closed-loop controller demo")
    p.add_argument("--terrain", choices=["recorded", "ramp", "flat"], default="recorded",
                   help="recorded=faithful real patches (launches); "
                        "ramp/flat=synthetic (approximate terrain convention)")
    p.add_argument("--data", default="data/raw/dynamics_data_0000.h5",
                   help="h5 file for recorded terrain")
    p.add_argument("--episode", type=int, default=0, help="which jump episode to use")
    p.add_argument("--reach", type=float, default=3.2,
                   help="recorded mode: goal distance straight ahead through the ramp (m)")
    p.add_argument("--replay", action="store_true",
                   help="recorded mode: drive the RECORDED actions (reference launch), "
                        "not the MPPI controller")
    p.add_argument("--steps", type=int, default=40, help="max control steps")
    p.add_argument("--samples", type=int, default=1024, help="MPPI rollouts/step (K)")
    p.add_argument("--horizon", type=int, default=12, help="MPPI horizon (T)")
    p.add_argument("--temp", type=float, default=1.0, help="MPPI temperature (lambda)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=APPROACH_SPEED, help="initial speed m/s")
    p.add_argument("--goal-x", type=float, default=3.5, help="goal x, past the landing")
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--ramp-x", type=float, default=1.2, help="ramp start world x (short approach)")
    p.add_argument("--ramp-rise", type=float, default=0.28, help="ramp rise (m)")
    p.add_argument("--no-plots", action="store_true")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
