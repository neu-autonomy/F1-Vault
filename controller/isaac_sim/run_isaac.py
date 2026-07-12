"""
Milestone 2 — run the MPPI jump controller in Isaac Sim (WheeledLab kicker env).

This is the independent, model-vs-real check that controller/README.md defers to.
In Milestone 1 the learned model was BOTH planner and plant. Here the plant is the
full Isaac-Sim MuSHR car on a steep kicker ramp; the learned model is ONLY the
planner's internal dynamics. So this can finally expose model-vs-sim gaps.

Each 0.1 s control step (env decimation 10 @ sim.dt 0.01 = 10 Hz, exactly the
model's control_dt):

  1. read obs["policy"] -> core = state[:22], live elevation patch = state[-676:]
  2. splat the live patch into an online global heightmap (global_terrain.py)
  3. MPPI plans through JumpDynamics over that map, returns [throttle, steering]
  4. also predict the model's 1-step next-core from (core, live patch, action)
     -> logged against the sim's actual next state = the FAITHFULNESS metric
  5. env.step(action)  -> sim advances -> repeat

Runs headless on the lab GPU. Outputs a figure + a printed summary per episode.

Launch (from the WheeledLab conda env, which has isaacsim + isaaclab):

    /home/emir/miniforge3/envs/WL/bin/python \
        controller/isaac_sim/run_isaac.py --headless

Add --enable_cameras (and --video) to watch/record. See controller/isaac_sim/README.md.
"""

import argparse
import os
import sys

# --- make the F1-Vault controller + learned_dynamics importable -----------------
_F1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_F1_ROOT, os.path.join(_F1_ROOT, "learned_dynamics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher  # noqa: E402

# The kicker/jump env lives ONLY in the NUWheeledLab fork; the `WL` conda env has the
# BASE WheeledLab `wheeledlab_tasks` pip-installed (no kicker). Prepend NUWheeledLab's
# extension source so `wheeledlab_tasks` resolves to the fork (same trick the data-
# collection script uses). Override with --wheeledlab-src if the checkout moved.
DEFAULT_NU_SRC = "/home/emir/Lab/NUWheeledLab/source/wheeledlab_tasks"


def build_argparser():
    p = argparse.ArgumentParser(description="MPPI jump controller in Isaac Sim (kicker env)")
    p.add_argument("--wheeledlab-src", default=DEFAULT_NU_SRC,
                   help="NUWheeledLab wheeledlab_tasks extension root (has the kicker env)")
    p.add_argument("--episodes", type=int, default=3, help="how many jump attempts to run")
    p.add_argument("--max-steps", type=int, default=45, help="max control steps per episode")
    p.add_argument("--reach", type=float, default=3.0,
                   help="goal distance straight ahead through the ramp [m]")
    # MPPI knobs (mirror controller/run_controller.py defaults)
    p.add_argument("--samples", type=int, default=1024, help="MPPI rollouts/step (K)")
    p.add_argument("--horizon", type=int, default=10, help="MPPI horizon (T)")
    p.add_argument("--temp", type=float, default=1.0, help="MPPI temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=os.path.join(_F1_ROOT, "models/cnn_dynamics_v3/best_model.pt"),
                   help="dynamics checkpoint for the planner")
    p.add_argument("--episode-length", type=float, default=4.5,
                   help="sim episode length [s] (override cfg's 2.0 to see the full landing)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="print per-step action / pose / tilt and the firing termination term")
    p.add_argument("--replay-random", action="store_true",
                   help="drive the data-collection RANDOM action policy instead of MPPI "
                        "(reference: shows the ramp launches at all)")
    AppLauncher.add_app_launcher_args(p)
    return p


def main():
    args = build_argparser().parse_args()

    # Launch Isaac Sim BEFORE importing isaaclab.envs / wheeledlab (required).
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    # resolve `wheeledlab_tasks` to the NUWheeledLab fork (the kicker env), shadowing the
    # base install pip-linked in the WL env.
    if os.path.isdir(args.wheeledlab_src) and args.wheeledlab_src not in sys.path:
        sys.path.insert(0, args.wheeledlab_src)
    from wheeledlab_tasks.elevation.kicker_ramp_cfg import KickerRampDataCollectionEnvCfg

    from dynamics_api import JumpDynamics
    from controller.mppi import MPPI, MPPIConfig
    from controller.cost import GoalCost, GoalCostConfig, GROUND_AIR_REF, quat_roll_pitch
    from controller.isaac_sim.global_terrain import GlobalHeightMap, describe_patch_orientation

    # --- one env: sim is the plant, we drive env 0 with the controller -----------
    env_cfg = KickerRampDataCollectionEnvCfg()
    env_cfg.num_envs = 1
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    env_cfg.episode_length_s = args.episode_length      # longer, to watch the landing/settle
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    print(f"[setup] obs dim {obs_dim} (core 22 + last_action 2 + joints 20 + elevation 676), "
          f"control_dt {env_cfg.sim.dt * env_cfg.decimation:.2f}s")

    dyn = JumpDynamics(args.model)
    print(f"[setup] planner model on {dyn.device}, elevation {dyn.elev_size} "
          f"-> {dyn.grid}x{dyn.grid}")

    def get_state():
        return obs["policy"][0].detach().cpu().numpy().astype(np.float32)

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = get_state()
        core = state[:22].copy()
        gmap = GlobalHeightMap(core[0], core[1])

        heading = np.array([np.cos(core[9]), np.sin(core[9])], np.float32)
        goal_xy = (core[:2] + heading * args.reach).astype(np.float32)

        planner = MPPI(action_dim=2, cfg=MPPIConfig(
            horizon=args.horizon, num_samples=args.samples,
            temperature=args.temp, seed=args.seed + ep,
            # Low steering exploration: the ramp is 3 m wide and the goal is ~straight
            # ahead, so the car barely needs to steer -- and steering ON the ramp rolls it
            # (the model can't predict roll to foresee that). A straight committed run
            # clears the ramp; keep MPPI close to that.
            noise_sigma=np.array([0.30, 0.08])))
        cost = GoalCost(GoalCostConfig(goal_xy=goal_xy))

        def rollout_fn(core0, actions):
            return dyn.rollout(core0, actions, gmap.get_patch)

        print(f"\n=== episode {ep} ===")
        print(f"[start] x={core[0]:.2f} y={core[1]:.2f} yaw={core[9]:.2f} "
              f"v={np.linalg.norm(core[16:18]):.2f} m/s  goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f})")

        states, actions = [core.copy()], []
        model_pred_next, sim_next = [], []       # for the 1-step model-vs-sim metric
        ori_reported = False

        for step in range(args.max_steps):
            live_patch = state[-676:].reshape(26, 26).copy()
            gmap.update(core, live_patch)

            if not ori_reported and (live_patch > (gmap.H.min() + 0.05)).any():
                print(f"[terrain] {describe_patch_orientation(live_patch)}")
                print(f"[terrain] map round-trip err {gmap.resample_error(core, live_patch)*100:.1f} cm")
                ori_reported = True

            if args.replay_random:
                thr = float(torch.distributions.Beta(4.0, 1.5).sample())
                steer = float(torch.distributions.Beta(2.0, 2.0).sample()) * 2 - 1
                u = np.array([thr, steer], np.float32)
            else:
                u = planner.command(core, rollout_fn, cost)

            # model's 1-step prediction from the TRUE sim state + terrain (faithfulness probe)
            pred_next = dyn.predict_step(core, live_patch, u)

            act = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
            act[0] = torch.from_numpy(u)
            obs, _, terminated, truncated, _ = env.step(act)
            done = bool((terminated | truncated)[0].item())

            state = get_state()
            core = state[:22].copy()
            states.append(core.copy())
            actions.append(u.copy())
            if not done:            # skip boundary: post-reset next state is a teleport
                model_pred_next.append(pred_next[:3])
                sim_next.append(core[:3])

            if args.debug:
                import numpy as _np
                w, x, y, z = core[3], core[4], core[5], core[6]
                tilt = _np.degrees(_np.arccos(max(-1.0, min(1.0, 1 - 2*(x*x + y*y)))))
                fired = [n for n in env.termination_manager.active_terms
                         if bool(env.termination_manager.get_term(n)[0].item())]
                print(f"[dbg] step {step:2d} u=({u[0]:+.2f},{u[1]:+.2f}) "
                      f"pos=({core[0]:.2f},{core[1]:.2f},{core[2]:.3f}) tilt={tilt:.0f} "
                      f"vx_w={core[16]:.2f} fired={fired}")

            if done:
                fired = [n for n in env.termination_manager.active_terms
                         if bool(env.termination_manager.get_term(n)[0].item())]
                why = "terminated" if bool(terminated[0].item()) else "timeout"
                print(f"[done]  sim episode ended at step {step+1}: {why} {fired}")
                break
            if not args.replay_random and cost.arrived(core):
                print(f"[done]  reached goal at step {step+1} ({(step+1)*0.1:.1f} s)")
                break
        else:
            print(f"[done]  ran full {args.max_steps} steps")

        S = np.asarray(states)
        A = np.asarray(actions)
        res = summarize(S, A, goal_xy, np.asarray(model_pred_next), np.asarray(sim_next),
                        cost, quat_roll_pitch, GROUND_AIR_REF)
        res["ep"] = ep
        results.append(res)
        if not args.no_plots:
            fig_dir = os.path.join(os.path.dirname(__file__), "figures")
            os.makedirs(fig_dir, exist_ok=True)
            plot_episode(S, A, goal_xy, gmap, ep, args.replay_random, fig_dir,
                         quat_roll_pitch, GROUND_AIR_REF)

    print("\n================= OVERALL =================")
    jumped = sum(r["peak_air"] > 0.03 for r in results)
    reached = sum(r["reached"] for r in results)
    print(f"  episodes           : {len(results)}")
    print(f"  reached goal        : {reached}/{len(results)}")
    print(f"  launched (>3cm air) : {jumped}/{len(results)}")
    if results:
        print(f"  peak air (max/mean) : {max(r['peak_air'] for r in results)*100:.1f} / "
              f"{np.mean([r['peak_air'] for r in results])*100:.1f} cm")
        me = [r["model_z_mae"] for r in results if r["model_z_mae"] == r["model_z_mae"]]
        if me:
            print(f"  model 1-step Z MAE  : {np.mean(me)*100:.2f} cm  "
                  f"(model-vs-sim faithfulness; lower = model matches sim)")

    # Isaac's carb logger swallows stdout in headless runs, so ALSO persist results to a
    # file the user can actually read after the run.
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "isaac_results.json")
    try:
        import json
        payload = {"mode": "random" if args.replay_random else "mppi",
                   "horizon": args.horizon, "samples": args.samples, "reach": args.reach,
                   "episodes": results,
                   "reached_goal": int(reached), "launched": int(jumped),
                   "model_z_mae_cm": (float(np.mean(me)) * 100 if me else None)}
        with open(res_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[results] wrote {os.path.relpath(res_path, _F1_ROOT)}")
    except Exception as e:      # never let logging kill a good run
        print(f"[results] could not write results file: {e}")

    env.close()
    simulation_app.close()


def summarize(S, A, goal_xy, pred_next, sim_next, cost, quat_roll_pitch, air_ref):
    import numpy as np
    air = S[:, 2] - air_ref
    speed = np.linalg.norm(S[:, 16:18], axis=1)
    roll, pitch = quat_roll_pitch(S[:, 3:7])
    tilt = np.degrees(np.maximum(np.abs(roll), np.abs(pitch)))
    final_dist = float(np.linalg.norm(S[-1, :2] - goal_xy))
    reached = bool(cost.arrived(S[-1]))
    airborne_steps = int((air > 0.03).sum())

    if pred_next.size and sim_next.size:
        z_mae = float(np.abs(pred_next[:, 2] - sim_next[:, 2]).mean())
        xy_mae = float(np.linalg.norm(pred_next[:, :2] - sim_next[:, :2], axis=1).mean())
    else:
        z_mae = xy_mae = float("nan")

    print("--- summary ---")
    print(f"  steps               : {len(A)}  ({len(A)*0.1:.1f} s)")
    print(f"  reached goal        : {reached}  (final dist {final_dist:.2f} m)")
    print(f"  distance travelled  : {np.linalg.norm(S[-1,:2]-S[0,:2]):.2f} m")
    print(f"  peak air            : {air.max()*100:.1f} cm  (airborne {airborne_steps} steps"
          f" ~ {airborne_steps*0.1:.1f} s, launch: {air.max()>0.03})")
    print(f"  speed min/mean/max  : {speed.min():.2f}/{speed.mean():.2f}/{speed.max():.2f} m/s")
    print(f"  max tilt            : {tilt.max():.1f} deg")
    print(f"  mean throttle/steer : {A[:,0].mean():.2f} / {A[:,1].mean():.2f}")
    print(f"  MODEL 1-step Z MAE  : {z_mae*100:.2f} cm   XY MAE {xy_mae*100:.2f} cm  "
          f"(planner-model vs sim plant)")
    return {"peak_air": float(air.max()), "reached": reached, "max_tilt": float(tilt.max()),
            "model_z_mae": z_mae, "model_xy_mae": xy_mae, "steps": int(len(A)),
            "distance": float(np.linalg.norm(S[-1, :2] - S[0, :2])),
            "speed_max": float(speed.max()), "airborne_steps": airborne_steps}


def plot_episode(S, A, goal_xy, gmap, ep, replay, fig_dir, quat_roll_pitch, air_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.arange(len(S)) * 0.1
    air = S[:, 2] - air_ref
    speed = np.linalg.norm(S[:, 16:18], axis=1)
    roll, pitch = quat_roll_pitch(S[:, 3:7])
    tilt = np.degrees(np.maximum(np.abs(roll), np.abs(pitch)))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    extent = [gmap.ox, gmap.ox + gmap.H.shape[1] * gmap.res,
              gmap.oy, gmap.oy + gmap.H.shape[0] * gmap.res]
    im = a.imshow(gmap.H, origin="lower", extent=extent, aspect="auto",
                  cmap="terrain", alpha=0.9)
    fig.colorbar(im, ax=a, label="accumulated terrain height (m)", fraction=0.046)
    a.plot(S[:, 0], S[:, 1], "-", color="tab:blue", lw=2, label="driven path (sim)")
    a.scatter([S[0, 0]], [S[0, 1]], c="k", s=40, zorder=5, label="start")
    a.scatter([goal_xy[0]], [goal_xy[1]], marker="*", c="tab:red", s=200, zorder=5, label="goal")
    a.set_xlabel("world x (m)"); a.set_ylabel("world y (m)"); a.axis("equal")
    a.set_title("top-down path over accumulated terrain"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(t, air * 100, "-o", ms=3, color="tab:blue", label="robot air height (sim)")
    a.axhline(3, color="gray", ls=":", lw=1, label="3 cm launch thresh")
    a.set_xlabel("t (s)"); a.set_ylabel("height above flat (cm)")
    a.set_title("jump profile"); a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(t, speed, "-", color="tab:green", label="world speed")
    a.plot(t, tilt / 10.0, "--", color="tab:red", label="tilt/10 (deg)")
    a.axhline(3.0, color="gray", ls=":", lw=1, label="3 m/s hw cap")
    a.set_xlabel("t (s)"); a.set_ylabel("m/s  |  deg/10")
    a.set_title("speed & tilt"); a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(t[:-1], A[:, 0], "-", color="tab:purple", label="throttle")
    a.plot(t[:-1], A[:, 1], "-", color="tab:orange", label="steering")
    a.axhline(0, color="gray", ls=":", lw=1); a.set_ylim(-1.05, 1.05)
    a.set_xlabel("t (s)"); a.set_ylabel("action [-1,1]")
    who = "RANDOM policy" if replay else "MPPI"
    a.set_title(f"executed actions ({who})"); a.legend(fontsize=8)

    fig.suptitle(f"{'RANDOM' if replay else 'MPPI'} controller in Isaac Sim — kicker ramp, "
                 f"episode {ep}", fontsize=13)
    fig.tight_layout()
    tag = "random" if replay else "mppi"
    out = os.path.join(fig_dir, f"isaac_{tag}_ep{ep}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[fig]   saved {os.path.relpath(out, _F1_ROOT)}")


if __name__ == "__main__":
    main()
