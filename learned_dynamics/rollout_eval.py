"""
Closed-loop MULTI-STEP rollout evaluation -- the metric roadmap item #1 targets.

sanity_check.py / jump_regime_eval.py measure SINGLE-step accuracy. But the controller
rolls the model out many steps, feeding predictions back, so what matters is how error
COMPOUNDS over a horizon. This script measures exactly that: for each held-out real jump
episode it runs the model closed-loop from the true start state under the recorded actions,
feeding the RECORDED elevation patch each step (the oracle terrain a controller supplies --
never hallucinated), and reports per-horizon |error| vs the true trajectory.

Use it to compare a single-step checkpoint against a multi-step-trained one (and/or an
ensemble): the single-step model should track well for 1-2 steps then drift; the multi-step
model should stay closer deeper into the horizon. That gap is the exposure-bias win.

    # compare the base single-step model vs a multi-step-trained model
    python learned_dynamics/rollout_eval.py \
        --models v3=models/cnn_dynamics_v3/best_model.pt ms5=models/cnn_dynamics_ms5/best_model.pt \
        --data data/raw/dynamics_data_0000.h5 --horizon 10

    # an ensemble dir works too (label=dir); adds an uncertainty-vs-error calibration line
    python learned_dynamics/rollout_eval.py --models ens=models/ens_ms5 --data ... --horizon 10
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from dynamics_api import JumpDynamics, EnsembleJumpDynamics
from dynamics_model import replace_inf_elevation

GROUND_AIR_REF = 0.19


def load_model(path):
    """label -> JumpDynamics (a .pt file) or EnsembleJumpDynamics (a dir of member_*/)."""
    p = Path(path)
    if p.is_dir():
        return EnsembleJumpDynamics.from_dir(p), True
    return JumpDynamics(str(p)), False


def collect_episodes(data, elev_size, horizon, max_eps, air_lo, air_hi):
    """Return list of episodes (each a dict of arrays), sorted-by-timestep, jump-bearing,
    at least `horizon`+1 states long. Uses recorded patches as the terrain oracle."""
    with h5py.File(data, "r") as f:
        S = replace_inf_elevation(f["states"][:].astype(np.float32), elev_size)
        A = f["actions"][:].astype(np.float32)
        env, ep, ts = f["env_ids"][:], f["episode_ids"][:], f["timesteps"][:]
        done = f["terminated"][:] | f["truncated"][:]
    key = env.astype(np.int64) * 1_000_000 + ep.astype(np.int64)
    air = S[:, 2] - GROUND_AIR_REF
    eps = []
    for k in np.unique(key):
        r = np.where(key == k)[0]
        if len(r) < horizon + 1:
            continue
        o = r[np.argsort(ts[r])]
        d = done[o]
        e = int(np.argmax(d)) if d.any() else len(o)
        o = o[:e]
        if len(o) < horizon + 1 or not (air_lo <= air[o].max() <= air_hi):
            continue
        eps.append({
            "core0": S[o[0], :22].copy(),
            "actions": A[o[:horizon]].copy(),
            "patches": S[o[:horizon + 1], -elev_size:].reshape(-1, 26, 26).copy(),
            "true_core": S[o[:horizon + 1], :22].copy(),
        })
        if len(eps) >= max_eps:
            break
    if not eps:
        raise ValueError(f"no jump episodes (>= {horizon+1} steps, air in [{air_lo},{air_hi}]) "
                         f"in {data}")
    return eps


def main():
    ap = argparse.ArgumentParser(description="Closed-loop multi-step rollout eval")
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path entries; path is a .pt checkpoint or an ensemble dir")
    ap.add_argument("--data", default="data/raw/dynamics_data_0000.h5")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_eps", type=int, default=200)
    ap.add_argument("--air_lo", type=float, default=0.08)
    ap.add_argument("--air_hi", type=float, default=0.40)
    args = ap.parse_args()

    models = {}
    is_ens = {}
    for entry in args.models:
        if "=" not in entry:
            raise SystemExit(f"--models entries must be label=path, got {entry!r}")
        label, path = entry.split("=", 1)
        m, ens = load_model(path)
        models[label], is_ens[label] = m, ens

    elev_size = next(iter(models.values())).elev_size
    H = args.horizon
    eps = collect_episodes(args.data, elev_size, H, args.max_eps, args.air_lo, args.air_hi)
    print(f"\nEvaluating on {len(eps)} held-out jump episodes, horizon {H} "
          f"(recorded/oracle terrain).  Peak air range [{args.air_lo},{args.air_hi}] m.\n")

    for label, dyn in models.items():
        # (E, H+1, 3) abs errors on x,y,z; and hold-initial naive baseline
        z_err = np.zeros((len(eps), H + 1)); xy_err = np.zeros((len(eps), H + 1))
        z_naive = np.zeros((len(eps), H + 1))
        unc = np.zeros((len(eps), H + 1))     # ensemble positional std per step
        for i, e in enumerate(eps):
            patches = e["patches"]
            get_patch = lambda core, t: patches[min(t, len(patches) - 1)][None]
            if is_ens[label]:
                traj, var = dyn.rollout(e["core0"], e["actions"], get_patch, return_var=True)
                unc[i] = np.sqrt(np.clip(var[:, :3].sum(axis=1), 0, None))
            else:
                traj = dyn.rollout(e["core0"], e["actions"], get_patch)
            tc = e["true_core"]
            z_err[i] = np.abs(traj[:, 2] - tc[:, 2])
            xy_err[i] = np.linalg.norm(traj[:, :2] - tc[:, :2], axis=1)
            z_naive[i] = np.abs(e["core0"][2] - tc[:, 2])

        print(f"=== {label} ===")
        print(f"  step   Z MAE     XY MAE   | Z naive-hold" +
              ("     ens pos-std" if is_ens[label] else ""))
        for t in range(1, H + 1):
            line = (f"  {t:>3}   {z_err[:,t].mean()*100:6.2f}cm  {xy_err[:,t].mean()*100:6.2f}cm  | "
                    f"{z_naive[:,t].mean()*100:6.2f}cm")
            if is_ens[label]:
                line += f"     {unc[:,t].mean()*100:6.2f}cm"
            print(line)
        print(f"  --> horizon-mean Z MAE {z_err[:,1:].mean()*100:.2f}cm  "
              f"final-step XY {xy_err[:,-1].mean()*100:.2f}cm")
        if is_ens[label]:
            # calibration: does higher ensemble uncertainty track higher actual error?
            u = unc[:, 1:].ravel(); a = (z_err[:, 1:] + xy_err[:, 1:]).ravel()
            if u.std() > 1e-9:
                c = np.corrcoef(u, a)[0, 1]
                print(f"  --> uncertainty/error corr: {c:+.3f}  "
                      f"(positive = ensemble is unsure where it's actually wrong = useful)")
        print()


if __name__ == "__main__":
    main()
