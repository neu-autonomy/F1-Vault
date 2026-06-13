"""
Jump-regime evaluation: is the JUMP specifically learnable, or is the good
aggregate MAE just carried by easy flat-ground driving?

The aggregate eval in train_cnn.py mixes all transitions. This splits the SAME
boundary-filtered validation rows by regime (flat / on-ramp / airborne) and by
vertical-motion magnitude, and reports per-axis single-step MAE vs the predict-zero
baseline IN EACH BUCKET. If Z error stays well below naive on the airborne/ramp rows,
the model has learned the jump dynamics; if it collapses to ~naive there while the
aggregate looks fine, it hasn't -- the flat rows are hiding it.

Uses the same data file, boundary filter, seed (42) and split as train_cnn.py, so
these are genuinely held-out rows the model never trained on.

Run from the F1-Vault repo root:
    .venv/bin/python learned_dynamics/jump_regime_eval.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dynamics_model import load_model, normalization_mode, replace_inf_elevation, grid_from_size

# core-state column indices (core_dim = 22)
I_POS = slice(0, 3)        # x, y, z (world)
I_YAW = 9                  # world_euler yaw
I_BVZ = 12                 # base_lin_vel z (body vertical)
I_WVX = 16                 # root_lin_vel_w x (world forward)
I_WVZ = 18                 # root_lin_vel_w z (world vertical) -- gravity/launch/landing live here
I_WYAWRATE = 21            # root_ang_vel_w z (world yaw rate)

DATA_FILE = "data/raw"                              # file OR dir of batches (match train_cnn)
DATA_ONE_BATCH = "data/raw/dynamics_data_0000.h5"   # single batch for episode-tracking (clean keys)
CKPT = "models/cnn_dynamics_v3/best_model.pt"
VAL_SPLIT = 0.2
SEED = 42
GROUND_Z = 0.19


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, stats, core_dim = load_model(CKPT, device)
    elev_size = config["elevation_map_size"]
    grid = grid_from_size(elev_size)
    input_norm, output_denorm = normalization_mode(config)
    core_t_mean = np.asarray(stats["core_target_mean"], dtype=np.float32)
    core_t_std = np.asarray(stats["core_target_std"], dtype=np.float32)
    print(f"model: core_dim={core_dim} elev={elev_size} ({grid}x{grid}) "
          f"input_norm={input_norm} output_denorm={output_denorm}")

    # ---- Load + boundary filter + reproduce the exact train/val split ----
    # Load ALL batches in the same sorted order train_cnn uses, so the seeded split
    # reproduces the identical held-out val rows.
    dp = Path(DATA_FILE)
    files = sorted(dp.glob("*.h5")) if dp.is_dir() else [dp]
    Sx, Ax, Nx, Tx, Rx = [], [], [], [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            Sx.append(f["states"][:]); Ax.append(f["actions"][:]); Nx.append(f["next_states"][:])
            Tx.append(f["terminated"][:]); Rx.append(f["truncated"][:])
    states = np.concatenate(Sx); actions = np.concatenate(Ax); next_states = np.concatenate(Nx)
    terminated = np.concatenate(Tx); truncated = np.concatenate(Rx)
    del Sx, Ax, Nx, Tx, Rx

    valid = ~(terminated | truncated)
    states, actions, next_states = states[valid], actions[valid], next_states[valid]
    states = replace_inf_elevation(states.astype(np.float32), elev_size)
    next_states = replace_inf_elevation(next_states.astype(np.float32), elev_size)

    n = len(states)
    np.random.seed(SEED)                       # same seeding as train_cnn.py main()
    idx = np.random.permutation(n)
    n_train = int(n * (1 - VAL_SPLIT))
    val_idx = idx[n_train:]
    print(f"val rows: {len(val_idx):,} (held out, never trained on)\n")

    s = states[val_idx]
    ns = next_states[val_idx]
    a = actions[val_idx].astype(np.float32)

    # ---- Single-step prediction over the val set (batched) ----
    core = s[:, :core_dim]
    elev = s[:, -elev_size:].reshape(-1, grid, grid)
    true_dcore = (ns[:, :core_dim] - core)        # ground-truth delta

    preds = np.zeros_like(core)
    model.eval()
    with torch.no_grad():
        B = 4096
        for i in range(0, len(s), B):
            ct = torch.from_numpy(core[i:i+B]).to(device)
            et = torch.from_numpy(elev[i:i+B]).unsqueeze(1).to(device)
            at = torch.from_numpy(a[i:i+B]).to(device)
            pc, _ = model(ct, et, at)
            pc = pc.cpu().numpy()
            if output_denorm:
                pc = pc * core_t_std + core_t_mean
            preds[i:i+B] = pc

    # predicted vs true POSITION delta (cols 0:3 = x,y,z)
    err = np.abs(preds[:, :3] - true_dcore[:, :3])     # (N,3) model |error|
    naive = np.abs(true_dcore[:, :3])                  # predict-zero error == |true delta|

    # ---- Regime classification from the CURRENT state ----
    z = s[:, 2]
    air = z - GROUND_Z
    pitch = s[:, 8]                                     # world_euler pitch
    pitch_w = (pitch + np.pi) % (2 * np.pi) - np.pi
    absp = np.abs(np.degrees(pitch_w))

    regimes = {
        "flat (level, grounded)": (air < 0.03) & (absp < 10),
        "on-ramp (pitched >10deg)": (absp >= 10) & (air < 0.05),
        "airborne (>5cm up)":      air >= 0.05,
    }

    def report(mask, label):
        if mask.sum() == 0:
            print(f"  {label:32s}  n=0"); return
        m = err[mask].mean(axis=0) * 100      # cm
        nv = naive[mask].mean(axis=0) * 100   # cm
        print(f"  {label:32s}  n={mask.sum():>7,}  "
              f"X {m[0]:5.2f}/{nv[0]:5.2f}  Y {m[1]:5.2f}/{nv[1]:5.2f}  Z {m[2]:5.2f}/{nv[2]:5.2f}  "
              f"(Zratio {m[2]/max(nv[2],1e-9):.2f})")

    print("Per-regime single-step position error  [model cm / naive cm], by axis:")
    print("(Zratio = model Z MAE / naive Z MAE; <1 means the model beats predict-zero)\n")
    for label, mask in regimes.items():
        report(mask, label)
    report(np.ones(len(s), bool), "ALL val rows")

    # ---- Cut by vertical-motion magnitude (does it predict the big Z changes?) ----
    print("\nBy true vertical motion |Δz| this step:")
    dz = np.abs(true_dcore[:, 2])
    for lo, hi, lab in [(0.0, 0.01, "|Δz| < 1cm (flat-ish)"),
                        (0.01, 0.05, "|Δz| 1-5cm (gentle)"),
                        (0.05, 0.15, "|Δz| 5-15cm (real jump)"),
                        (0.15, 9.9,  "|Δz| > 15cm (big air)")]:
        report((dz >= lo) & (dz < hi), lab)

    # ---- VELOCITY-delta error by regime: does it predict ACCELERATION? ----
    # Position delta is partly velocity*dt (easy). The ramp physics -- launch impulse,
    # gravity, landing -- lives in how the VELOCITY changes. Naive here = constant
    # velocity (predict Δv=0); beating it means the model captures the forces.
    verr = np.abs(preds - true_dcore)
    vnaive = np.abs(true_dcore)
    print("\nVelocity-delta error by regime  [model / naive m/s] (Δ over one 0.1s step):")
    print("(world Δvz carries gravity + ramp launch + landing; beating naive => real physics)\n")

    def vreport(mask, label):
        if mask.sum() == 0:
            print(f"  {label:32s} n=0"); return
        mvz, nvz = verr[mask, I_WVZ].mean(), vnaive[mask, I_WVZ].mean()
        mvx, nvx = verr[mask, I_WVX].mean(), vnaive[mask, I_WVX].mean()
        print(f"  {label:32s} n={mask.sum():>7,}  "
              f"Δvz {mvz:.3f}/{nvz:.3f} ({mvz/max(nvz,1e-9):.2f})  "
              f"Δvx {mvx:.3f}/{nvx:.3f} ({mvx/max(nvx,1e-9):.2f})")

    for label, mask in regimes.items():
        vreport(mask, label)
    vreport(np.ones(len(s), bool), "ALL val rows")

    # ---- Steering-sign resolution (the left/right plot question) ----
    print("\nSteering-sign check (resolves the XY-vs-3D left/right confusion):")
    steer = a[:, 1]
    moving = np.abs(s[:, I_WVX]) > 0.5
    c = np.corrcoef(steer[moving], ns[moving, I_WYAWRATE])[0, 1]
    data_dir = "+yaw (CCW = robot-left = +Y)" if c > 0 else "-yaw (CW = robot-right = -Y)"
    print(f"  DATA : corr(steering, next world yaw-rate) = {c:+.3f}  -> +steering turns {data_dir}")
    sel = np.where(regimes["flat (level, grounded)"] & moving)[0][:4000]

    def model_next_yawrate(steer_val):
        cc, ee, aa = core[sel].copy(), elev[sel], a[sel].copy()
        aa[:, 1] = steer_val
        with torch.no_grad():
            pc, _ = model(torch.from_numpy(cc).to(device),
                          torch.from_numpy(ee).unsqueeze(1).to(device),
                          torch.from_numpy(aa.astype(np.float32)).to(device))
            pc = pc.cpu().numpy()
            if output_denorm:
                pc = pc * core_t_std + core_t_mean
        return (cc[:, I_WYAWRATE] + pc[:, I_WYAWRATE]).mean()  # predicted next yaw-rate
    yr_pos, yr_neg = model_next_yawrate(0.488), model_next_yawrate(-0.488)
    agree = (yr_pos - yr_neg) * c > 0
    print(f"  MODEL: steering=+max -> next yaw-rate {yr_pos:+.3f} ; steering=-max -> {yr_neg:+.3f}")
    print(f"  -> model and data {'AGREE' if agree else 'DISAGREE'} on which way +steering turns.")
    print("     (AGREE => model steering is physically correct; the plot's 'left'/'right'")
    print("      labels are just names for a steering sign, and the 3D view flips Y visually.)")

    # ---- Teacher-forced tracking along a REAL jump (honest multi-step, real terrain) ----
    track_real_jump(model, config, stats, core_dim, device,
                    core_t_mean, core_t_std, output_denorm)


def track_real_jump(model, config, stats, core_dim, device,
                    core_t_mean, core_t_std, output_denorm):
    """Walk the model one step ahead along a REAL jump episode, feeding the TRUE state
    and TRUE elevation at every step (teacher-forced). This avoids the elevation
    hallucination of synthetic_rollout, so it honestly shows whether the model tracks
    the actual takeoff/airborne/landing arc. Saves a plot."""
    elev_size = config["elevation_map_size"]
    grid = grid_from_size(elev_size)
    # single batch so env/episode/timestep keys are consistent (no cross-batch collisions)
    with h5py.File(DATA_ONE_BATCH, "r") as f:
        states = replace_inf_elevation(f["states"][:].astype(np.float32), elev_size)
        next_states = replace_inf_elevation(f["next_states"][:].astype(np.float32), elev_size)
        actions = f["actions"][:].astype(np.float32)
        env_ids = f["env_ids"][:]
        episode_ids = f["episode_ids"][:]
        timesteps = f["timesteps"][:]
        terminated = f["terminated"][:]
        truncated = f["truncated"][:]

    key = env_ids.astype(np.int64) * 1_000_000 + episode_ids.astype(np.int64)
    air_all = states[:, 2] - GROUND_Z
    # pick a jumpy episode: clear air, enough length, with a terminal row to trim before
    best = None
    for k in np.unique(key):
        rows = np.where(key == k)[0]
        if len(rows) < 12:
            continue
        order = rows[np.argsort(timesteps[rows])]
        peak = air_all[order].max()
        if 0.10 <= peak <= 0.30:
            best = order
            break
    if best is None:
        print("\n[jump-tracking] no clean jump episode found to plot; skipping.")
        return

    # trim at first terminal row (next_state after it is a teleport)
    done = terminated[best] | truncated[best]
    end = int(np.argmax(done)) if done.any() else len(best)
    seq = best[:end if end > 0 else len(best)]
    if len(seq) < 8:
        print("\n[jump-tracking] episode too short after trimming; skipping.")
        return

    core = states[seq, :core_dim]
    elev = states[seq, -elev_size:].reshape(-1, grid, grid)
    act = actions[seq]
    true_next_z = next_states[seq, 2]
    true_z = states[seq, 2]

    with torch.no_grad():
        pc, _ = model(torch.from_numpy(core).to(device),
                      torch.from_numpy(elev).unsqueeze(1).to(device),
                      torch.from_numpy(act).to(device))
        pc = pc.cpu().numpy()
        if output_denorm:
            pc = pc * core_t_std + core_t_mean
    pred_next_z = core[:, 2] + pc[:, 2]              # one-step-ahead predicted z
    pred_next_wvz = core[:, I_WVZ] + pc[:, I_WVZ]    # predicted next world vz

    t = timesteps[seq] * 0.1
    z_mae = np.abs(pred_next_z - true_next_z).mean() * 100
    print(f"\n[jump-tracking] env={env_ids[seq[0]]} ep={episode_ids[seq[0]]} "
          f"len={len(seq)} peak_air={air_all[seq].max()*100:.0f}cm  "
          f"one-step-ahead Z MAE along the jump = {z_mae:.2f} cm")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, true_z, "o-", label="true z", color="black")
    ax[0].plot(t[:-1] + 0.1, pred_next_z[:-1], "x--", label="model 1-step-ahead z", color="tab:red")
    ax[0].axhline(GROUND_Z, color="gray", ls=":", lw=1, label="ground rest (0.19)")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("z (m)")
    ax[0].set_title(f"Real jump arc vs model 1-step prediction\n(Z MAE {z_mae:.2f} cm)")
    ax[0].legend()
    ax[1].plot(t, states[seq, I_WVZ], "o-", label="true world vz", color="black")
    ax[1].plot(t[:-1] + 0.1, pred_next_wvz[:-1], "x--", label="model next vz", color="tab:red")
    ax[1].axhline(0, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("world vz (m/s)")
    ax[1].set_title("Vertical velocity through the jump")
    ax[1].legend()
    out = Path("learned_dynamics/figures/jump_tracking.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
