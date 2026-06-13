"""
Per-transition diagnostics + synthetic rollouts for the CNN dynamics model.

What it does and why:

  - Respects the checkpoint's normalization flags (target-only models are
    fed RAW inputs -- normalizing them anyway once caused rollouts to
    diverge to 4500m).
  - Applies the SAME boundary filter as training by default (boundary rows
    are post-reset teleports). Use --include-boundaries to skip.
  - Pre-flight check compares saved stats to current-data stats; if they
    disagree, the data file or filter differs from training and the rest
    of the report is meaningless.
  - All headline metrics report the naive predict-zero baseline next to
    model MAE so a useless model is obvious.

Usage:
    python learned_dynamics/visualize_trajectories.py \\
        --model models/cnn_dynamics_v2/best_model.pt \\
        --data data/raw/dynamics_data_0000.h5
"""

import matplotlib
matplotlib.use("Agg")  # MUST come before pyplot import

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dynamics_model import (DT, load_model, normalization_mode, grid_from_size,
                            replace_inf_elevation, synthetic_rollout)


# ============================================================================
# BATCHED PREDICTION
# ============================================================================

def predict_transitions_batched(model, states, actions, config, stats,
                                core_dim, device, batch_size=512):
    elev_size = config["elevation_map_size"]
    grid = grid_from_size(elev_size)
    predict_delta = config["predict_delta"]
    input_norm, output_denorm = normalization_mode(config)

    N = len(states)
    pred_core_all = np.zeros((N, core_dim), dtype=np.float32)
    pred_elev_all = np.zeros((N, grid, grid), dtype=np.float32)

    states = states.copy()
    replace_inf_elevation(states, elev_size)

    core_in = states[:, :core_dim].astype(np.float32)
    elev_in = states[:, -elev_size:].reshape(N, grid, grid).astype(np.float32)
    act_in = actions.astype(np.float32)

    if input_norm:
        core_in_use = (core_in - stats["core_state_mean"]) / stats["core_state_std"]
        elev_in_use = (elev_in - stats["elevation_mean"]) / stats["elevation_std"]
        act_in_use = (act_in - stats["action_mean"]) / stats["action_std"]
    else:
        core_in_use, elev_in_use, act_in_use = core_in, elev_in, act_in

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            ct = torch.from_numpy(core_in_use[start:end]).to(device)
            et = torch.from_numpy(elev_in_use[start:end]).unsqueeze(1).to(device)
            at = torch.from_numpy(act_in_use[start:end]).to(device)
            pc, pe = model(ct, et, at)
            pc = pc.cpu().numpy()
            pe = pe.squeeze(1).cpu().numpy()

            if output_denorm:
                pc = pc * stats["core_target_std"] + stats["core_target_mean"]
                pe = pe * stats["elevation_target_std"] + stats["elevation_target_mean"]

            if predict_delta:
                pc = core_in[start:end] + pc
                pe = elev_in[start:end] + pe

            pred_core_all[start:end] = pc
            pred_elev_all[start:end] = pe

    return pred_core_all, pred_elev_all


# ============================================================================
# PLOTTING
# ============================================================================

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_pred_vs_actual_scatter(actual_delta, pred_delta, out_dir, n_show=3000):
    if len(actual_delta) > n_show:
        idx = np.random.choice(len(actual_delta), n_show, replace=False)
        actual_delta = actual_delta[idx]
        pred_delta = pred_delta[idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for i, label in enumerate(["X", "Y", "Z"]):
        ax = axes[i]
        a, p = actual_delta[:, i], pred_delta[:, i]
        ax.scatter(a, p, s=4, alpha=0.3, color="steelblue")
        lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        corr = np.corrcoef(a, p)[0, 1] if a.std() > 1e-8 and p.std() > 1e-8 else float("nan")
        ax.set_xlabel(f"Actual Δ{label} (m)")
        ax.set_ylabel(f"Predicted Δ{label} (m)")
        ax.set_title(f"{label}-axis  |  ρ = {corr:.3f}  |  n = {len(a):,}")
        ax.legend(loc="best"); ax.grid(True, alpha=0.3); ax.axis("equal")
    _save(fig, out_dir / "scatter_pred_vs_actual.png")


def plot_error_histograms(errors_per_axis, actual_delta, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    flat = axes.flatten()
    for i, label in enumerate(["X", "Y", "Z"]):
        ax = flat[i]
        e = errors_per_axis[:, i] * 100
        ax.hist(e, bins=60, alpha=0.75, edgecolor="black", color="steelblue")
        mae = np.abs(e).mean()
        rmse = np.sqrt((e ** 2).mean())
        p95 = np.percentile(np.abs(e), 95)
        naive = np.abs(actual_delta[:, i]).mean() * 100
        ax.axvline(0, color="k", linestyle="--", alpha=0.4)
        ax.set_xlabel(f"{label} prediction error (cm)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: MAE {mae:.2f}cm (naive {naive:.2f}cm)  "
                     f"RMSE {rmse:.2f}cm  95% {p95:.2f}cm")
        ax.grid(True, alpha=0.3, axis="y")

    ax = flat[3]
    e3 = np.linalg.norm(errors_per_axis, axis=1) * 100
    ax.hist(e3, bins=60, alpha=0.75, edgecolor="black", color="purple")
    ax.axvline(e3.mean(), color="r", linestyle="--", lw=2,
               label=f"Mean: {e3.mean():.2f}cm")
    ax.axvline(np.median(e3), color="g", linestyle="--", lw=2,
               label=f"Median: {np.median(e3):.2f}cm")
    ax.set_xlabel("3D position error (cm)"); ax.set_ylabel("Count")
    ax.set_title(f"3D | Mean: {e3.mean():.2f}cm  "
                 f"95th pct: {np.percentile(e3, 95):.2f}cm")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / "error_histograms.png")


def plot_error_vs_action(errors_per_axis, actions, out_dir, n_show=3000):
    if len(errors_per_axis) > n_show:
        idx = np.random.choice(len(errors_per_axis), n_show, replace=False)
        errors_per_axis = errors_per_axis[idx]
        actions = actions[idx]

    err_3d = np.linalg.norm(errors_per_axis, axis=1) * 100
    act_mag = np.linalg.norm(actions, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(act_mag, err_3d, s=4, alpha=0.3, color="steelblue")
    axes[0].set_xlabel("|action|"); axes[0].set_ylabel("3D prediction error (cm)")
    axes[0].set_title("Error vs action magnitude"); axes[0].grid(True, alpha=0.3)
    axes[1].scatter(actions[:, 0], err_3d, s=4, alpha=0.3, color="steelblue")
    axes[1].set_xlabel("Throttle (action[0])"); axes[1].set_ylabel("3D prediction error (cm)")
    axes[1].set_title("Error vs throttle"); axes[1].grid(True, alpha=0.3)
    axes[2].scatter(actions[:, 1], err_3d, s=4, alpha=0.3, color="steelblue")
    axes[2].set_xlabel("Steering (action[1])"); axes[2].set_ylabel("3D prediction error (cm)")
    axes[2].set_title("Error vs steering"); axes[2].grid(True, alpha=0.3)
    _save(fig, out_dir / "error_vs_action.png")


def plot_synthetic_rollouts(model, states, actions, config, stats, core_dim,
                            device, out_dir, n_rollouts=4, rollout_steps=30):
    np.random.seed(0)
    starts = np.random.choice(len(states), n_rollouts, replace=False)
    cols = min(2, n_rollouts)
    rows = int(np.ceil(n_rollouts / cols))
    fig = plt.figure(figsize=(7 * cols, 5 * rows))

    for k, s_idx in enumerate(starts):
        x0 = states[s_idx]
        a_seq = actions[s_idx:s_idx + rollout_steps]
        if len(a_seq) < rollout_steps:
            continue
        rollout = synthetic_rollout(model, x0, a_seq, config, stats, core_dim, device)
        pos = rollout[:, :3]
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "r-", lw=2, alpha=0.8,
                label="Synthetic rollout")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c="green", s=100, marker="o",
                   label=f"Start (row {s_idx})")
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c="red", s=80, marker="x",
                   label="End")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        total = np.linalg.norm(pos[-1] - pos[0])
        ax.set_title(f"Rollout {k} | {rollout_steps} steps | "
                     f"net displacement: {total:.2f}m")
        ax.legend(fontsize=8); ax.view_init(elev=20, azim=45)
    _save(fig, out_dir / "synthetic_rollouts_3d.png")


def plot_synthetic_rollouts_xy(model, states, actions, config, stats, core_dim,
                               device, out_dir, n_rollouts=4, rollout_steps=30):
    np.random.seed(0)
    starts = np.random.choice(len(states), n_rollouts, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_xy, ax_x, ax_y, ax_z = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for k, s_idx in enumerate(starts):
        x0 = states[s_idx]
        a_seq = actions[s_idx:s_idx + rollout_steps]
        if len(a_seq) < rollout_steps:
            continue
        rollout = synthetic_rollout(model, x0, a_seq, config, stats, core_dim, device)
        pos = rollout[:, :3]
        ts = np.arange(len(pos)) * DT
        label = f"rollout {k}"
        ax_xy.plot(pos[:, 0], pos[:, 1], lw=2, alpha=0.8, label=label)
        ax_xy.scatter(pos[0, 0], pos[0, 1], c="green", s=60, marker="o", zorder=5)
        ax_x.plot(ts, pos[:, 0], lw=2, alpha=0.8, label=label)
        ax_y.plot(ts, pos[:, 1], lw=2, alpha=0.8, label=label)
        ax_z.plot(ts, pos[:, 2], lw=2, alpha=0.8, label=label)

    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Synthetic rollouts (top-down)")
    ax_xy.grid(True, alpha=0.3); ax_xy.legend(); ax_xy.axis("equal")
    for ax, lbl in [(ax_x, "X"), (ax_y, "Y"), (ax_z, "Z")]:
        ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{lbl} (m)")
        ax.set_title(f"{lbl} position over time")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out_dir / "synthetic_rollouts_2d.png")


# ============================================================================
# MAIN
# ============================================================================

def main(data_file, model_path, output_dir, n_transitions, n_rollouts,
         rollout_steps, include_boundaries):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output dir: {out_dir.resolve()}")

    print(f"\nLoading model from {model_path} ...")
    model, config, stats, core_dim = load_model(model_path, device)
    input_norm, output_denorm = normalization_mode(config)
    print(f"  core_state_dim: {core_dim}")
    print(f"  predict_delta:  {config['predict_delta']}")
    print(f"  normalization:  input_norm={input_norm}  output_denorm={output_denorm}")
    print(f"  (target_only_normalization={config.get('target_only_normalization', False)}, "
          f"use_normalization={config.get('use_normalization', False)})")

    print(f"\nLoading data from {data_file} ...")
    with h5py.File(data_file, "r") as f:
        states      = f["states"][:]
        actions     = f["actions"][:]
        next_states = f["next_states"][:]
        terminated  = f["terminated"][:]
        truncated   = f["truncated"][:]
    print(f"  raw rows: {len(states):,}")

    if include_boundaries:
        print(f"  --include-boundaries set: NOT filtering. Expect inflated MAE")
        print(f"  driven by teleport transitions (next_state = post-reset spawn).")
    else:
        keep = ~(terminated | truncated)
        states = states[keep]
        actions = actions[keep]
        next_states = next_states[keep]
        print(f"  boundary filter: keeping {int(keep.sum()):,} "
              f"({100*keep.mean():.1f}% of raw rows) — matches training filter")

    # Subsample
    if n_transitions and n_transitions < len(states):
        np.random.seed(42)
        idx = np.random.choice(len(states), n_transitions, replace=False)
        states = states[idx]
        actions = actions[idx]
        next_states = next_states[idx]
        print(f"  using random subsample of {len(states):,} transitions")

    # Pre-flight: saved stats vs current-data stats
    print("\nPre-flight: saved-stats vs data-stats sanity check")
    print(f"  {'axis':>4}   {'saved Δ mean':>15}  {'data Δ mean':>15}  "
          f"{'saved Δ std':>15}  {'data Δ std':>15}")
    print(f"  " + "-" * 76)
    for i, lbl in enumerate(["dX", "dY", "dZ"]):
        m_s = stats["core_target_mean"][i]
        s_s = stats["core_target_std"][i]
        d = next_states[:, i].astype(np.float32) - states[:, i].astype(np.float32)
        print(f"  {lbl:>4}  {m_s:+15.5f}  {d.mean():+15.5f}  "
              f"{s_s:15.5f}  {d.std():15.5f}")
    print("  (if columns disagree, the data file or filter differs from training)")

    # Predict every transition
    print("\nRunning model on all transitions...")
    pred_core_abs, _ = predict_transitions_batched(
        model, states, actions, config, stats, core_dim, device,
    )

    actual_pos = next_states[:, :3]
    pred_pos = pred_core_abs[:, :3]
    initial_pos = states[:, :3]
    actual_delta = actual_pos - initial_pos
    pred_delta = pred_pos - initial_pos
    pos_errors = pred_pos - actual_pos
    err_3d = np.linalg.norm(pos_errors, axis=1)

    print("\n" + "=" * 78)
    print("PER-TRANSITION PREDICTION ACCURACY")
    print("=" * 78)
    print(f"  {'axis':>4}    {'model MAE':>11}  {'naive MAE':>11}  "
          f"{'95%':>9}  {'verdict':>10}")
    print(f"  " + "-" * 58)
    for i, lbl in enumerate(["X", "Y", "Z"]):
        mae = np.abs(pos_errors[:, i]).mean() * 100
        naive = np.abs(actual_delta[:, i]).mean() * 100
        p95 = np.percentile(np.abs(pos_errors[:, i]), 95) * 100
        verdict = "BETTER" if mae < naive else "WORSE"
        print(f"  {lbl:>4}    {mae:>10.2f}cm  {naive:>10.2f}cm  "
              f"{p95:>8.2f}cm  {verdict:>10}")
    print(f"\n  3D mean error: {err_3d.mean()*100:.2f}cm   "
          f"median: {np.median(err_3d)*100:.2f}cm   "
          f"95%: {np.percentile(err_3d, 95)*100:.2f}cm")

    # Plots
    print("\nGenerating plots...")
    plot_pred_vs_actual_scatter(actual_delta, pred_delta, out_dir)
    plot_error_histograms(pos_errors, actual_delta, out_dir)
    plot_error_vs_action(pos_errors, actions, out_dir)

    print("\nRunning synthetic rollouts (closed-loop on the model)...")
    plot_synthetic_rollouts(
        model, states, actions, config, stats, core_dim, device, out_dir,
        n_rollouts=n_rollouts, rollout_steps=rollout_steps,
    )
    plot_synthetic_rollouts_xy(
        model, states, actions, config, stats, core_dim, device, out_dir,
        n_rollouts=n_rollouts, rollout_steps=rollout_steps,
    )

    print("\n" + "=" * 78)
    print(f"All figures saved to: {out_dir.resolve()}/")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostics + synthetic rollouts")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="learned_dynamics/figures")
    parser.add_argument("--n_transitions", type=int, default=20000,
                        help="Random subsample size (0 = all)")
    parser.add_argument("--n_rollouts", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=30)
    parser.add_argument("--include-boundaries", action="store_true",
                        help="Don't apply the boundary filter (will show "
                             "inflated MAE from teleport transitions).")
    args = parser.parse_args()

    main(
        data_file=args.data,
        model_path=args.model,
        output_dir=args.output_dir,
        n_transitions=args.n_transitions if args.n_transitions > 0 else None,
        n_rollouts=args.n_rollouts,
        rollout_steps=args.rollout_steps,
        include_boundaries=args.include_boundaries,
    )