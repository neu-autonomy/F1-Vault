"""
action_conditioning_test.py

The single most important diagnostic before using a learned dynamics model
in a controller. The model passes if its rollouts depend strongly on the
action sequence; it fails if all action sequences produce similar trajectories.

The setup: pick ONE starting state. Roll the model out N steps under several
very different action sequences (max throttle, zero throttle, min throttle,
hard left steering, hard right steering, etc.). Plot all rollouts on the
same axes. Visualize the spread.

How to read the output:

  PASS:
    - Trajectories fan out clearly. Max-throttle goes further than zero-throttle.
    - Left vs right steering produce mirrored XY paths.
    - Spread grows over time (small at t=0, large at t=T).
    - "Action sensitivity" number (defined below) is well above zero.

  FAIL:
    - All trajectories look similar regardless of action.
    - Z trajectory has the same shape for every action sequence (this was the
      red flag in the synthetic_rollouts plots).
    - Sensitivity number near zero -> model ignores action entirely.

Usage:
    python action_conditioning_test.py \\
        --model models/cnn_dynamics_v2/best_model.pt \\
        --data data/raw/dynamics_data_0000.h5 \\
        --output_dir learned_dynamics/figures \\
        --steps 30
"""

import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dynamics_model import DT, load_model, normalization_mode, synthetic_rollout


# ============================================================================
# ACTION-SEQUENCE LIBRARY
# ============================================================================

def make_action_sequences(steps):
    """Return dict of named action sequences, each (steps, 2) shape.

    action[0] = throttle in [-1, 1]
    action[1] = steering in [-1, 1]
    """
    seqs = {}
    seqs["full_throttle"]    = np.tile([+1.0,  0.0], (steps, 1)).astype(np.float32)
    seqs["zero_throttle"]    = np.tile([ 0.0,  0.0], (steps, 1)).astype(np.float32)
    seqs["reverse_throttle"] = np.tile([-1.0,  0.0], (steps, 1)).astype(np.float32)
    seqs["full_left"]        = np.tile([+1.0, -1.0], (steps, 1)).astype(np.float32)
    seqs["full_right"]       = np.tile([+1.0, +1.0], (steps, 1)).astype(np.float32)
    seqs["half_throttle"]    = np.tile([+0.5,  0.0], (steps, 1)).astype(np.float32)
    return seqs


SEQ_COLORS = {
    "full_throttle":    "#d62728",  # red
    "half_throttle":    "#ff7f0e",  # orange
    "zero_throttle":    "#7f7f7f",  # gray
    "reverse_throttle": "#1f77b4",  # blue
    "full_left":        "#2ca02c",  # green
    "full_right":       "#9467bd",  # purple
}


# ============================================================================
# PLOTTING
# ============================================================================

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_xy_overlay(rollouts, start_idx, out_dir):
    """Top-down view of all rollouts from the same starting state."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_xy = axes[0]
    ax_z = axes[1]

    for name, traj in rollouts.items():
        pos = traj[:, :3]
        ts = np.arange(len(pos)) * DT
        c = SEQ_COLORS.get(name, "black")
        ax_xy.plot(pos[:, 0], pos[:, 1], lw=2, alpha=0.85, color=c, label=name)
        ax_z.plot(ts, pos[:, 2], lw=2, alpha=0.85, color=c, label=name)

    start = next(iter(rollouts.values()))[0, :3]
    ax_xy.scatter(start[0], start[1], c="black", s=120, marker="*", zorder=10,
                  label="start")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title(f"Top-down trajectories (same start, different actions)\n"
                    f"start row {start_idx}")
    ax_xy.legend(fontsize=8, loc="best")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")

    ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Z (m)")
    ax_z.set_title("Z over time (same start, different actions)")
    ax_z.legend(fontsize=8, loc="best")
    ax_z.grid(True, alpha=0.3)

    _save(fig, out_dir / "action_conditioning_xy_z.png")


def plot_3d_overlay(rollouts, start_idx, out_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, traj in rollouts.items():
        pos = traj[:, :3]
        c = SEQ_COLORS.get(name, "black")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=2, alpha=0.85, color=c, label=name)
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c=c, s=40, marker="x")
    start = next(iter(rollouts.values()))[0, :3]
    ax.scatter(start[0], start[1], start[2], c="black", s=160, marker="*", label="start")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"3D trajectories (start row {start_idx})")
    ax.legend(fontsize=8); ax.view_init(elev=20, azim=45)
    _save(fig, out_dir / "action_conditioning_3d.png")


def plot_pairwise_divergence(rollouts, out_dir):
    """For each pair of action sequences, plot |traj_a - traj_b| over time
    in XY position. If pairs all stay near zero, model isn't action-conditional.
    """
    names = list(rollouts.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = rollouts[names[i]][:, :2]
            b = rollouts[names[j]][:, :2]
            d = np.linalg.norm(a - b, axis=1)
            ts = np.arange(len(d)) * DT
            ax.plot(ts, d, lw=1.5, alpha=0.75,
                    label=f"{names[i]} vs {names[j]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("XY-position divergence (m)")
    ax.set_title("Pairwise XY divergence between rollouts\n"
                 "(grows over time = model uses actions)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "action_conditioning_divergence.png")


# ============================================================================
# SENSITIVITY METRIC
# ============================================================================

def compute_sensitivity(rollouts):
    """Quantify how much rollouts depend on the action sequence.

    Returns a scalar 'action sensitivity' = mean pairwise XY distance at the
    final timestep, divided by typical per-step motion (~0.5m). A value near 0
    means actions don't matter to the model; > 1 means actions strongly shape
    the predicted trajectory.

    Also returns per-axis ranges of the final positions across action sequences,
    which tells you which axes the model can be steered in.
    """
    names = list(rollouts.keys())
    final_pos = np.array([rollouts[n][-1, :3] for n in names])  # (k, 3)

    # Pairwise final-XY distances
    pair_dists = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(final_pos[i, :2] - final_pos[j, :2])
            pair_dists.append(d)
    mean_pair_dist = float(np.mean(pair_dists))

    # Per-axis range
    per_axis_range = final_pos.max(axis=0) - final_pos.min(axis=0)

    return mean_pair_dist, per_axis_range, final_pos


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output dir: {out_dir.resolve()}")

    print(f"\nLoading model from {args.model} ...")
    model, config, stats, core_dim = load_model(args.model, device)
    input_norm, output_denorm = normalization_mode(config)
    print(f"  core_dim={core_dim}  predict_delta={config['predict_delta']}  "
          f"input_norm={input_norm}  output_denorm={output_denorm}")

    print(f"\nLoading data from {args.data} ...")
    with h5py.File(args.data, "r") as f:
        states_full = f["states"][:]
        terminated = f["terminated"][:]
        truncated = f["truncated"][:]

    keep = ~(terminated | truncated)
    states_pool = states_full[keep]
    print(f"  filtered to {len(states_pool):,} valid starting states")

    # Pick a starting state. Default: random. User can pass --start_row.
    if args.start_row is not None:
        if args.start_row >= len(states_full):
            raise ValueError(f"start_row {args.start_row} out of range")
        x0 = states_full[args.start_row]
        start_idx = args.start_row
    else:
        np.random.seed(args.seed)
        start_idx = int(np.random.choice(len(states_pool)))
        x0 = states_pool[start_idx]

    print(f"\nStarting state (row {start_idx}):")
    print(f"  position: X={x0[0]:.3f} Y={x0[1]:.3f} Z={x0[2]:.3f}")
    print(f"  base_lin_vel: {x0[10:13]}")

    # Build all the action sequences and run each
    seqs = make_action_sequences(args.steps)
    print(f"\nRunning {len(seqs)} rollouts of {args.steps} steps each "
          f"from the same start ...")
    rollouts = {}
    for name, a_seq in seqs.items():
        rollouts[name] = synthetic_rollout(
            model, x0, a_seq, config, stats, core_dim, device,
        )

    # Sensitivity metric
    print("\n" + "=" * 78)
    print("ACTION SENSITIVITY")
    print("=" * 78)
    mean_pair, per_axis_range, final_pos = compute_sensitivity(rollouts)

    print(f"\n  Mean pairwise final-XY distance:  {mean_pair:.3f}m")
    print(f"  Range of final positions across action sequences:")
    print(f"    X: {per_axis_range[0]:.3f}m")
    print(f"    Y: {per_axis_range[1]:.3f}m")
    print(f"    Z: {per_axis_range[2]:.3f}m")

    print(f"\n  Final positions per action sequence:")
    print(f"  {'sequence':<20}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
          f"{'net displ.':>11}")
    for i, name in enumerate(seqs.keys()):
        fp = final_pos[i]
        net = np.linalg.norm(fp - x0[:3])
        print(f"  {name:<20}  {fp[0]:>8.3f}  {fp[1]:>8.3f}  {fp[2]:>8.3f}  "
              f"{net:>10.3f}m")

    print("\n  Interpretation:")
    if mean_pair < 0.10:
        verdict = "FAIL"
        msg = ("Trajectories are nearly identical regardless of action.\n"
               "  Model is NOT action-conditional. Don't use it for control --\n"
               "  improving the model architecture won't help; this is a\n"
               "  training/data problem.")
    elif mean_pair < 0.5:
        verdict = "WEAK"
        msg = ("Trajectories depend on action somewhat but the effect is small\n"
               "  compared to typical per-step motion (~0.5m). MPC might work\n"
               "  for gentle maneuvers but not for high-acceleration tasks\n"
               "  like jumping.")
    else:
        verdict = "PASS"
        msg = ("Trajectories clearly depend on action. Model is responsive\n"
               "  enough for control. Quality of control will depend on whether\n"
               "  the action-response directions match physical reality (check\n"
               "  the plots: full_left should sweep left of full_right in XY,\n"
               "  reverse_throttle should go opposite of full_throttle, etc).")
    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    # Plots
    print("\nGenerating plots...")
    plot_xy_overlay(rollouts, start_idx, out_dir)
    plot_3d_overlay(rollouts, start_idx, out_dir)
    plot_pairwise_divergence(rollouts, out_dir)

    print("\n" + "=" * 78)
    print("Next: open the plots and check whether the action-response *directions*")
    print("are physically sensible, not just whether the trajectories differ.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether the dynamics model is action-conditional"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="learned_dynamics/figures")
    parser.add_argument("--steps", type=int, default=30,
                        help="Rollout length per action sequence (default 30)")
    parser.add_argument("--start_row", type=int, default=None,
                        help="Specific data row to use as start. "
                             "Default: random valid row.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for picking start row "
                             "(if --start_row not given)")
    args = parser.parse_args()
    main(args)