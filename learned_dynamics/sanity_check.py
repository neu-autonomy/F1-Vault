"""
Sanity check for trained CNN dynamics model.

Applies the SAME boundary filter as training by default. The filter
removes rows where `terminated` or `truncated` is True, since those rows
have `next_states[i]` set to the post-reset spawn (a teleport) rather
than the result of stepping the simulator. Including teleports in eval
makes MAE numbers meaningless because no model can predict random
respawns.

Use --include-boundaries if you want to see what happens without the
filter (you'll see ~10x larger errors driven entirely by the teleports).

Usage:
    python sanity_check.py \\
        --model models/cnn_dynamics_v2/best_model.pt \\
        --data data/raw/dynamics_data_0000.h5
"""

import argparse
import h5py
import numpy as np
import torch

from dynamics_model import load_model, grid_from_size, replace_inf_elevation, INF_FILL


# ============================================================================
# MAIN
# ============================================================================

def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main(args):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    # -- Checkpoint ----------------------------------------------------------
    hr("CHECKPOINT")
    model, cfg, stats, core_dim = load_model(args.model, device)
    elev_size = cfg["elevation_map_size"]
    elev_grid = grid_from_size(elev_size)

    print(f"  predict_delta:               {cfg['predict_delta']}")
    print(f"  use_normalization:           {cfg.get('use_normalization', False)}")
    print(f"  target_only_normalization:   {cfg.get('target_only_normalization', False)}")
    print(f"  trained_with_boundary_filter:{cfg.get('trained_with_boundary_filter', '?')}")
    print(f"  core_state_dim:              {core_dim}")

    # -- Data ----------------------------------------------------------------
    hr("DATA")
    with h5py.File(args.data, "r") as f:
        states_full = f["states"][:]
        actions_full = f["actions"][:]
        next_states_full = f["next_states"][:]
        terminated = f["terminated"][:]
        truncated = f["truncated"][:]
    print(f"  raw rows: {len(states_full):,}")

    if args.include_boundaries:
        print(f"  --include-boundaries set: NOT filtering. Expect inflated MAE")
        print(f"  driven by teleport transitions (post-reset spawns).")
        keep = np.ones(len(states_full), dtype=bool)
    else:
        keep = ~(terminated | truncated)
        print(f"  applying boundary filter: keeping {int(keep.sum()):,} / "
              f"{len(states_full):,} ({100*keep.mean():.1f}%)")
        print(f"  (filter matches training -- evaluates on real physics rows only)")

    states_pool = states_full[keep]
    actions_pool = actions_full[keep]
    next_states_pool = next_states_full[keep]

    np.random.seed(0)
    N = min(args.n, len(states_pool))
    idx = np.random.choice(len(states_pool), N, replace=False)
    states = states_pool[idx]
    actions = actions_pool[idx]
    next_states = next_states_pool[idx]
    print(f"  using random subsample of {N} transitions")

    # -- (1) State layout sanity check --------------------------------------
    hr("(1) STATE LAYOUT")
    print(f"  Across all {N} sampled transitions:")
    for i, lbl in enumerate(["X", "Y", "Z"]):
        s = states[:, i]
        n = next_states[:, i]
        d = n - s
        print(f"    {lbl}:  state range [{s.min():7.3f}, {s.max():7.3f}]  "
              f"|delta| mean={np.abs(d).mean():.4f}m  max={np.abs(d).max():.3f}m")

    # -- (2) Stats sanity check ---------------------------------------------
    hr("(2) SAVED STATS vs. CURRENT-DATA STATS")
    print("  Should match to within sampling noise. If they don't, the data")
    print("  file or the filter setting differs from training.\n")
    print("  axis     saved Δ mean        data Δ mean         saved Δ std         data Δ std")
    for i, lbl in enumerate(["dX", "dY", "dZ"]):
        m_saved = stats["core_target_mean"][i]
        s_saved = stats["core_target_std"][i]
        d = next_states[:, i] - states[:, i]
        print(f"  {lbl}    {m_saved:+12.5f}    {d.mean():+12.5f}     "
              f"{s_saved:11.5f}     {d.std():11.5f}")

    # -- (3) Model -----------------------------------------------------------
    hr("(3) MODEL")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded, {n_params:,} parameters "
          f"(elevation grid {elev_grid}x{elev_grid})")

    # Replace inf
    s = states.copy().astype(np.float32)
    n_inf = int(np.isinf(s[:, -elev_size:]).sum())
    replace_inf_elevation(s, elev_size)
    if n_inf:
        print(f"  replaced {n_inf} inf values in elevation -> {INF_FILL}")

    core_in = s[:, :core_dim]
    elev_in = s[:, -elev_size:].reshape(N, elev_grid, elev_grid)
    act_in = actions.astype(np.float32)

    # Run model in batches (target-only normalization: inputs raw)
    pred_core_norm = np.zeros_like(core_in)
    BS = 256
    with torch.no_grad():
        for start in range(0, N, BS):
            end = min(start + BS, N)
            ct = torch.from_numpy(core_in[start:end]).to(device)
            et = torch.from_numpy(elev_in[start:end]).unsqueeze(1).to(device)
            at = torch.from_numpy(act_in[start:end]).to(device)
            pc, _ = model(ct, et, at)
            pred_core_norm[start:end] = pc.cpu().numpy()

    # Denormalize: target-only normalization expects (pred * std + mean) -> physical delta
    pred_delta = pred_core_norm * stats["core_target_std"] + stats["core_target_mean"]
    actual_delta = next_states[:, :core_dim].astype(np.float32) - core_in

    # -- (4) A few raw rows -------------------------------------------------
    hr("(4) RAW PREDICTIONS  --  first 5 transitions (positions only)")
    for i in range(5):
        print(f"\n  i={i}")
        print(f"    state[:3]      = "
              f"[{core_in[i,0]:8.4f}, {core_in[i,1]:8.4f}, {core_in[i,2]:8.4f}]")
        print(f"    actual_delta   = "
              f"[{actual_delta[i,0]:+8.4f}, {actual_delta[i,1]:+8.4f}, "
              f"{actual_delta[i,2]:+8.4f}]")
        print(f"    pred_delta     = "
              f"[{pred_delta[i,0]:+8.4f}, {pred_delta[i,1]:+8.4f}, "
              f"{pred_delta[i,2]:+8.4f}]")
        err = pred_delta[i] - actual_delta[i]
        print(f"    err            = "
              f"[{err[0]:+8.4f}, {err[1]:+8.4f}, {err[2]:+8.4f}]")

    # -- (5) Per-axis MAE vs naive baseline ---------------------------------
    hr("(5) PER-AXIS MAE  --  model vs. naive 'predict zero motion' baseline")
    print("  All numbers in METERS.\n")
    print(f"  {'axis':>6}  {'model MAE':>11}  {'naive MAE':>11}  "
          f"{'ratio':>8}  {'verdict':>10}")
    print("  " + "-" * 58)
    for i, lbl in enumerate(["X", "Y", "Z"]):
        mae = np.abs(pred_delta[:, i] - actual_delta[:, i]).mean()
        naive = np.abs(actual_delta[:, i]).mean()
        ratio = mae / naive if naive > 1e-9 else float("inf")
        verdict = "BETTER" if mae < naive else "WORSE"
        print(f"  {lbl:>6}  {mae:>10.5f}m  {naive:>10.5f}m  "
              f"{ratio:>7.2f}x  {verdict:>10}")

    print("\n  Interpretation:")
    print("    'naive MAE' is the error you'd get by predicting delta = 0")
    print("    (i.e. claiming the robot doesn't move). If model MAE > naive,")
    print("    the model is actively making predictions worse than zero.")

    # -- (6) Full core MAE for direct training-log comparison --------------
    hr("(6) FULL CORE MAE  --  compare with training log's reported numbers")
    err_full = np.abs(pred_delta - actual_delta)
    print(f"  Mean MAE across all {core_dim} core dims: "
          f"{err_full.mean():.5f}m")
    print(f"  Per-dim MAE (first 7 dims = position+quaternion):")
    for d in range(min(7, core_dim)):
        m = err_full[:, d].mean()
        n_ = np.abs(actual_delta[:, d]).mean()
        marker = " <-- worse than naive" if m > n_ else ""
        print(f"    dim {d:2d}: {m:.5f}m  (naive {n_:.5f}m){marker}")

    print("\n  These should match the training log's per-axis MAE within")
    print("  sampling noise (~10% on 2000 samples).")

    hr("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check for trained model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data",  type=str, required=True)
    parser.add_argument("--n",     type=int, default=2000,
                        help="Number of transitions to sample (default 2000)")
    parser.add_argument("--include-boundaries", action="store_true",
                        help="Skip the boundary filter. WILL show inflated MAE "
                             "due to teleport transitions; useful only for "
                             "comparing filter on/off.")
    args = parser.parse_args()
    main(args)