"""
Quick health report for a dynamics H5 file. Read-only, no model needed.

Merges the old analyze_data.py (transition integrity) and analyze_data_new.py
(action stats), plus the episode-structure checks that explained why the
first models failed (see CLAUDE.md): if most episodes are 1-2 steps long,
the filtered training set is dominated by post-spawn transients and a
one-step model will learn the spawn ballistics instead of driving dynamics.

Usage:
    python learned_dynamics/inspect_data.py [--data data/raw/dynamics_data_0000.h5]
"""

import argparse

import h5py
import numpy as np


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(args):
    with h5py.File(args.data, "r") as f:
        states = f["states"][:]
        actions = f["actions"][:]
        next_states = f["next_states"][:]
        terminated = f["terminated"][:]
        truncated = f["truncated"][:]
        episode_ids = f["episode_ids"][:]
        env_ids = f["env_ids"][:]
        timesteps = f["timesteps"][:]
        attrs = dict(f.attrs)

    hr("FILE ATTRS")
    for k in sorted(attrs):
        print(f"  {k}: {attrs[k]}")

    valid = ~(terminated | truncated)

    hr("EPISODE STRUCTURE (the thing that broke the first models)")
    key = env_ids.astype(np.int64) * 1_000_000 + episode_ids
    _, ep_lens = np.unique(key, return_counts=True)
    print(f"  rows: {len(states):,}   episodes: {len(ep_lens):,}")
    print(f"  boundary rows (terminated|truncated): {(~valid).sum():,} "
          f"({100 * (~valid).mean():.1f}% of all rows)")
    print(f"  episode length steps: median={np.median(ep_lens):.0f}  "
          f"mean={ep_lens.mean():.1f}  max={ep_lens.max()}")
    print(f"  episodes <= 2 steps: {100 * (ep_lens <= 2).mean():.1f}%")
    for t in range(3):
        frac = (timesteps[valid] == t).mean()
        print(f"  fraction of FILTERED rows at timestep {t}: {100 * frac:.1f}%")
    print("  (healthy data: long episodes, low boundary fraction, timesteps spread out)")

    hr("ACTION STATS (filtered rows)")
    a = actions[valid]
    s = states[valid]
    print(f"  throttle: mean={a[:, 0].mean():+.3f}  std={a[:, 0].std():.3f}  "
          f"range=[{a[:, 0].min():+.2f}, {a[:, 0].max():+.2f}]")
    print(f"  steering: mean={a[:, 1].mean():+.3f}  std={a[:, 1].std():.3f}  "
          f"range=[{a[:, 1].min():+.2f}, {a[:, 1].max():+.2f}]")
    print(f"  |throttle| > 0.9:  {(np.abs(a[:, 0]) > 0.9).mean() * 100:.1f}%")
    print(f"  |steering| > 0.9:  {(np.abs(a[:, 1]) > 0.9).mean() * 100:.1f}%")

    dp = next_states[valid][:, :3] - s[:, :3]
    print("\n  corr(action, one-step position delta) -- low values mean a one-step")
    print("  model has almost no action signal to learn from:")
    for ai, an in [(0, "throttle"), (1, "steering")]:
        cs = [np.corrcoef(a[:, ai], dp[:, i])[0, 1] for i in range(3)]
        print(f"    {an}: dX={cs[0]:+.3f}  dY={cs[1]:+.3f}  dZ={cs[2]:+.3f}")

    hr("TRANSITION INTEGRITY (states[i+1] vs next_states[i] within episodes)")
    # Rows are interleaved across envs (all envs at t, then t+1, ...), so sort
    # into per-env order first.
    order = np.lexsort((timesteps, env_ids))
    e2, t2, ep2 = env_ids[order], timesteps[order], episode_ids[order]
    bnd2 = (terminated | truncated)[order]
    consecutive = ((e2[:-1] == e2[1:]) & (ep2[:-1] == ep2[1:])
                   & (t2[1:] == t2[:-1] + 1) & ~bnd2[:-1])
    n = int(consecutive.sum())
    if n:
        d = np.abs(next_states[order][:-1][consecutive][:, :3]
                   - states[order][1:][consecutive][:, :3])
        print(f"  {n:,} consecutive same-episode pairs")
        print(f"  max position mismatch: {d.max():.4f}m  "
              f"(should be ~0; large values mean next_states is not states[i+1])")
    else:
        print("  no consecutive same-episode pairs found")

    hr("BOUNDARY-ROW TELEPORTS (why the boundary filter exists)")
    d_all = np.abs(next_states[:, :3] - states[:, :3])
    d_val = d_all[valid]
    d_bnd = d_all[~valid]
    print(f"  max |Δpos| per axis, filtered rows:  {d_val.max(axis=0).round(2)}")
    print(f"  max |Δpos| per axis, boundary rows:  {d_bnd.max(axis=0).round(2)}")
    print("  (boundary next_states are post-reset spawns, not physics)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H5 dynamics-data health report")
    parser.add_argument("--data", type=str, default="data/raw/dynamics_data_0000.h5")
    main(parser.parse_args())
