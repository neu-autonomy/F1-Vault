"""
Load real jump episodes from the collected data, for the faithful (recorded-
terrain) controller demo.

The dataset stores one-step transitions from 512 parallel envs, interleaved
across envs. An episode is one (env_id, episode_id) group sorted by timestep,
truncated at the first terminated|truncated row (those next_states are post-reset
teleports — see CLAUDE.md). We keep episodes that contain a real jump so the
controller has a ramp to clear.
"""

import os
import sys

import h5py
import numpy as np

_LD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "learned_dynamics"))
if _LD not in sys.path:
    sys.path.insert(0, _LD)
from dynamics_model import replace_inf_elevation  # noqa: E402

GROUND_AIR_REF = 0.19


def _straightness(poses):
    """Max perpendicular deviation [m] of poses from the start->end chord.

    Small = straight corridor. The recorded-terrain oracle only trusts poses near
    the corridor, so a straight episode lets 'drive to the goal' keep the car on it.
    """
    a, b = poses[0], poses[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-6:
        return np.inf
    n = np.array([-ab[1], ab[0]]) / L          # unit normal to the chord
    return float(np.abs((poses - a) @ n).max())


def load_jump_episode(h5_path="data/raw/dynamics_data_0000.h5",
                      min_len=18, air_lo=0.10, air_hi=0.30, index=0,
                      min_span=3.0, max_candidates=2000):
    """Return a recorded jump episode: a long, STRAIGHT corridor with a real jump.

    Scans episodes with length >= min_len, peak air in [air_lo, air_hi], and
    start->end span >= min_span (so the jump happens along a real forward run), then
    ranks by straightness (see `_straightness`) and returns the `index`-th
    straightest. A straight corridor lets 'drive to the goal' keep the car on the
    recorded terrain. Returns a dict:
        core0    (22,)      first core state (spawn, forward-moving)
        poses    (M, 2)     world xy per step
        yaws     (M,)       yaw per step
        patches  (M,26,26)  recorded, inf-filled elevation patches
        actions  (M-1, 2)   recorded actions (for sanity replay)
        true_z   (M,)       recorded root z
        peak_air float
    """
    with h5py.File(h5_path, "r") as f:
        S = replace_inf_elevation(f["states"][:].astype(np.float32), 676)
        A = f["actions"][:].astype(np.float32)
        env, ep, ts = f["env_ids"][:], f["episode_ids"][:], f["timesteps"][:]
        done = f["terminated"][:] | f["truncated"][:]

    key = env.astype(np.int64) * 1_000_000 + ep.astype(np.int64)
    air = S[:, 2] - GROUND_AIR_REF
    cands = []
    for k in np.unique(key):
        r = np.where(key == k)[0]
        if len(r) < min_len:
            continue
        o = r[np.argsort(ts[r])]
        d = done[o]
        e = int(np.argmax(d)) if d.any() else len(o)
        o = o[:e]
        if len(o) < min_len or not (air_lo <= air[o].max() <= air_hi):
            continue
        poses = S[o, :2]
        if np.linalg.norm(poses[-1] - poses[0]) < min_span:
            continue
        cands.append((_straightness(poses), o))
        if len(cands) >= max_candidates:
            break
    if not cands:
        raise ValueError(f"no jump episode (len>={min_len}, air in "
                         f"[{air_lo},{air_hi}]) in {h5_path}")
    cands.sort(key=lambda c: c[0])
    _, o = cands[min(index, len(cands) - 1)]
    return {
        "core0": S[o[0], :22].copy(),
        "poses": S[o, :2].copy(),
        "yaws": S[o, 9].copy(),
        "patches": S[o, -676:].reshape(-1, 26, 26).copy(),
        "actions": A[o[:-1]].copy(),
        "true_z": S[o, 2].copy(),
        "peak_air": float(air[o].max()),
    }
