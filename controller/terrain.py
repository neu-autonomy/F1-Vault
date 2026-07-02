"""
Synthetic terrain for the MuSHR jump controller.

The learned dynamics model does NOT know the terrain — every step the controller
must supply the local 26x26 elevation patch (yaw-aligned, 0.1 m res, flat ground
~ FLAT_Z). See CONTROLLER_HANDOFF.md: feeding correct terrain is the one thing
you must get right; the hallucinating rollout misses jumps.

Here we build a KNOWN static heightmap (flat ground + a "kicker" ramp) in the
training convention and wrap it in the project's `TerrainMap`, which samples the
yaw-aligned patch for any pose. Because the demo uses the same TerrainMap for
both the plant and the planner, terrain is always consistent and never
hallucinated — this is a controller test against the learned model as plant, not
an independent physics ground truth (that step needs Isaac Sim / WheeledLab).

Heightmap convention (matches TerrainMap): H[iy, ix] is world-frame terrain
height; world x = origin_x + ix*res, world y = origin_y + iy*res. Flat ground is
FLAT_Z; a ramp rising `rise` metres reads FLAT_Z + rise at its lip.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# make learned_dynamics importable regardless of CWD
_LD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "learned_dynamics"))
if _LD not in sys.path:
    sys.path.insert(0, _LD)
from dynamics_api import TerrainMap  # noqa: E402

# Flat-ground elevation-map value in the training convention (see handoff/CLAUDE.md).
FLAT_Z = -0.20


@dataclass
class RampSpec:
    """A kicker ramp crossing the field laterally (varies with world x only).

    The car approaches along +x; terrain rises linearly from `x_start` to the lip
    at `x_start + up_len`, reaching height `rise`, then drops back to flat so the
    car launches off the lip. `width` is finite lateral extent (metres) centred on
    y=0; beyond it the ground is flat (set very large for a full-width ridge).
    """
    x_start: float = 3.0
    up_len: float = 0.7      # ~0.6-0.75 m ramp footprint in the data
    rise: float = 0.28       # real ramps read ~0.26-0.29 m rise in the elevation map
    width: float = 4.0


def build_ramp_terrain(ramp: RampSpec = None,
                       x_range=(-1.5, 6.0), y_range=(-3.0, 3.0), res=0.05):
    """Flat ground with one kicker ramp -> (TerrainMap, meta dict)."""
    ramp = ramp or RampSpec()
    xs = np.arange(x_range[0], x_range[1] + res, res)
    ys = np.arange(y_range[0], y_range[1] + res, res)
    H = np.full((ys.size, xs.size), FLAT_Z, np.float32)

    lip = ramp.x_start + ramp.up_len
    # rise profile as a function of world x, clipped to the up-ramp segment
    up = np.clip((xs - ramp.x_start) / max(ramp.up_len, 1e-6), 0.0, 1.0)
    up[xs > lip] = 0.0                                   # drop back to flat past the lip
    lateral = (np.abs(ys) <= ramp.width / 2).astype(np.float32)  # (ny,)
    H += (ramp.rise * up)[None, :] * lateral[:, None]

    tm = TerrainMap(H, res=res, origin=(x_range[0], y_range[0]))
    meta = {"ramp": ramp, "lip_x": lip, "x_range": x_range, "y_range": y_range,
            "res": res, "H": H, "xs": xs, "ys": ys}
    return tm, meta


def build_flat_terrain(x_range=(-1.5, 6.0), y_range=(-3.0, 3.0), res=0.05):
    """Featureless flat ground (sanity baseline) -> (TerrainMap, meta)."""
    xs = np.arange(x_range[0], x_range[1] + res, res)
    ys = np.arange(y_range[0], y_range[1] + res, res)
    H = np.full((ys.size, xs.size), FLAT_Z, np.float32)
    tm = TerrainMap(H, res=res, origin=(x_range[0], y_range[0]))
    return tm, {"H": H, "xs": xs, "ys": ys, "x_range": x_range,
                "y_range": y_range, "res": res}


class RecordedTerrain:
    """Faithful terrain oracle backed by REAL recorded elevation patches.

    This is the terrain source the handoff blesses ("the same RayCaster/elevation
    pipeline used in data collection ... pass those maps straight in"): no
    orientation/zero-offset guesswork, so the model launches exactly as it did in
    the data. Given one recorded episode's poses + patches, `get_patch` returns the
    patch of the nearest recorded pose (xy distance + a small yaw penalty). Valid
    while the controller stays near the recorded corridor — which it does when the
    goal lies along that corridor. `max_dist_seen` lets the caller detect straying.

    poses:   (M, 2) recorded world xy
    yaws:    (M,)   recorded yaw [rad]
    patches: (M, 26, 26) recorded, inf-filled elevation patches
    """

    def __init__(self, poses, yaws, patches, yaw_weight=0.3):
        self.poses = np.asarray(poses, np.float32)
        self.yaws = np.asarray(yaws, np.float32)
        self.patches = np.asarray(patches, np.float32)
        self.yaw_weight = yaw_weight
        self.max_dist_seen = 0.0

    def get_patch(self, core_batch, t=None):
        cb = np.atleast_2d(np.asarray(core_batch, np.float32))
        out = np.empty((cb.shape[0], 26, 26), np.float32)
        for i, c in enumerate(cb):
            dxy = np.linalg.norm(self.poses - c[:2], axis=1)
            dyaw = np.abs((self.yaws - c[9] + np.pi) % (2 * np.pi) - np.pi)
            j = int((dxy + self.yaw_weight * dyaw).argmin())
            self.max_dist_seen = max(self.max_dist_seen, float(dxy[j]))
            out[i] = self.patches[j]
        return out


def make_get_patch(terrain_map):
    """Adapt a TerrainMap / RecordedTerrain into the `get_patch(core_batch, t)` callable."""
    return lambda core_batch, t=None: terrain_map.get_patch(core_batch, t)
