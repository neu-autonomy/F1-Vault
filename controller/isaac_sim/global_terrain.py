"""
Online global heightmap for the Isaac-Sim controller test.

The MPPI planner rolls candidate action sequences several steps into the future,
so it needs terrain at *hypothetical* poses ahead of the robot — not just the one
26x26 patch under the robot right now. We build that terrain the faithful way:
every control step the sim hands us the REAL yaw-aligned 26x26 elevation patch
(the same RayCaster/world_height_map pipeline the model trained on). We "splat"
each patch into a persistent world-frame heightmap; the planner then samples
yaw-aligned patches from that map at any candidate pose via the project's
`TerrainMap`.

Why this and not the analytic ramp (controller/terrain.py build_ramp_terrain):
CONTROLLER_HANDOFF.md warns the analytic TerrainMap's zero-offset/orientation is
best-effort and *under-drives* the model's launch. Here we never invent geometry —
we accumulate the sim's own patches, so the map carries the true ramp shape in the
exact training convention. Unobserved cells default to flat (FLAT_Z), which is
correct: the landing zone past the kicker is flat ground.

Convention (matches TerrainMap / the sim GridPattern): a patch reshaped (26,26)[i,j]
has i along the robot heading (+forward), j across (+left); cell spacing 0.1 m,
extent +/-1.25 m. Flat ground reads FLAT_Z.
"""

import os
import sys

import numpy as np

_LD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "learned_dynamics"))
if _LD not in sys.path:
    sys.path.insert(0, _LD)
from dynamics_api import TerrainMap  # noqa: E402

FLAT_Z = -0.20      # flat-ground elevation value in the training convention
GRID = 26           # 26x26 patch
PATCH_RES = 0.1     # patch cell spacing [m]
# Plausible terrain-height band for a splatted cell. world_height_map clamps ray MISSES
# (airborne rays that hit nothing) to +2.0 and floor/neg cases to the low end; those are
# "unobserved", NOT obstacles. Splatting the +2 cells put PHANTOM WALLS in the planner's
# map ahead of the robot -> after clearing the ramp MPPI "saw" a wall and turned away from
# the goal (diagnosed 2026-07-12). Flat ground ~ -0.20, the ramp tops out ~ +0.3, so a real
# cell sits well inside [-0.9, 1.0]; anything outside is a ray-miss/clamp artifact -> skip.
VALID_LO = -0.9
VALID_HI = 1.0


class GlobalHeightMap:
    """World-frame heightmap accumulated from live sim elevation patches.

    Query it exactly like a terrain source: ``get_patch(core_batch, t) -> (B,26,26)``.
    """

    def __init__(self, x0, y0, res=0.05, back=3.0, fwd=9.0, side=4.0, fill=FLAT_Z):
        self.res = float(res)
        self.ox = float(x0) - float(back)
        self.oy = float(y0) - float(side)
        nx = int(round((back + fwd) / res)) + 1
        ny = int(round((2.0 * side) / res)) + 1
        self.H = np.full((ny, nx), float(fill), np.float32)
        self._g = (np.arange(GRID) - (GRID - 1) / 2.0) * PATCH_RES   # -1.25 .. 1.25
        # TerrainMap keeps a reference to self.H (float32 -> no copy), so in-place
        # splats below are visible to get_patch without rewrapping.
        self._tm = TerrainMap(self.H, res=self.res, origin=(self.ox, self.oy))

    def update(self, core, patch):
        """Splat the live yaw-aligned patch under `core` into the global map.

        The patch is sampled on a 0.1 m grid but the world map is finer (res, default
        0.05 m). A nearest-neighbour splat therefore writes only ~1 of every (0.1/res)^2
        map cells and leaves HOLES between them; a later yaw-rotated get_patch query lands
        in those holes and bilinearly interpolates against flat filler -> big round-trip
        error (was ~20 cm at any nonzero yaw, on a ~25 cm ramp -> planner half-blind to the
        ramp off-axis). Fix: write each patch cell into the (2r+1)^2 block of map cells it
        covers (r spans half the patch spacing), so the splatted region is dense -- no holes
        for rotated queries to fall into. Adjacent blocks overlap on real ramp values, so
        overwrite order is harmless."""
        x, y, yaw = float(core[0]), float(core[1]), float(core[9])
        c, s = np.cos(yaw), np.sin(yaw)
        gx, gy = np.meshgrid(self._g, self._g, indexing="ij")        # (26,26) along, cross
        wx = (x + gx * c - gy * s).ravel()
        wy = (y + gx * s + gy * c).ravel()
        vals = np.asarray(patch, np.float32).ravel()
        # Drop ray-miss / clamp artifacts: they are unobserved cells, not obstacles.
        # Splatting them (esp. the +2 "wall") corrupts the planner's map (see VALID_* note).
        valid = (vals > VALID_LO) & (vals < VALID_HI)
        if valid.mean() < 0.5:
            # >half the patch is artifacts -> robot is airborne / rays mostly miss; this
            # reading is unreliable, so don't write anything (keep the map as last seen).
            return
        ix0 = np.round((wx - self.ox) / self.res).astype(int)
        iy0 = np.round((wy - self.oy) / self.res).astype(int)
        r = int(np.ceil((PATCH_RES / self.res) / 2.0))               # footprint half-width
        ny, nx = self.H.shape
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ix = ix0 + dj
                iy = iy0 + di
                m = valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                self.H[iy[m], ix[m]] = vals[m]

    def get_patch(self, core_batch, t=None):
        return self._tm.get_patch(core_batch, t)

    def resample_error(self, core, patch):
        """Max |map-resampled patch - live patch| at `core` [m] (round-trip sanity)."""
        got = self._tm.get_patch(np.asarray(core, np.float32)[None])[0]
        return float(np.max(np.abs(got - np.asarray(patch, np.float32))))


def describe_patch_orientation(patch):
    """Report where the elevated terrain sits in a live patch, to confirm the
    heading axis. At spawn the kicker is ~1 m AHEAD, so the high cells should fall
    at along-index (row) > center. Returns a human string."""
    p = np.asarray(patch, np.float32).reshape(GRID, GRID)
    high = p > (FLAT_Z + 0.05)
    if not high.any():
        return "no elevated cells yet (flat under robot)"
    ai, cj = np.where(high)
    along = (ai.mean() - (GRID - 1) / 2.0) * PATCH_RES
    cross = (cj.mean() - (GRID - 1) / 2.0) * PATCH_RES
    where = "AHEAD" if along > 0.15 else ("BEHIND" if along < -0.15 else "under")
    return (f"elevated terrain centroid {along:+.2f} m along / {cross:+.2f} m across "
            f"heading -> {where} (expected AHEAD near spawn)")
