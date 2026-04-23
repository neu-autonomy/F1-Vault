#!/usr/bin/env python3
"""
Per-TRANSITION diagnostic. Counts fraction of samples (not episodes) where:
  - airborne: robot_z > terrain_under_robot + epsilon (truly off the ground)
  - on_terrain_feature: robot_z > 0.22 (on elevated surface, 3cm above ground)
  - fast: |base_vx| > 2 m/s
  - jumping_at_speed: airborne AND fast simultaneously
  - on_ramp_at_speed: on_terrain_feature AND fast simultaneously

These are what you actually train on — per-transition fractions, not per-episode.
A single 5-second "good" episode with fast ramp driving produces 50 useful transitions,
which is what matters for the dynamics model.

Usage:
    python diagnose_transitions.py --file_path dynamics_data_0000.h5
"""

import argparse
import h5py
import numpy as np


IDX_POS = slice(0, 3)
IDX_EULER = slice(7, 10)
IDX_BASE_LIN_VEL = slice(10, 13)
# Elevation map is the LAST 625 values (25x25 grid). Center cell is the height
# directly under the robot.
ELEVATION_CENTER_IDX = -625 + (12 * 25 + 12)  # row 12, col 12 of 25x25

GROUND_Z = 0.19
FEATURE_THRESHOLD = 0.22   # 3cm above ground = on an elevated surface
AIRBORNE_EPSILON = 0.05    # 5cm above local terrain = truly airborne
FAST_VEL = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True)
    parser.add_argument('--sample_n', type=int, default=0,
                        help='only analyze first N transitions (0 = all)')
    args = parser.parse_args()

    with h5py.File(args.file_path, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]

    if args.sample_n > 0:
        states = states[:args.sample_n]
        actions = actions[:args.sample_n]

    N = len(states)
    print(f"Loaded {N:,} transitions from {args.file_path}\n")

    # Extract fields
    robot_z = states[:, 2]
    pitch = states[:, 8]
    base_vx = states[:, 10]
    throttle = actions[:, 0]

    # Terrain height under robot (center of elevation map)
    # The elevation map is in world frame relative to ground plane at z=0,
    # so terrain_under_robot is the height of the ground at robot's XY.
    terrain_under = states[:, ELEVATION_CENTER_IDX]

    # For airborne: how far is robot ABOVE the local terrain?
    # robot_z is base_link height. Robot sits with base_link ~0.19m above contact surface.
    # So if contact surface is at terrain_under, robot should be at terrain_under + 0.19.
    # Airborne = robot_z > (terrain_under + 0.19 + epsilon)
    airborne_clearance = robot_z - terrain_under - GROUND_Z
    airborne = airborne_clearance > AIRBORNE_EPSILON

    # On elevated feature (not necessarily airborne)
    on_feature = robot_z > FEATURE_THRESHOLD

    # Fast
    fast = np.abs(base_vx) > FAST_VEL
    fast_forward = base_vx > FAST_VEL  # strictly forward-fast

    # Combined
    jump_at_speed = airborne & fast
    ramp_at_speed = on_feature & fast
    feature_any = on_feature | airborne

    print("=" * 70)
    print("PER-TRANSITION FRACTIONS")
    print("=" * 70)

    print(f"\nVelocity:")
    print(f"  |base_vx| > 1.0 m/s:        {(np.abs(base_vx) > 1.0).mean()*100:6.2f}%")
    print(f"  |base_vx| > 2.0 m/s (fast): {fast.mean()*100:6.2f}%")
    print(f"  |base_vx| > 2.5 m/s:        {(np.abs(base_vx) > 2.5).mean()*100:6.2f}%")
    print(f"  base_vx > 2.0 (fast fwd):   {fast_forward.mean()*100:6.2f}%")
    print(f"  median |base_vx|: {np.median(np.abs(base_vx)):.3f} m/s")
    print(f"  mean |base_vx|:   {np.abs(base_vx).mean():.3f} m/s")

    print(f"\nHeight (robot base_link z):")
    print(f"  Z > 0.20 (slightly elevated):     {(robot_z > 0.20).mean()*100:6.2f}%")
    print(f"  Z > 0.22 (on feature, 3cm up):    {(robot_z > 0.22).mean()*100:6.2f}%")
    print(f"  Z > 0.30 (major elevation):       {(robot_z > 0.30).mean()*100:6.2f}%")
    print(f"  Z > 0.40 (significant):           {(robot_z > 0.40).mean()*100:6.2f}%")

    print(f"\nTerrain context:")
    print(f"  terrain under robot (median): {np.median(terrain_under):.3f} m")
    print(f"  terrain under robot (max):    {terrain_under.max():.3f} m")
    print(f"  robot clearance above terrain (median): {np.median(airborne_clearance):.3f} m")
    print(f"  airborne (clearance > {AIRBORNE_EPSILON}m): {airborne.mean()*100:6.2f}%")

    print(f"\n>>> THE KEY METRICS <<<")
    print(f"  on terrain feature (any speed):     {on_feature.mean()*100:6.2f}%")
    print(f"  on terrain feature AND fast:        {ramp_at_speed.mean()*100:6.2f}%   <- jumping candidates")
    print(f"  truly airborne:                     {airborne.mean()*100:6.2f}%")
    print(f"  airborne AND fast:                  {jump_at_speed.mean()*100:6.2f}%   <- actual jumps")

    print(f"\nOrientation:")
    pitch_deg = np.rad2deg(np.unwrap(pitch))
    print(f"  |pitch| > 10 deg (on incline):      {(np.abs(pitch_deg) > 10).mean()*100:6.2f}%")
    print(f"  |pitch| > 20 deg (steep):           {(np.abs(pitch_deg) > 20).mean()*100:6.2f}%")

    print(f"\nAction distribution check:")
    print(f"  |throttle| > 0.8 (committed):       {(np.abs(throttle) > 0.8).mean()*100:6.2f}%")
    print(f"  mean |throttle|:                    {np.abs(throttle).mean():.3f}")
    print(f"  (Beta(0.3, 0.3) baseline: mean ~0.70, committed ~56%)")


if __name__ == "__main__":
    main()