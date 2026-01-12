"""
F1Tenth Jump Course - FLIPPED RAMPS
All ramps are rotated 180 degrees to face opposite direction
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from omni.isaac.core.objects import FixedCuboid, VisualCuboid
from isaacsim.asset.importer.urdf import _urdf

# ============================================================================
# CONFIGURATION
# ============================================================================
URDF_PATH = "/home/nail/Desktop/F1-Vault/urdf/f1-tenth_corrected.urdf"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def create_flipped_ramp(world, name, position, length=2.0, width=1.5, angle_deg=25):
    """Create a launch ramp FLIPPED 180 degrees"""
    print(f"  Creating {name} ({angle_deg}° angle, FLIPPED)...")
    
    # Flat approach section - FLIPPED
    base = world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Ramps/{name}_base",
            name=f"{name}_base",
            position=np.array([position[0] + length*0.6, position[1], position[2] + 0.05]),  # +0.6 instead of -0.6
            scale=np.array([length*0.5, width, 0.1]),
            color=np.array([0.3, 0.3, 0.3])
        )
    )
    
    # Angled launch section - FLIPPED
    angle_rad = np.radians(angle_deg)
    ramp_height = length * np.sin(angle_rad) / 2
    
    ramp = world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Ramps/{name}_ramp",
            name=f"{name}_ramp",
            position=np.array([position[0], position[1], position[2] + ramp_height/2]),
            scale=np.array([length, width, 0.15]),
            color=np.array([1.0, 0.4, 0.0])  # Orange
        )
    )
    
    # Create rotation: tilt + 180-degree flip
    # Tilt rotation (around Y-axis)
    tilt_quat = np.array([np.cos(angle_rad / 2), 0, np.sin(angle_rad / 2), 0])
    
    # 180-degree rotation around Z-axis (vertical flip)
    flip_quat = np.array([0, 0, 0, 1])  # 180° around Z
    
    # Combine rotations
    final_quat = quaternion_multiply(flip_quat, tilt_quat)
    
    ramp.set_world_pose(
        position=np.array([position[0], position[1], position[2] + ramp_height/2]),
        orientation=final_quat
    )
    
    return base, ramp

def create_landing_zone(world, name, position, size=3.0):
    """Create green landing zone marker"""
    landing = world.scene.add(
        VisualCuboid(
            prim_path=f"/World/Ramps/{name}_landing",
            name=f"{name}_landing",
            position=np.array([position[0], position[1], position[2] + 0.01]),
            scale=np.array([size, size, 0.02]),
            color=np.array([0.2, 0.8, 0.2])  # Green
        )
    )
    return landing

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print("🏁 F1TENTH JUMP COURSE - FLIPPED RAMPS (180°) 🏁")
    print("="*70 + "\n")
    
    # Create world
    print("Step 1: Creating world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    print("✓ World created\n")
    
    # Build jump course with FLIPPED ramps
    print("Step 2: Building jump course (all ramps FLIPPED 180°)...\n")
    
    print("  🔺 Ramp 1: Easy Jump (FLIPPED)")
    create_flipped_ramp(my_world, "ramp1", [6.0, 0.0, 0.0], length=2.5, angle_deg=20)
    create_landing_zone(my_world, "landing1", [3.0, 0.0, 0.0], size=3.0)  # Landing before ramp now
    
    print("  🔺 Ramp 2: Medium Jump (FLIPPED)")
    create_flipped_ramp(my_world, "ramp2", [3.0, -8.0, 0.0], length=3.0, angle_deg=28)
    create_landing_zone(my_world, "landing2", [-1.0, -8.0, 0.0], size=3.5)
    
    print("  🔺 Ramp 3: Big Jump (FLIPPED)")
    create_flipped_ramp(my_world, "ramp3", [-6.0, -3.0, 0.0], length=3.5, angle_deg=30)
    create_landing_zone(my_world, "landing3", [-11.0, -3.0, 0.0], size=4.0)
    
    print("\n✓ Course built: 3 FLIPPED ramps + landing zones\n")
    
    # Configure URDF import
    print("Step 3: Configuring URDF import settings...")
    
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = False  # CRITICAL!
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = True
    import_config.default_drive_strength = 10000.0
    import_config.default_position_drive_damping = 1000.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    
    print("✓ Config ready\n")
    
    # Import URDF
    print("Step 4: Importing F1Tenth URDF...")
    print(f"  File: {URDF_PATH}")
    
    try:
        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=URDF_PATH,
            import_config=import_config,
        )
        
        if not result:
            print("❌ Import failed!")
            simulation_app.close()
            return
        
        print(f"✓ URDF imported at: {prim_path}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        simulation_app.close()
        return
    
    # Find articulation root
    print("Step 5: Creating articulation...")
    
    possible_paths = [
        prim_path,
        f"{prim_path}/car_1_base_link",
        "/car_1_base_link",
    ]
    
    f1tenth = None
    
    for path in possible_paths:
        try:
            print(f"  Trying: {path}")
            f1tenth = my_world.scene.add(
                SingleArticulation(prim_path=path, name="f1tenth")
            )
            print(f"  ✓ Success!\n")
            break
        except:
            continue
    
    if f1tenth is None:
        print("❌ Could not create articulation")
        simulation_app.close()
        return
    
    # Reset
    print("Step 6: Resetting world...")
    my_world.reset()
    print("✓ Ready!\n")
    
    print("="*70)
    print("🎬 STARTING AUTONOMOUS JUMP SEQUENCE (FLIPPED RAMPS)")
    print("="*70)
    print(f"Robot DOFs: {f1tenth.num_dof}")
    print(f"Note: Ramps now face OPPOSITE direction!")
    print("="*70 + "\n")
    
    # Driving sequence - ADJUSTED for flipped ramps
    driving_sequence = [
        # === RAMP 1: EASY JUMP (now approach from opposite side) ===
        (2.0,  0.0, 1.5),   # Approach
        (1.5,  0.0, 3.5),   # 🚀 BOOST for jump!
        (1.0,  0.0, 2.0),   # Landing
        
        # === NAVIGATE TO RAMP 2 ===
        (2.0, -0.4, 1.8),   # Turn right
        (1.0, -0.3, 2.0),   # Continue turn
        (1.0,  0.0, 1.5),   # Straighten out
        
        # === RAMP 2: MEDIUM JUMP ===
        (1.0,  0.0, 2.5),   # Speed up
        (1.2,  0.0, 4.0),   # 🚀 BIG BOOST!
        (1.0,  0.0, 2.0),   # Landing
        
        # === NAVIGATE TO RAMP 3 ===
        (2.0,  0.5, 1.8),   # Turn left
        (1.5,  0.4, 2.0),   # Continue turn
        (1.5,  0.1, 2.0),   # Adjust heading
        
        # === RAMP 3: BIG JUMP ===
        (1.0,  0.0, 3.0),   # Build up speed
        (1.5,  0.0, 4.5),   # 🚀 MEGA BOOST!
        (1.5,  0.0, 2.0),   # Landing
        
        # === RETURN TO START ===
        (3.0, -0.3, 2.0),   # Navigate back
        (2.0,  0.0, 1.5),   # Slow down to start
    ]
    
    # Simulation variables
    i = 0
    seq_idx = 0
    seq_timer = 0
    lap = 1
    max_height = 0.0
    air_time_frames = 0
    jump_count = 0
    last_airborne = False
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            position, _ = f1tenth.get_world_pose()
            velocity = f1tenth.get_linear_velocity()
            speed = np.linalg.norm(velocity)
            
            # Track maximum jump height
            if position[2] > max_height:
                max_height = position[2]
            
            # Track air time and count jumps
            is_airborne = position[2] > 0.2
            if is_airborne:
                air_time_frames += 1
                if not last_airborne:
                    jump_count += 1
                    print(f"    💥 JUMP #{jump_count}! Current height: {position[2]:.2f}m")
            last_airborne = is_airborne
            
            # Get current command from sequence
            if seq_idx < len(driving_sequence):
                duration, steering, velocity_ms = driving_sequence[seq_idx]
                frames = duration * 60
                
                if seq_timer >= frames:
                    seq_idx += 1
                    seq_timer = 0
                else:
                    seq_timer += 1
            else:
                steering, velocity_ms = 0.0, 0.0
            
            # Create joint commands
            velocities = np.zeros(6)
            positions = np.zeros(6)
            
            # Convert m/s to rad/s
            wheel_angular_vel = velocity_ms / 0.05
            
            # Set all wheel velocities
            velocities[0] = wheel_angular_vel  # left rear
            velocities[2] = wheel_angular_vel  # right rear
            velocities[4] = wheel_angular_vel  # left front
            velocities[5] = wheel_angular_vel  # right front
            
            # Set steering angles
            positions[1] = steering  # left steering
            positions[3] = steering  # right steering
            
            # Apply commands
            f1tenth.set_joint_velocities(velocities)
            f1tenth.set_joint_positions(positions)
            
            # Print telemetry every second
            if i % 60 == 0:
                # Determine phase name
                if velocity_ms > 3.0:
                    phase = "🚀 BOOST!"
                elif abs(steering) > 0.2:
                    phase = "↻ Turning"
                else:
                    phase = "→ Cruising"
                
                # Airborne indicator
                airborne = "✈️  FLYING!" if is_airborne else ""
                
                print(f"[Lap {lap}] {phase:12s} | "
                      f"Pos: [{position[0]:5.1f}, {position[1]:5.1f}, {position[2]:4.2f}] | "
                      f"Speed: {speed:4.1f} m/s | "
                      f"MaxH: {max_height:4.2f}m | "
                      f"Jumps: {jump_count} {airborne}")
            
            # Reset on lap completion or fall
            if position[2] < -0.3 or seq_idx >= len(driving_sequence):
                air_time_sec = air_time_frames / 60.0
                
                print(f"\n{'='*70}")
                print(f"🏁 LAP {lap} COMPLETE!")
                print(f"{'='*70}")
                print(f"  Max jump height: {max_height:.2f} meters")
                print(f"  Total jumps: {jump_count}")
                print(f"  Total air time: {air_time_sec:.1f} seconds")
                print(f"  Course time: {i/60.0:.1f} seconds")
                print(f"{'='*70}\n")
                
                # Reset for next lap
                my_world.reset()
                lap += 1
                seq_idx = 0
                seq_timer = 0
                max_height = 0.0
                air_time_frames = 0
                jump_count = 0
                last_airborne = False
                i = 0
                continue
            
            i += 1
    
    simulation_app.close()
    print("\n🏁 Simulation ended")

if __name__ == "__main__":
    main()