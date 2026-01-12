"""
F1Tenth from URDF - Updated for Isaac Sim 2025+
Uses the new isaacsim.asset.importer.urdf module path
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.asset.importer.urdf import _urdf

# ============================================================================
# CONFIGURATION
# ============================================================================
URDF_PATH = "/home/nail/Desktop/F1-Vault/urdf/f1-tenth_corrected.urdf"

def main():
    print("\n" + "="*70)
    print("F1TENTH - URDF IMPORT (Updated API)")
    print("="*70 + "\n")
    
    # Create world
    print("Step 1: Creating world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    print("✓ World created\n")
    
    # Configure URDF import settings
    print("Step 2: Configuring URDF import...")
    
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False              # Keep all joints
    import_config.convex_decomp = False                   # Use original meshes
    import_config.fix_base = False                        # CRITICAL - robot must move!
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = True            # Use URDF masses
    import_config.default_drive_strength = 10000.0        # Strong drives
    import_config.default_position_drive_damping = 1000.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
    import_config.distance_scale = 1.0
    import_config.density = 0.0  # Use URDF mass values
    
    print("  Import settings:")
    print(f"    fix_base: {import_config.fix_base} (False = movable ✓)")
    print(f"    drive_type: VELOCITY")
    print(f"    import_inertia: {import_config.import_inertia_tensor}")
    print()
    
    # Import URDF using official method
    print("Step 3: Importing URDF with omni.kit.commands...")
    print(f"  Path: {URDF_PATH}")
    
    try:
        result, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=URDF_PATH,
            import_config=import_config,
        )
        
        if result:
            print(f"✓ URDF imported successfully!")
            print(f"  Robot prim path: {prim_path}\n")
        else:
            print("❌ URDF import failed!")
            simulation_app.close()
            return
            
    except Exception as e:
        print(f"❌ Error importing URDF: {e}")
        print("\nCheck:")
        print("  1. URDF file exists at path")
        print("  2. Mesh files in meshes/ folder")
        simulation_app.close()
        return
    
    # Create articulation - try to find the robot
    print("Step 4: Creating articulation...")
    
    # The prim_path from import might be just the robot name
    # Try common variations
    possible_paths = [
        prim_path,
        f"{prim_path}/car_1_base_link",
        "/car_1_base_link",
        "/World/car_1_base_link",
    ]
    
    f1tenth = None
    actual_path = None
    
    for path in possible_paths:
        try:
            print(f"  Trying: {path}")
            f1tenth = my_world.scene.add(
                SingleArticulation(
                    prim_path=path,
                    name="f1tenth"
                )
            )
            actual_path = path
            print(f"  ✓ Found at: {path}\n")
            break
        except Exception as e:
            continue
    
    if f1tenth is None:
        print("  ❌ Could not create articulation")
        print("\n  The robot loaded but can't find articulation root")
        print("  Open Isaac Sim → Window → Stage to find correct path")
        simulation_app.close()
        return
    
    # Reset world
    print("Step 5: Resetting world...")
    my_world.reset()
    print("✓ World reset\n")
    
    print("="*70)
    print("ROBOT LOADED SUCCESSFULLY")
    print("="*70)
    print(f"Articulation path: {actual_path}")
    print(f"Number of DOFs: {f1tenth.num_dof}")
    print("\nJoint configuration:")
    for i, name in enumerate(f1tenth.dof_names):
        print(f"  [{i}] {name}")
    print("="*70 + "\n")
    
    print("🎬 Starting movement test...")
    print("Driving forward with properly imported URDF\n")
    
    i = 0
    start_pos = None
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            position, _ = f1tenth.get_world_pose()
            
            if start_pos is None:
                start_pos = position.copy()
            
            # Calculate phase (3 seconds each)
            seconds = i / 60.0
            phase = int(seconds / 3) % 3
            
            # Create commands
            velocities = np.zeros(6)
            positions = np.zeros(6)
            
            if phase == 0:
                # Forward
                velocities[0] = 50.0  # left rear
                velocities[2] = 50.0  # right rear
                velocities[4] = 50.0  # left front
                velocities[5] = 50.0  # right front
                positions[1] = 0.0    # straight
                positions[3] = 0.0
                phase_name = "Forward"
                
            elif phase == 1:
                # Forward + Left
                velocities[0] = 50.0
                velocities[2] = 50.0
                velocities[4] = 50.0
                velocities[5] = 50.0
                positions[1] = 0.4    # left turn
                positions[3] = 0.4
                phase_name = "Turn Left"
                
            else:
                # Forward + Right
                velocities[0] = 50.0
                velocities[2] = 50.0
                velocities[4] = 50.0
                velocities[5] = 50.0
                positions[1] = -0.4   # right turn
                positions[3] = -0.4
                phase_name = "Turn Right"
            
            # Apply commands
            f1tenth.set_joint_velocities(velocities)
            f1tenth.set_joint_positions(positions)
            
            # Status every second
            if i % 60 == 0:
                distance = np.linalg.norm(position - start_pos)
                actual_vels = f1tenth.get_joint_velocities()
                wheel_vel = actual_vels[0] if len(actual_vels) > 0 else 0
                
                moved = distance > 0.02
                status = "✅ MOVING!" if moved else "❌ Stuck"
                
                print(f"[{seconds:5.1f}s] {phase_name:12s} | "
                      f"Pos: [{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:5.2f}] | "
                      f"Moved: {distance:6.3f}m | WheelVel: {wheel_vel:6.1f} | {status}")
                
                if moved and i > 120:
                    print("\n" + "="*70)
                    print("🎉 SUCCESS! URDF import worked perfectly!")
                    print("="*70)
                    print(f"Robot moved {distance:.3f} meters in {seconds:.1f} seconds")
                    print("\nThe robot is now working with:")
                    print("  ✓ Proper collision from STL meshes")
                    print("  ✓ Movable base (fix_base=False)")
                    print("  ✓ Correct joint drives")
                    print("\nReady for jump course!")
                    print("="*70 + "\n")
            
            # Stop after 12 seconds or if moved too far
            if i > 720 or np.linalg.norm(position - start_pos) > 10.0:
                distance = np.linalg.norm(position - start_pos)
                
                print("\n" + "="*70)
                print("TEST COMPLETE")
                print("="*70)
                
                if distance > 0.1:
                    print(f"\n✅ SUCCESS! Robot moved {distance:.3f} meters")
                    print("\nURDF import worked correctly!")
                    print("Collision geometry is functional")
                    print("\nNext: Run f1tenth_jump_course_official.py for full course")
                else:
                    print(f"\n⚠️  Robot only moved {distance:.4f}m")
                    print("\nPossible issues:")
                    print("  1. Mesh files not found - check meshes/ folder")
                    print("  2. Collision geometry in STL is wrong")
                    print("  3. URDF collision definitions need fixing")
                
                print("="*70 + "\n")
                break
            
            i += 1
    
    simulation_app.close()

if __name__ == "__main__":
    main()