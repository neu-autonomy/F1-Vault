"""
F1Tenth - Add Wheel Collisions
Adds proper collision cylinders and friction to wheels
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics, UsdGeom, PhysxSchema, Gf

# ============================================================================
# CONFIGURATION
# ============================================================================
USD_PATH = "/home/nail/Desktop/F1-Vault/urdf/f1-tenth_corrected/f1-tenth_corrected.usd"

def add_wheel_collision_and_friction(stage, wheel_path, radius=0.05, width=0.045):
    """
    Add collision cylinder and friction material to a wheel
    
    Args:
        stage: USD stage
        wheel_path: Path to wheel prim (e.g., "/World/F1Tenth/car_1_left_rear_wheel")
        radius: Wheel radius in meters
        width: Wheel width in meters
    """
    wheel_prim = stage.GetPrimAtPath(wheel_path)
    
    if not wheel_prim.IsValid():
        print(f"  ❌ Wheel not found: {wheel_path}")
        return False
    
    print(f"  Configuring: {wheel_path}")
    
    # Create collision shape path
    collision_path = f"{wheel_path}/collision_cylinder"
    
    # Check if collision already exists
    collision_prim = stage.GetPrimAtPath(collision_path)
    if not collision_prim.IsValid():
        # Create cylinder for collision
        collision_prim = UsdGeom.Cylinder.Define(stage, collision_path)
        print(f"    → Created collision cylinder")
    else:
        collision_prim = UsdGeom.Cylinder(collision_prim)
    
    # Set cylinder dimensions
    collision_prim.GetRadiusAttr().Set(radius)
    collision_prim.GetHeightAttr().Set(width)
    collision_prim.GetAxisAttr().Set("Z")  # Z-axis aligned (horizontal wheel)
    
    # Add collision API
    if not collision_prim.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(collision_prim.GetPrim())
        print(f"    → Added CollisionAPI")
    
    # Add physics material with friction
    if not collision_prim.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        material_api = UsdPhysics.MaterialAPI.Apply(collision_prim.GetPrim())
    else:
        material_api = UsdPhysics.MaterialAPI(collision_prim.GetPrim())
    
    # Set friction values
    material_api.CreateStaticFrictionAttr().Set(1.0)   # High static friction
    material_api.CreateDynamicFrictionAttr().Set(0.9)  # High dynamic friction
    material_api.CreateRestitutionAttr().Set(0.1)      # Low bounce
    print(f"    → Added friction (static=1.0, dynamic=0.9)")
    
    # Add PhysX material for better control
    if not collision_prim.GetPrim().HasAPI(PhysxSchema.PhysxMaterialAPI):
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(collision_prim.GetPrim())
    
    return True

def main():
    print("\n" + "="*70)
    print("F1TENTH - ADD WHEEL COLLISIONS & FRICTION")
    print("="*70 + "\n")
    
    # Create world
    print("Step 1: Creating world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    print("✓ World created\n")
    
    # Load robot
    print("Step 2: Loading robot...")
    add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/F1Tenth")
    print("✓ Robot loaded\n")
    
    # Get stage
    stage = my_world.stage
    
    # Add collision to all 4 wheels
    print("Step 3: Adding collision shapes and friction to wheels...")
    
    wheel_configs = [
        ("car_1_left_rear_wheel", 0.05, 0.045),
        ("car_1_right_rear_wheel", 0.05, 0.045),
        ("car_1_left_front_wheel", 0.05, 0.045),
        ("car_1_right_front_wheel", 0.05, 0.045),
    ]
    
    fixed_count = 0
    for wheel_name, radius, width in wheel_configs:
        wheel_path = f"/World/F1Tenth/{wheel_name}"
        if add_wheel_collision_and_friction(stage, wheel_path, radius, width):
            fixed_count += 1
    
    print(f"\n✓ Fixed {fixed_count}/4 wheels\n")
    
    # Create articulation
    print("Step 4: Creating articulation...")
    f1tenth = my_world.scene.add(
        Articulation(
            prim_path="/World/F1Tenth/car_1_base_link",
            name="f1tenth"
        )
    )
    print("✓ Articulation created\n")
    
    # Reset world
    print("Step 5: Resetting world...")
    my_world.reset()
    print("✓ World reset\n")
    
    print("="*70)
    print("TESTING MOVEMENT WITH COLLISION FIX")
    print("="*70 + "\n")
    
    print("🎬 Applying STRONG drive commands...\n")
    
    i = 0
    start_pos = None
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            position, _ = f1tenth.get_world_pose()
            
            if start_pos is None:
                start_pos = position.copy()
            
            # Apply STRONG velocity commands
            velocities = np.zeros(6)
            velocities[0] = 50.0  # Very fast
            velocities[2] = 50.0
            velocities[4] = 50.0
            velocities[5] = 50.0
            
            # Also apply torque
            efforts = np.zeros(6)
            efforts[0] = 100.0
            efforts[2] = 100.0
            efforts[4] = 100.0
            efforts[5] = 100.0
            
            f1tenth.set_joint_velocities(velocities)
            f1tenth.set_joint_efforts(efforts)
            
            # Straight steering
            positions = np.zeros(6)
            f1tenth.set_joint_positions(positions)
            
            # Check movement
            if i % 60 == 0:
                distance = np.linalg.norm(position - start_pos)
                actual_vels = f1tenth.get_joint_velocities()
                wheel_vel = actual_vels[0] if len(actual_vels) > 0 else 0
                
                moved = distance > 0.02  # Allow small threshold
                status = "✅ MOVING!" if moved else "❌ Still stuck"
                
                print(f"[{i//60:2d}s] Pos: [{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:5.2f}] | "
                      f"Moved: {distance:6.3f}m | WheelVel: {wheel_vel:6.1f} | {status}")
                
                if moved and i > 60:
                    print("\n" + "="*70)
                    print("🎉 SUCCESS! Wheels now have collision and friction!")
                    print("="*70)
                    print(f"Robot moved {distance:.3f} meters")
                    print("The collision fix worked! You can now use this for jump course")
                    print("="*70 + "\n")
            
            # Stop after 10 seconds
            if i > 600:
                distance = np.linalg.norm(position - start_pos)
                
                print("\n" + "="*70)
                print("TEST COMPLETE")
                print("="*70)
                print(f"Total distance: {distance:.4f} meters")
                
                if distance > 0.05:
                    print("\n✅ SUCCESS! Robot is moving with collision fix!")
                    print("\nNext steps:")
                    print("  1. This fix is temporary (in memory only)")
                    print("  2. To make permanent, save the modified stage")
                    print("  3. Or re-import URDF with better collision settings")
                else:
                    print("\n⚠️  Still not moving much")
                    print("\nThe USD file may have deeper issues.")
                    print("Consider re-importing URDF from scratch.")
                
                print("="*70 + "\n")
                break
            
            i += 1
    
    simulation_app.close()

if __name__ == "__main__":
    main()