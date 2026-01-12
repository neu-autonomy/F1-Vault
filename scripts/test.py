"""
Simple Test Vehicle
Creates a basic car from primitives to test if physics works at all
If this moves, the problem is definitely in your USD file
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from pxr import UsdPhysics, PhysxSchema, Gf

def main():
    print("\n" + "="*70)
    print("SIMPLE TEST VEHICLE - PHYSICS VERIFICATION")
    print("="*70 + "\n")
    
    print("Creating a simple box + sphere wheels from scratch...")
    print("If THIS moves, your F1Tenth USD file is the problem\n")
    
    # Create world
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    
    # Create chassis (box)
    print("Creating chassis...")
    chassis = my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestCar/chassis",
            name="chassis",
            position=np.array([0, 0, 0.3]),
            scale=np.array([0.5, 0.3, 0.15]),
            color=np.array([0.8, 0.2, 0.2]),
            mass=10.0
        )
    )
    
    # Create 4 wheels (spheres for simplicity)
    print("Creating wheels with friction...")
    wheel_positions = [
        ("wheel_fl", [ 0.3,  0.25, 0.15]),  # Front left
        ("wheel_fr", [ 0.3, -0.25, 0.15]),  # Front right
        ("wheel_rl", [-0.3,  0.25, 0.15]),  # Rear left
        ("wheel_rr", [-0.3, -0.25, 0.15]),  # Rear right
    ]
    
    wheels = []
    for name, pos in wheel_positions:
        wheel = my_world.scene.add(
            DynamicSphere(
                prim_path=f"/World/TestCar/{name}",
                name=name,
                position=np.array(pos),
                radius=0.1,
                color=np.array([0.1, 0.1, 0.1]),
                mass=0.5
            )
        )
        
        # Add HIGH friction to wheels
        stage = my_world.stage
        wheel_prim = stage.GetPrimAtPath(f"/World/TestCar/{name}")
        
        if not wheel_prim.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI.Apply(wheel_prim)
        else:
            mat = UsdPhysics.MaterialAPI(wheel_prim)
        
        mat.CreateStaticFrictionAttr().Set(2.0)   # VERY high friction
        mat.CreateDynamicFrictionAttr().Set(1.5)
        
        wheels.append(wheel)
    
    print("✓ Test vehicle created\n")
    
    # Reset
    my_world.reset()
    
    print("="*70)
    print("PHYSICS TEST")
    print("="*70)
    print("\nApplying constant forward force to chassis...")
    print("If physics works, the box should slide forward\n")
    
    i = 0
    start_pos = None
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            position, _ = chassis.get_world_pose()
            
            if start_pos is None:
                start_pos = position.copy()
            
            # Apply forward force
            chassis.apply_force(force=np.array([50.0, 0.0, 0.0]))  # Push forward
            
            if i % 60 == 0:
                distance = np.linalg.norm(position - start_pos)
                moved = distance > 0.05
                status = "✅ PHYSICS WORKS!" if moved else "❌ Stuck"
                
                print(f"[{i//60:2d}s] Pos: [{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:5.2f}] | "
                      f"Moved: {distance:6.3f}m | {status}")
            
            if i > 300:  # 5 seconds
                distance = np.linalg.norm(position - start_pos)
                
                print("\n" + "="*70)
                print("RESULT")
                print("="*70)
                
                if distance > 0.1:
                    print(f"\n✅ Physics works! Box moved {distance:.3f} meters")
                    print("\nThis means:")
                    print("  → Isaac Sim physics is working fine")
                    print("  → Your F1Tenth USD file is the problem")
                    print("  → The collision geometry in USD is broken")
                    print("\nSolution:")
                    print("  1. Try: f1tenth_add_collisions.py")
                    print("  2. Or re-import URDF with fixed settings")
                else:
                    print(f"\n❌ Physics not working! Box only moved {distance:.4f}m")
                    print("\nThis means:")
                    print("  → Something wrong with Isaac Sim setup")
                    print("  → Check if PhysX is enabled")
                    print("  → Try restarting Isaac Sim")
                
                print("="*70 + "\n")
                break
            
            i += 1
    
    simulation_app.close()

if __name__ == "__main__":
    main()