"""
F1Tenth Minimal Test - Just Load and Drive
Ultra-simple version to test basic functionality
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np

# Try different import patterns based on Isaac Sim version
try:
    # Isaac Sim 2023+
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.wheeled_robots.controllers import AckermannController
    print("✓ Using omni.isaac.core imports (Isaac Sim 2023+)")
except ImportError:
    # Older Isaac Sim
    try:
        from isaacsim.core.world import World
        from isaacsim.core.articulations import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.objects import DynamicCuboid
        from isaacsim.wheeled_robots.controllers import AckermannController
        print("✓ Using isaacsim imports (Older Isaac Sim)")
    except ImportError as e:
        print(f"ERROR: Could not import Isaac Sim modules: {e}")
        simulation_app.close()
        exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
USD_PATH = "/home/nail/Desktop/F1-Vault/urdf/f1-tenth_corrected/f1-tenth_corrected.usd"



# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("F1TENTH MINIMAL TEST")
    print("="*60 + "\n")
    
    # Create world
    print("Step 1: Creating world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    print("✓ World created\n")
    
    # Create a simple ramp
    print("Step 2: Creating ramp...")
    try:
        ramp = my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/Ramp",
                name="test_ramp",
                position=np.array([3.0, 0.0, 0.3]),
                scale=np.array([2.0, 1.5, 0.1]),
                color=np.array([1.0, 0.5, 0.0]),
                mass=50000.0
            )
        )
        # Tilt ramp 20 degrees
        angle_rad = np.radians(20)
        ramp.set_world_pose(
            position=np.array([3.0, 0.0, 0.3]),
            orientation=np.array([np.cos(angle_rad/2), 0, np.sin(angle_rad/2), 0])
        )
        print("✓ Ramp created\n")
    except Exception as e:
        print(f"Warning: Could not create ramp: {e}")
        print("Continuing without ramp...\n")
    
    # Load robot
    print("Step 3: Loading robot...")
    try:
        add_reference_to_stage(usd_path=USD_PATH, prim_path="/World/F1Tenth")
        print("✓ USD loaded\n")
    except Exception as e:
        print(f"ERROR: Could not load USD: {e}")
        print(f"Make sure file exists at: {USD_PATH}")
        simulation_app.close()
        return
    
    # Create articulation
    print("Step 4: Creating articulation...")
    try:
        f1tenth = my_world.scene.add(
            Articulation(
                prim_path="/World/F1Tenth/car_1_base_link",
                name="f1tenth"
            )
        )
        print("✓ Articulation created\n")
    except Exception as e:
        print(f"ERROR: Could not create articulation: {e}")
        print("Check if prim_path is correct for your URDF")
        simulation_app.close()
        return
    
    # Create controller
    print("Step 5: Creating controller...")
    try:
        ackermann = AckermannController(
            name="ackermann",
            wheel_base=0.325,
            track_width=0.2,
            wheel_radius=0.05
        )
        print("✓ Controller created\n")
    except Exception as e:
        print(f"Warning: Could not create controller: {e}")
        print("Will use direct joint control instead\n")
        ackermann = None
    
    # Reset
    print("Step 6: Resetting world...")
    my_world.reset()
    print("✓ World reset\n")
    
    print("="*60)
    print("ROBOT INFO")
    print("="*60)
    print(f"DOFs: {f1tenth.num_dof}")
    print(f"DOF Names: {f1tenth.dof_names}")
    position, _ = f1tenth.get_world_pose()
    print(f"Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
    print("="*60 + "\n")
    
    print("🎬 Starting simulation...")
    print("Robot will drive straight for 3 seconds, then loop\n")
    
    # Simple driving sequence
    i = 0
    phase = 0
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            position, _ = f1tenth.get_world_pose()
            
            # Simple 3-phase sequence
            if phase == 0:  # Drive straight
                steering = 0.0
                velocity = 2.0
                if i > 180:  # 3 seconds
                    phase = 1
                    i = 0
            elif phase == 1:  # Turn left
                steering = 0.3
                velocity = 1.5
                if i > 120:  # 2 seconds
                    phase = 2
                    i = 0
            else:  # Turn right
                steering = -0.3
                velocity = 1.5
                if i > 120:  # 2 seconds
                    phase = 0
                    i = 0
            
            # Apply control
            try:
                if ackermann is not None:
                    actions = ackermann.forward(command=[steering, velocity])
                    if actions is not None:
                        f1tenth.apply_action(actions)
                else:
                    # Fallback: manual joint control
                    # This is a simplified version
                    vel_array = np.zeros(f1tenth.num_dof)
                    if f1tenth.num_dof >= 2:
                        vel_array[0] = velocity * 10  # rear left
                        vel_array[1] = velocity * 10  # rear right
                    f1tenth.set_joint_velocities(vel_array)
            except Exception as e:
                if i == 0:
                    print(f"Control error: {e}")
            
            # Print status every second
            if i % 60 == 0:
                phase_name = ["Straight", "Left Turn", "Right Turn"][phase]
                print(f"Phase: {phase_name:12s} | "
                      f"Pos: [{position[0]:5.1f}, {position[1]:5.1f}, {position[2]:4.2f}] | "
                      f"Steering: {steering:5.2f} | Vel: {velocity:4.1f}")
            
            # Reset if robot falls
            if position[2] < -0.5:
                print("\n⚠️  Robot fell! Resetting...\n")
                my_world.reset()
                i = 0
                phase = 0
                continue
            
            i += 1
    
    print("\n🏁 Simulation ended")
    simulation_app.close()

if __name__ == "__main__":
    main()