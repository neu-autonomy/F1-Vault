"""
F1Tenth Jump Course - Simple Version
Clear ramp designs with pre-programmed driving sequences

Course Layout:
- 3 jump ramps at different positions
- 1 steep hill climb
- Circular driving pattern
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.articulations import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.objects import VisualCuboid, DynamicCuboid
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

# ============================================================================
# CONFIGURATION
# ============================================================================
URDF_PATH = "/home/nail/Desktop/F1-Vault/urdf/f1-tenth_corrected.urdf"
WHEEL_BASE = 0.325
TRACK_WIDTH = 0.2
WHEEL_RADIUS = 0.05

# ============================================================================
# RAMP CREATION FUNCTIONS
# ============================================================================
def create_ramp(world, name, position, length=2.0, width=1.5, angle_deg=25):
    """
    Create a launch ramp
    
    Args:
        world: Isaac Sim world
        name: Ramp name
        position: [x, y, z] base position
        length: Length of ramp
        width: Width of ramp
        angle_deg: Launch angle in degrees
    """
    # Create ramp base (flat approach)
    base = DynamicCuboid(
        prim_path=f"/World/Ramps/{name}_base",
        name=f"{name}_base",
        position=np.array([position[0] - length*0.6, position[1], position[2] + 0.05]),
        scale=np.array([length*0.5, width, 0.1]),
        color=np.array([0.3, 0.3, 0.3]),
        mass=50000.0
    )
    world.scene.add(base)
    
    # Create angled ramp section
    angle_rad = np.radians(angle_deg)
    ramp_height = length * np.sin(angle_rad) / 2
    
    ramp = DynamicCuboid(
        prim_path=f"/World/Ramps/{name}_ramp",
        name=f"{name}_ramp",
        position=np.array([position[0], position[1], position[2] + ramp_height/2]),
        scale=np.array([length, width, 0.15]),
        color=np.array([1.0, 0.4, 0.0]),  # Orange
        mass=50000.0
    )
    world.scene.add(ramp)
    
    # Tilt the ramp
    qw = np.cos(angle_rad / 2)
    qy = np.sin(angle_rad / 2)
    ramp.set_local_pose(
        translation=np.array([position[0], position[1], position[2] + ramp_height/2]),
        orientation=np.array([qw, 0, qy, 0])
    )
    
    print(f"  ✓ Ramp '{name}' created at {position} with {angle_deg}° angle")
    return base, ramp

def create_landing_zone(world, name, position, size=3.0):
    """Create a visual landing zone marker"""
    landing = VisualCuboid(
        prim_path=f"/World/Ramps/{name}_landing",
        name=f"{name}_landing",
        position=np.array([position[0], position[1], position[2] + 0.01]),
        scale=np.array([size, size, 0.02]),
        color=np.array([0.2, 0.8, 0.2])  # Green
    )
    world.scene.add(landing)
    return landing

def create_hill(world, name, position, size=[4.0, 3.0, 1.5]):
    """Create a hill obstacle"""
    hill = DynamicCuboid(
        prim_path=f"/World/Ramps/{name}",
        name=name,
        position=np.array([position[0], position[1], position[2] + size[2]/2]),
        scale=np.array(size),
        color=np.array([0.5, 0.3, 0.2]),  # Brown
        mass=50000.0
    )
    world.scene.add(hill)
    
    # Smooth the top
    angle_rad = np.radians(20)
    qw = np.cos(angle_rad / 2)
    qy = np.sin(angle_rad / 2)
    
    print(f"  ✓ Hill '{name}' created at {position}")
    return hill

# ============================================================================
# DRIVING SEQUENCE CONTROLLER
# ============================================================================
class SequenceController:
    """Simple timed sequence controller"""
    
    def __init__(self, sequences):
        """
        Args:
            sequences: List of (duration_sec, steering, velocity) tuples
        """
        self.sequences = sequences
        self.current_seq = 0
        self.seq_timer = 0.0
        self.fps = 60.0
    
    def update(self):
        """Update and get current command"""
        if self.current_seq >= len(self.sequences):
            # Loop back
            self.current_seq = 0
            self.seq_timer = 0.0
        
        duration, steering, velocity = self.sequences[self.current_seq]
        frames = duration * self.fps
        
        if self.seq_timer >= frames:
            self.current_seq += 1
            self.seq_timer = 0.0
            return self.update()
        
        self.seq_timer += 1
        return steering, velocity
    
    def get_phase_name(self):
        """Get current phase description"""
        if self.current_seq >= len(self.sequences):
            return "Reset"
        duration, steering, velocity = self.sequences[self.current_seq]
        
        if velocity > 2.5:
            return "🚀 BOOST!"
        elif velocity > 1.5:
            return "Fast"
        elif abs(steering) > 0.2:
            return "Turning"
        else:
            return "Cruising"

# ============================================================================
# MAIN SIMULATION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("   🏁 F1TENTH JUMP COURSE - AUTONOMOUS STUNT DRIVING 🏁")
    print("="*70 + "\n")
    
    # Create world
    print("📦 Creating world...")
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    print("   ✓ World created\n")
    
    # Build the course
    print("🏗️  Building jump course...\n")
    
    # Ramp 1: Easy jump
    print("  Ramp 1: Easy Jump")
    create_ramp(my_world, "ramp1", [6.0, 0.0, 0.0], length=2.5, angle_deg=20)
    create_landing_zone(my_world, "landing1", [10.0, 0.0, 0.0])
    
    # Ramp 2: Medium jump (to the side)
    print("  Ramp 2: Medium Jump")
    create_ramp(my_world, "ramp2", [3.0, -8.0, 0.0], length=3.0, angle_deg=28)
    create_landing_zone(my_world, "landing2", [7.0, -8.0, 0.0])
    
    # Ramp 3: Big jump
    print("  Ramp 3: Big Jump")
    create_ramp(my_world, "ramp3", [-6.0, -3.0, 0.0], length=3.5, angle_deg=30)
    create_landing_zone(my_world, "landing3", [-1.0, -3.0, 0.0])
    
    # Hill obstacle
    print("  Hill: Climb Challenge")
    create_hill(my_world, "hill1", [-8.0, 6.0, 0.0], size=[3.0, 3.0, 1.2])
    
    print("\n   ✓ Course complete!\n")
    
    # Load robot
    print("🏎️  Loading F1Tenth robot...")
    add_reference_to_stage(usd_path=URDF_PATH, prim_path="/World/F1Tenth")
    
    f1tenth = Articulation(
        prim_path="/World/F1Tenth/car_1_base_link",
        name="f1tenth"
    )
    my_world.scene.add(f1tenth)
    print("   ✓ Robot loaded\n")
    
    # Create controller
    ackermann = AckermannController(
        name="ackermann",
        wheel_base=WHEEL_BASE,
        track_width=TRACK_WIDTH,
        front_wheel_radius=WHEEL_RADIUS
    )
    
    # Define driving sequence
    # Format: (duration_seconds, steering_angle, velocity)
    driving_sequence = [
        # Start and approach ramp 1
        (2.0,  0.0,  1.5),   # Drive straight slowly
        (1.5,  0.0,  3.5),   # BOOST to ramp 1!
        (1.0,  0.0,  2.0),   # Landing
        
        # Turn toward ramp 2
        (2.0, -0.4,  1.8),   # Turn right
        (1.0, -0.3,  2.0),   # Continue turn
        (1.0,  0.0,  1.5),   # Straighten
        
        # Approach ramp 2
        (1.0,  0.0,  2.5),   # Speed up
        (1.2,  0.0,  4.0),   # BOOST to ramp 2!
        (1.0,  0.0,  2.0),   # Landing
        
        # Turn back
        (2.0,  0.5,  1.8),   # Turn left
        (1.5,  0.4,  2.0),   # Continue turn
        
        # Approach ramp 3
        (1.5,  0.1,  2.0),   # Slight adjustment
        (1.0,  0.0,  3.0),   # Speed up
        (1.5,  0.0,  4.5),   # MEGA BOOST to ramp 3!
        (1.5,  0.0,  2.0),   # Landing
        
        # Navigate to hill
        (2.0,  0.3,  2.0),   # Turn toward hill
        (2.0,  0.2,  2.5),   # Approach hill
        (2.0,  0.0,  3.0),   # Climb hill!
        
        # Return to start
        (2.0, -0.3,  2.0),   # Turn back
        (3.0, -0.2,  2.0),   # Navigate home
        (2.0,  0.0,  1.5),   # Slow down
    ]
    
    sequence = SequenceController(driving_sequence)
    
    # Set camera
    set_camera_view(
        eye=np.array([0, -20, 10]),
        target=np.array([0, 0, 0]),
        camera_prim_path="/OmniverseKit_Persp"
    )
    
    # Reset world
    my_world.reset()
    
    print("="*70)
    print("🎬 STARTING SIMULATION")
    print("="*70)
    print(f"Course: {len(driving_sequence)} phases")
    print(f"Robot DOFs: {f1tenth.num_dof}")
    print("="*70 + "\n")
    
    # Simulation loop
    i = 0
    lap = 1
    max_height = 0.0
    air_time_frames = 0
    
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if my_world.is_playing():
            # Get robot state
            position, _ = f1tenth.get_world_pose()
            velocity = f1tenth.get_linear_velocity()
            speed = np.linalg.norm(velocity)
            
            # Track statistics
            if position[2] > max_height:
                max_height = position[2]
            
            if position[2] > 0.2:  # Airborne threshold
                air_time_frames += 1
            
            # Get command from sequence
            steering, target_vel = sequence.update()
            
            # Calculate and apply actions
            actions = ackermann.forward(
                command=[steering, 0.0, target_vel, 0.0, 1/60.0]
            )
            
            if actions is not None:
                f1tenth.apply_action(actions)
            
            # Update camera to follow
            if i % 10 == 0:
                set_camera_view(
                    eye=np.array([position[0] - 8, position[1] - 12, 8]),
                    target=position,
                    camera_prim_path="/OmniverseKit_Persp"
                )
            
            # Print telemetry every second
            if i % 60 == 0:
                phase = sequence.get_phase_name()
                airborne = "✈️  FLYING!" if position[2] > 0.2 else ""
                
                print(f"[Lap {lap}] Phase: {phase:10s} | "
                      f"Pos: [{position[0]:5.1f}, {position[1]:5.1f}, {position[2]:4.2f}] | "
                      f"Speed: {speed:4.1f} m/s | "
                      f"Max H: {max_height:4.2f}m {airborne}")
            
            # Reset if robot flips or completes sequence
            if position[2] < -0.3 or sequence.current_seq >= len(driving_sequence):
                air_time_sec = air_time_frames / 60.0
                print(f"\n{'='*70}")
                print(f"LAP {lap} COMPLETE!")
                print(f"Max jump height: {max_height:.2f}m")
                print(f"Total air time: {air_time_sec:.1f}s")
                print(f"{'='*70}\n")
                
                my_world.reset()
                ackermann.reset()
                sequence.current_seq = 0
                sequence.seq_timer = 0.0
                lap += 1
                i = 0
                max_height = 0.0
                air_time_frames = 0
                continue
            
            i += 1
    
    # Cleanup
    print("\n🏁 Simulation ended")
    my_world.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()