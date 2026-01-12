from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False}) #Initiqalize the simulation app


import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Issac Sim assets folder")
    simulation_app.close()
    sys.exit(1)

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane() #adding ground plane 


# Initialize Ramps 


# Adding Car to world from urdf file 


#


wheel_base = 1.65
track_width = 1.25
wheel_radius = 0.25
desired_forward_vel = 1.1  # rad/s
desired_steering_angle = 0.1  # rad

# Setting acceleration, steering velocity, and dt to 0 to instantly reach the target steering and velocity
acceleration = 0.0  # m/s^2
steering_velocity = 0.0  # rad/s
dt = 0.0  # secs

controller = AckermannController(
   "test_controller", wheel_base=wheel_base, track_width=track_width, front_wheel_radius=wheel_radius
)

actions = controller.forward(
      [desired_steering_angle, steering_velocity, desired_forward_vel, acceleration, dt]
)


my_world.reset()

i = 0
reset_needed = False

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            controller.reset()
            reset_needed = False
        if i >= 0 and i < 1000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.1, 0]))
            print(my_jetbot.get_linear_velocity())
        elif i >= 1000 and i < 1300:
            # rotate
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
            print(my_jetbot.get_angular_velocity())
        elif i >= 1300 and i < 2000:
            # forward
            my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.1, 0]))
        elif i == 2000:
            i = 0
        i += 1
    if args.test is True:
        break


my_world.stop()
simulation_app.close()