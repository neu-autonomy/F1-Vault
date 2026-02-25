"""
Test DPhysics simulator directly
"""

import torch
from f1_vault.physics.physicscfg import DPhysConfig
from f1_vault.physics.physics import DPhysics

# Create simulator
cfg = DPhysConfig()
cfg.dt = 0.1
cfg.traj_sim_time = 1.0
cfg.control_mode = "throttle_steer"

sim = DPhysics(dphys_cfg=cfg, device="cuda")
sim.eval()

# Test with safe inputs
height = torch.zeros(1, 128, 128).cuda()
friction = torch.ones(1, 128, 128).cuda()
controls = torch.zeros(1, 10, 2).cuda()

# Initial state
x0 = torch.tensor([[0, 0, 0.2]]).cuda()
v0 = torch.zeros(1, 3).cuda()
R0 = torch.eye(3).unsqueeze(0).cuda()
w0 = torch.zeros(1, 3).cuda()

state0 = (x0, v0, R0, w0)

# Simulate
try:
    with torch.no_grad():
        output = sim(z_grid=height, controls=controls, state=state0, friction=friction)
    
    print("✓ Simulation succeeded!")
    print(f"  Output type: {type(output)}")
    
    # DPhysics returns tuple: (states, contact_info)
    if isinstance(output, tuple):
        print(f"  Output is tuple with {len(output)} elements")
        
        states = output[0]
        print(f"  States type: {type(states)}")
        
        # States is also a tuple: (positions, velocities, rotations, angular_vels)
        if isinstance(states, tuple):
            print(f"  States tuple length: {len(states)}")
            
            positions = states[0]  # (B, T, 3)
            velocities = states[1]  # (B, T, 3)
            rotations = states[2]   # (B, T, 3, 3)
            ang_vels = states[3]    # (B, T, 3)
            
            print(f"\n  Positions shape: {positions.shape}")
            print(f"  Velocities shape: {velocities.shape}")
            print(f"  Rotations shape: {rotations.shape}")
            print(f"  Angular vels shape: {ang_vels.shape}")
            
            print(f"\n  Initial position: {positions[0, 0]}")
            print(f"  Final position: {positions[0, -1]}")
            
            # Check for NaN/Inf
            has_nan = torch.isnan(positions).any()
            has_inf = torch.isinf(positions).any()
            
            if has_nan or has_inf:
                print(f"\n  ✗ Positions have NaN/Inf!")
            else:
                print(f"\n  ✓ Positions are finite")
    
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()