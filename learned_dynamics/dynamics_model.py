"""
Shared architecture + checkpoint/rollout helpers for the learned-dynamics work.

Single source of truth: train_cnn.py, sanity_check.py, action_conditioning_test.py
and visualize_trajectories.py all import from here. (Previously each script
carried its own copy of these classes, which had to be kept in sync by hand.)

Checkpoint compatibility: the grid size is read from the checkpoint's
config_dict, so both old checkpoints (elevation_map_size=676, 26x26 -- trained
before we noticed the H5 file stores a 625-value 25x25 map) and new ones
(625, 25x25) load through `load_model`.
"""

import numpy as np
import torch
import torch.nn as nn

DT = 0.1          # control timestep, matches H5 attrs["control_dt"]
INF_FILL = -10.0  # replacement for +/-inf values in elevation maps


def grid_from_size(elevation_map_size):
    grid = int(round(elevation_map_size ** 0.5))
    if grid * grid != elevation_map_size:
        raise ValueError(f"elevation_map_size={elevation_map_size} is not a perfect square")
    return grid


# ============================================================================
# ARCHITECTURE
# ============================================================================

class ElevationEncoder(nn.Module):
    """Heightmap (any grid size) -> latent vector. Fully conv + adaptive pool."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(128, latent_dim)

    def forward(self, x):
        return self.projection(self.encoder(x))


class ElevationDecoder(nn.Module):
    """Latent vector -> grid_size x grid_size heightmap delta.

    The two ConvTranspose layers produce 28x28; the final valid (padding=0)
    conv crops to the target grid: kernel 3 -> 26x26 (old checkpoints),
    kernel 4 -> 25x25 (current data layout).
    """

    def __init__(self, latent_dim=128, grid_size=26):
        super().__init__()
        final_kernel = 29 - grid_size
        if not 1 <= final_kernel <= 28:
            raise ValueError(f"unsupported grid_size={grid_size}")
        self.projection = nn.Linear(latent_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, final_kernel, padding=0),
        )

    def forward(self, x):
        return self.decoder(self.projection(x))


class DynamicsCNN(nn.Module):
    """One-step dynamics: (core_state, elevation_map, action) -> (Δcore, Δelevation)."""

    def __init__(self, core_state_dim, action_dim, elevation_latent_dim,
                 hidden_dims, dropout=0.1, elevation_grid=26):
        super().__init__()
        self.core_state_dim = core_state_dim
        self.elevation_encoder = ElevationEncoder(elevation_latent_dim)

        layers, prev = [], core_state_dim + action_dim + elevation_latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, core_state_dim + elevation_latent_dim))
        self.dynamics = nn.Sequential(*layers)
        self.elevation_decoder = ElevationDecoder(elevation_latent_dim, elevation_grid)

    def forward(self, core_state, elevation_map, action):
        elev_latent = self.elevation_encoder(elevation_map)
        combined = torch.cat([core_state, action, elev_latent], dim=-1)
        prediction = self.dynamics(combined)
        pred_core = prediction[:, :self.core_state_dim]
        pred_elev_latent = prediction[:, self.core_state_dim:]
        pred_elevation = self.elevation_decoder(pred_elev_latent)
        return pred_core, pred_elevation


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_model(checkpoint_path, device):
    """Load a training checkpoint. Returns (model, config_dict, stats, core_dim)."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt["config_dict"]
    stats = ckpt["stats"]
    core_state_dim = len(stats["core_state_mean"])

    model = DynamicsCNN(
        core_state_dim=core_state_dim,
        action_dim=len(stats["action_mean"]),
        elevation_latent_dim=config["elevation_latent_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        elevation_grid=grid_from_size(config["elevation_map_size"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, stats, core_state_dim


def normalization_mode(config):
    """Return (input_norm, output_denorm) flags from the checkpoint config.

    - target_only_normalization=True  -> inputs RAW, outputs denormalized
    - use_normalization=True (legacy) -> inputs normalized, outputs denormalized
    - everything False                -> no normalization anywhere
    """
    if config.get("target_only_normalization", False):
        return False, True
    elif config.get("use_normalization", False):
        return True, True
    return False, False


# ============================================================================
# STATE HANDLING
# ============================================================================

def replace_inf_elevation(states, elev_size, fill=INF_FILL):
    """Replace +/-inf in the elevation slice of a (N, state_dim) array. In place."""
    elev = states[:, -elev_size:]
    inf_mask = np.isinf(elev)
    if inf_mask.any():
        elev[inf_mask] = fill
        states[:, -elev_size:] = elev
    return states


def split_state(full_state, core_dim, elev_size):
    grid = grid_from_size(elev_size)
    core = full_state[:core_dim]
    middle = full_state[core_dim:-elev_size]
    elevation = full_state[-elev_size:].reshape(grid, grid)
    return core, middle, elevation


def assemble_state(core, middle, elevation_2d):
    return np.concatenate([core, middle, elevation_2d.flatten()])


# ============================================================================
# CLOSED-LOOP ROLLOUT
# ============================================================================

def synthetic_rollout(model, initial_state, action_seq, config, stats,
                      core_dim, device):
    """Roll the model out closed-loop from one state under an action sequence.

    Returns (len(action_seq)+1, state_dim) array; row 0 is the initial state.
    The non-core, non-elevation dims are held constant (the model doesn't
    predict them).
    """
    elev_size = config["elevation_map_size"]
    predict_delta = config["predict_delta"]
    input_norm, output_denorm = normalization_mode(config)

    N = len(action_seq)
    rollout = np.zeros((N + 1, len(initial_state)), dtype=np.float32)
    rollout[0] = initial_state
    current = initial_state.copy()
    replace_inf_elevation(current[np.newaxis, :], elev_size)

    with torch.no_grad():
        for t in range(N):
            core, middle, elev_2d = split_state(current, core_dim, elev_size)
            action = action_seq[t]

            if input_norm:
                core_use = (core - stats["core_state_mean"]) / stats["core_state_std"]
                elev_use = (elev_2d - stats["elevation_mean"]) / stats["elevation_std"]
                act_use = (action - stats["action_mean"]) / stats["action_std"]
            else:
                core_use, elev_use, act_use = core, elev_2d, action

            ct = torch.from_numpy(core_use.astype(np.float32)).unsqueeze(0).to(device)
            et = torch.from_numpy(elev_use.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            at = torch.from_numpy(act_use.astype(np.float32)).unsqueeze(0).to(device)
            pc, pe = model(ct, et, at)
            pc = pc.squeeze(0).cpu().numpy()
            pe = pe.squeeze(0).squeeze(0).cpu().numpy()

            if output_denorm:
                pc = pc * stats["core_target_std"] + stats["core_target_mean"]
                pe = pe * stats["elevation_target_std"] + stats["elevation_target_mean"]

            if predict_delta:
                new_core = core + pc
                new_elev = elev_2d + pe
            else:
                new_core = pc
                new_elev = pe

            current = assemble_state(new_core, middle, new_elev)
            rollout[t + 1] = current

    return rollout
