"""
Train the one-step CNN dynamics model.

Hard-won lessons baked into this script (see CLAUDE.md for the full story):

  - Boundary filter is REQUIRED. Rows where `terminated` or `truncated` is
    True have `next_states[i]` set to the post-reset spawn position (a
    teleport, up to ~8m), not the result of stepping the simulator. Train
    AND eval must both filter them, or MAE numbers disagree by 10x.
  - Target-only normalization: inputs are fed RAW, only the prediction
    targets (deltas) are normalized. The original variance-weighting
    collapsed to uniform weights and let high-variance dims dominate.
  - Elevation map is the LAST 676 dims = 26x26 (GridPatternCfg size=2.5,
    res=0.1 -> 26 samples/axis). Verified against the data 2026-06-13:
    the joint region ends at col 44 (720-44=676) and the 26x26 reshape is
    spatially smooth where 25x25 is sheared. NOTE: the H5 `elevation_map_size`
    attr was hardcoded to a WRONG 625 in files collected before 2026-06-13 --
    so do not blindly trust the attr on old files; the real map is 676.
    Checkpoints in models/cnn_dynamics* trained on the 625 slice used a
    misaligned map and should be retrained.
  - Naive baselines (mean |true delta|, i.e. predict-zero) are printed at
    eval so a silently-broken model shows up immediately.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics_model import DynamicsCNN, grid_from_size


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    data_file = "data/raw"   # file OR directory of batch .h5 files (all are concatenated)
    # Elevation layout is read from the H5 attrs in main(); these are fallbacks.
    # Real value is 676 (26x26). Old files carry a wrong 625 attr -- see header.
    elevation_map_size = 676
    elevation_grid_size = 26
    random_seed = 42

    core_state_dim = 22

    elevation_latent_dim = 128
    hidden_dims = [256, 256]
    dropout = 0.1

    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 100        # base model: 100 epochs (val still descending at 50, not overfitting);
                            # ReduceLROnPlateau already halves LR on plateau
    val_split = 0.2
    predict_delta = True

    # ---- Multi-step rollout training (roadmap item #1) --------------------
    # horizon=1 -> the original single-step trainer (unchanged, fully backward
    # compatible). horizon>1 -> unroll the model N steps, feeding its OWN predicted
    # core state back in while feeding the RECORDED elevation patch at each step
    # (never hallucinate terrain -- exactly how the controller uses the model), and
    # backprop through the whole unrolled window. This attacks exposure bias: the
    # single-step model is never trained on its own (slightly wrong) inputs, so error
    # compounds in the controller's multi-step rollout. See CLAUDE.md.
    horizon = 1
    stride = 1              # window stride when horizon>1 (1 = max overlap / most windows)
    # Scheduled sampling: probability of feeding the TRUE core (teacher forcing) instead
    # of the model's own prediction as the next step's input. Decays linearly from
    # tf_start (epoch 0) to tf_end (final epoch) so training eases from teacher-forced
    # toward fully self-fed. tf like 1->0 is the standard curriculum.
    scheduled_sampling = True
    tf_start = 1.0
    tf_end = 0.0

    use_normalization = True       # for downstream-tool compatibility
    use_loss_weighting = False
    handle_inf_elevation = True

    # v3 = first version trained on the correct 25x25 elevation layout.
    # (v2 and earlier checkpoints have a different decoder shape; don't mix.)
    save_dir = Path("models/cnn_dynamics_v3")
    checkpoint_every = 10

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")


# ============================================================================
# DATASET (target-only normalization)
# ============================================================================

class CNNDynamicsDataset(Dataset):
    def __init__(self, states, actions, next_states, config, target_stats=None):
        self.config = config

        if config.handle_inf_elevation:
            states = self._replace_inf(states.copy())
            next_states = self._replace_inf(next_states.copy())

        core_dim = config.core_state_dim
        elev_dim = config.elevation_map_size
        grid = config.elevation_grid_size

        self.core_states = states[:, :core_dim]
        self.elevation_maps = states[:, -elev_dim:].reshape(-1, grid, grid)
        self.next_core_states = next_states[:, :core_dim]
        self.next_elevation_maps = next_states[:, -elev_dim:].reshape(-1, grid, grid)
        self.actions = actions

        self.core_targets = self.next_core_states - self.core_states
        self.elevation_targets = self.next_elevation_maps - self.elevation_maps

        # Input stats (saved for downstream tools, not used in __getitem__)
        self.core_state_mean = self.core_states.mean(axis=0)
        self.core_state_std = self.core_states.std(axis=0) + 1e-8
        self.elevation_mean = self.elevation_maps.mean()
        self.elevation_std = self.elevation_maps.std() + 1e-8
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8

        # Target stats (used to normalize targets at __getitem__ time)
        if target_stats is None:
            self.core_target_mean = self.core_targets.mean(axis=0)
            self.core_target_std = self.core_targets.std(axis=0) + 1e-6
            self.elevation_target_mean = self.elevation_targets.mean()
            self.elevation_target_std = self.elevation_targets.std() + 1e-6
        else:
            self.core_target_mean = target_stats["core_target_mean"]
            self.core_target_std = target_stats["core_target_std"]
            self.elevation_target_mean = target_stats["elevation_target_mean"]
            self.elevation_target_std = target_stats["elevation_target_std"]

        # Compat fields for checkpoint format
        self.core_weights = np.ones(self.core_targets.shape[1])
        self.elevation_weight = 1.0

        print(f"\nDataset: {len(states):,} transitions")
        print(f"  Δcore mean[:3]: {self.core_target_mean[:3]}")
        print(f"  Δcore std[:3]:  {self.core_target_std[:3]}")
        print(f"  Δcore max abs[:3]: "
              f"{np.abs(self.core_targets[:, :3]).max(axis=0)}")

    def _replace_inf(self, data):
        elev = data[:, -self.config.elevation_map_size:]
        inf_mask = np.isinf(elev)
        if inf_mask.any():
            elev[inf_mask] = -10.0
            data[:, -self.config.elevation_map_size:] = elev
        return data

    def __len__(self):
        return len(self.core_states)

    def __getitem__(self, idx):
        core_state = self.core_states[idx].copy()
        elevation_map = self.elevation_maps[idx].copy()
        action = self.actions[idx].copy()

        core_target = (self.core_targets[idx]
                       - self.core_target_mean) / self.core_target_std
        elev_target = (self.elevation_targets[idx]
                       - self.elevation_target_mean) / self.elevation_target_std

        return (
            torch.FloatTensor(core_state),
            torch.FloatTensor(elevation_map[np.newaxis, :, :]),
            torch.FloatTensor(action),
            torch.FloatTensor(core_target),
            torch.FloatTensor(elev_target[np.newaxis, :, :]),
        )


# ============================================================================
# TRAIN / EVAL  (model lives in dynamics_model.py)
# ============================================================================

def loss_fn(pred_core, target_core, pred_elev, target_elev):
    core_loss = nn.functional.mse_loss(pred_core, target_core)
    elev_loss = nn.functional.mse_loss(pred_elev, target_elev)
    return core_loss + 0.1 * elev_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for core_states, elev_maps, actions, core_t, elev_t in loader:
        core_states = core_states.to(device)
        elev_maps = elev_maps.to(device)
        actions = actions.to(device)
        core_t = core_t.to(device)
        elev_t = elev_t.to(device)

        optimizer.zero_grad()
        pred_core, pred_elev = model(core_states, elev_maps, actions)
        loss = loss_fn(pred_core, core_t, pred_elev, elev_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(core_states)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, dataset):
    model.eval()
    total_loss = 0
    core_errors_phys, elev_errors_phys, true_deltas_phys = [], [], []

    core_t_mean = torch.from_numpy(dataset.core_target_mean).float().to(device)
    core_t_std = torch.from_numpy(dataset.core_target_std).float().to(device)
    elev_t_mean = float(dataset.elevation_target_mean)
    elev_t_std = float(dataset.elevation_target_std)

    with torch.no_grad():
        for core_states, elev_maps, actions, core_t, elev_t in loader:
            core_states = core_states.to(device)
            elev_maps = elev_maps.to(device)
            actions = actions.to(device)
            core_t = core_t.to(device)
            elev_t = elev_t.to(device)

            pred_core, pred_elev = model(core_states, elev_maps, actions)
            loss = loss_fn(pred_core, core_t, pred_elev, elev_t)
            total_loss += loss.item() * len(core_states)

            pred_core_phys = pred_core * core_t_std + core_t_mean
            target_core_phys = core_t * core_t_std + core_t_mean
            pred_elev_phys = pred_elev * elev_t_std + elev_t_mean
            target_elev_phys = elev_t * elev_t_std + elev_t_mean

            core_errors_phys.append((pred_core_phys - target_core_phys).cpu().numpy())
            elev_errors_phys.append((pred_elev_phys - target_elev_phys).cpu().numpy())
            true_deltas_phys.append(target_core_phys.cpu().numpy())

    return (total_loss / len(loader.dataset),
            np.concatenate(core_errors_phys),
            np.concatenate(elev_errors_phys),
            np.concatenate(true_deltas_phys))


# ============================================================================
# MULTI-STEP ROLLOUT TRAINING  (roadmap item #1)
# ============================================================================

class SequenceDynamicsDataset(Dataset):
    """Windows of consecutive same-episode transitions for N-step rollout training.

    Rows in the H5 are one-step transitions interleaved across envs; an episode is one
    (env_id, episode_id) group sorted by timestep, valid up to the first boundary row
    (terminated|truncated -> next_state is a post-reset teleport, see CLAUDE.md). This
    dataset reconstructs episodes, then slides length-N windows over each episode's
    contiguous non-boundary run.

    Each item is a window starting at reordered position `start`:
        core0       (22,)          initial core state (fed once)
        patches     (N+1, 26, 26)  RECORDED elevation patches for steps 0..N (raw, inf-filled)
        actions     (N, 2)         recorded actions for steps 0..N-1
        true_cores  (N, 22)        recorded core states at steps 1..N (rollout targets)

    Target/input normalization stats are computed ONCE over all valid single-step
    transitions (identical convention to the single-step trainer) and passed in, so a
    multi-step checkpoint's `stats` are drop-in compatible with dynamics_api / eval tools.
    """

    def __init__(self, states, actions, boundary, keys, timesteps, config, stats):
        self.config = config
        self.N = config.horizon
        core_dim = config.core_state_dim
        elev_dim = config.elevation_map_size
        grid = config.elevation_grid_size

        if config.handle_inf_elevation:
            states = self._replace_inf(states.copy())

        # Reorder rows into episode-contiguous order: sort by (key, timestep).
        order = np.lexsort((timesteps, keys))
        states = states[order]
        actions = actions[order]
        boundary = boundary[order]
        keys = keys[order]

        self.core_states = np.ascontiguousarray(states[:, :core_dim], dtype=np.float32)
        self.elevation_maps = np.ascontiguousarray(
            states[:, -elev_dim:].reshape(-1, grid, grid), dtype=np.float32)
        self.actions = np.ascontiguousarray(actions, dtype=np.float32)

        # Build valid windows. A window [start .. start+N] needs N transitions at
        # positions start..start+N-1 that are all non-boundary and stay within one
        # episode (contiguous key). Slide with `stride`.
        n = len(keys)
        same_ep = np.zeros(n, dtype=bool)
        same_ep[:-1] = keys[:-1] == keys[1:]
        # position p is a valid transition if same episode as p+1 and p is not a boundary
        valid_trans = same_ep & ~boundary
        starts = []
        N, stride = self.N, config.stride
        p = 0
        while p <= n - N:
            # need positions p..p+N-1 all valid transitions AND contiguous (same_ep chain)
            if valid_trans[p:p + N].all():
                starts.append(p)
                p += stride
            else:
                # jump past the first invalid transition in this span
                bad = p + int(np.argmin(valid_trans[p:p + N]))
                p = bad + 1
        self.starts = np.asarray(starts, dtype=np.int64)

        # Stats (shared, computed once from train transitions -- see main()).
        self.core_state_mean = stats["core_state_mean"]
        self.core_state_std = stats["core_state_std"]
        self.elevation_mean = stats["elevation_mean"]
        self.elevation_std = stats["elevation_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]
        self.core_target_mean = stats["core_target_mean"]
        self.core_target_std = stats["core_target_std"]
        self.elevation_target_mean = stats["elevation_target_mean"]
        self.elevation_target_std = stats["elevation_target_std"]
        self.core_weights = np.ones(core_dim)
        self.elevation_weight = 1.0

        print(f"\nSequence dataset (horizon={N}, stride={stride}): "
              f"{len(self.starts):,} windows from {n:,} rows")

    def _replace_inf(self, data):
        elev = data[:, -self.config.elevation_map_size:]
        inf_mask = np.isinf(elev)
        if inf_mask.any():
            elev[inf_mask] = -10.0
            data[:, -self.config.elevation_map_size:] = elev
        return data

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        N = self.N
        core0 = self.core_states[s]                       # (22,)
        patches = self.elevation_maps[s:s + N + 1]        # (N+1, 26, 26)
        actions = self.actions[s:s + N]                   # (N, 2)
        true_cores = self.core_states[s + 1:s + N + 1]    # (N, 22)
        return (
            torch.from_numpy(core0.copy()),
            torch.from_numpy(patches.copy()),
            torch.from_numpy(actions.copy()),
            torch.from_numpy(true_cores.copy()),
        )


def _rollout_batch(model, core0, patches, actions, dataset, device,
                   tf_prob, true_cores=None):
    """Unroll the model N steps on a batch, feeding recorded patches each step.

    Returns (core_loss_norm, elev_loss_norm, pred_cores) where pred_cores is
    (B, N, 22) physical. Uses target-only normalization (inputs raw, delta targets
    normalized) -- the only mode this project trains. When tf_prob>0 (scheduled
    sampling) each step's NEXT input is the recorded core with prob tf_prob, else the
    model's own prediction (backprop flows only through the self-fed branch).
    """
    ct_mean = torch.as_tensor(dataset.core_target_mean, dtype=torch.float32, device=device)
    ct_std = torch.as_tensor(dataset.core_target_std, dtype=torch.float32, device=device)
    et_mean = float(dataset.elevation_target_mean)
    et_std = float(dataset.elevation_target_std)

    B, N = core0.shape[0], actions.shape[1]
    cur = core0                                            # (B, 22) physical, raw input
    core_loss = core0.new_zeros(())
    elev_loss = core0.new_zeros(())
    preds = []
    for t in range(N):
        patch_t = patches[:, t].unsqueeze(1)               # (B,1,26,26) raw
        pred_core_norm, pred_elev_norm = model(cur, patch_t, actions[:, t])
        pred_delta = pred_core_norm * ct_std + ct_mean     # denorm -> physical delta
        next_core = cur + pred_delta                       # (B,22) physical
        preds.append(next_core.unsqueeze(1))

        if true_cores is not None:
            true_t = true_cores[:, t]
            core_loss = core_loss + (((next_core - true_t) / ct_std) ** 2).mean()
            true_elev_delta = patches[:, t + 1] - patches[:, t]
            target_elev_norm = (true_elev_delta - et_mean) / et_std
            elev_loss = elev_loss + ((pred_elev_norm.squeeze(1) - target_elev_norm) ** 2).mean()

        if t < N - 1:
            if true_cores is not None and tf_prob > 0:
                tf_mask = (torch.rand(B, 1, device=device) < tf_prob)
                cur = torch.where(tf_mask, true_cores[:, t], next_core)
            else:
                cur = next_core
    return core_loss / N, elev_loss / N, torch.cat(preds, dim=1)


def tf_prob_for_epoch(config, epoch):
    """Linear scheduled-sampling curriculum: tf_start -> tf_end over training."""
    if not config.scheduled_sampling:
        return 0.0
    if config.num_epochs <= 1:
        return config.tf_end
    frac = epoch / (config.num_epochs - 1)
    return config.tf_start + (config.tf_end - config.tf_start) * frac


def train_epoch_seq(model, loader, optimizer, device, dataset, tf_prob):
    model.train()
    total_loss = 0.0
    for core0, patches, actions, true_cores in loader:
        core0 = core0.to(device); patches = patches.to(device)
        actions = actions.to(device); true_cores = true_cores.to(device)
        optimizer.zero_grad()
        core_loss, elev_loss, _ = _rollout_batch(
            model, core0, patches, actions, dataset, device, tf_prob, true_cores)
        loss = core_loss + 0.1 * elev_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(core0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch_seq(model, loader, device, dataset):
    """Pure closed-loop (self-fed, tf_prob=0) N-step eval. Returns (loss, per-step
    physical |error| for X/Y/Z as (steps,3), and the predict-'hold initial' baseline)."""
    model.eval()
    total_loss = 0.0
    N = dataset.N
    step_abs_err = np.zeros((N, 3)); step_naive = np.zeros((N, 3)); count = 0
    for core0, patches, actions, true_cores in loader:
        core0 = core0.to(device); patches = patches.to(device)
        actions = actions.to(device); true_cores = true_cores.to(device)
        core_loss, elev_loss, preds = _rollout_batch(
            model, core0, patches, actions, dataset, device, 0.0, true_cores)
        total_loss += (core_loss + 0.1 * elev_loss).item() * len(core0)
        b = len(core0)
        err = (preds[:, :, :3] - true_cores[:, :, :3]).abs().cpu().numpy()   # (b,N,3)
        naive = (core0[:, None, :3] - true_cores[:, :, :3]).abs().cpu().numpy()
        step_abs_err += err.sum(axis=0); step_naive += naive.sum(axis=0); count += b
    return total_loss / len(loader.dataset), step_abs_err / count, step_naive / count


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    config = Config()
    if args.data_file:  config.data_file = args.data_file
    if args.epochs:     config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.save_dir:   config.save_dir = Path(args.save_dir)
    if args.horizon is not None: config.horizon = args.horizon
    if args.stride is not None:  config.stride = args.stride
    if args.seed is not None:    config.random_seed = args.seed
    if args.no_scheduled_sampling: config.scheduled_sampling = False
    if args.tf_end is not None:  config.tf_end = args.tf_end

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    print("=" * 80)
    print("CNN DYNAMICS TRAINING")
    print("=" * 80)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data (a single .h5 file OR a directory of batch files) ----
    data_path = Path(config.data_file)
    files = sorted(data_path.glob("*.h5")) if data_path.is_dir() else [data_path]
    if not files:
        raise FileNotFoundError(f"no .h5 files at {data_path}")
    print(f"\nLoading {len(files)} data file(s) from {data_path} ...")
    S, A, NS, TM, TR = [], [], [], [], []
    EV, EP, TS = [], [], []   # env_ids, episode_ids, timesteps (needed for horizon>1)
    action_dim, elev_attr = None, config.elevation_map_size
    for fp in files:
        with h5py.File(fp, "r") as f:
            S.append(f["states"][:]); A.append(f["actions"][:]); NS.append(f["next_states"][:])
            TM.append(f["terminated"][:]); TR.append(f["truncated"][:])
            EV.append(f["env_ids"][:]); EP.append(f["episode_ids"][:]); TS.append(f["timesteps"][:])
            action_dim = int(f.attrs["action_dim"])
            elev_attr = int(f.attrs.get("elevation_map_size", elev_attr))
            print(f"    {fp.name}: {len(S[-1]):,} rows")
    states = np.concatenate(S); actions = np.concatenate(A); next_states = np.concatenate(NS)
    terminated = np.concatenate(TM); truncated = np.concatenate(TR)
    env_ids = np.concatenate(EV); episode_ids = np.concatenate(EP); timesteps = np.concatenate(TS)
    del S, A, NS, TM, TR, EV, EP, TS
    config.elevation_map_size = elev_attr
    # The attr was hardcoded to a WRONG 625 in files collected before 2026-06-13.
    # The real map is 676 (26x26). Correct it so old files don't train on a sheared map.
    if config.elevation_map_size == 625:
        print("  WARNING: elevation_map_size attr is 625 (known-wrong) -- overriding to 676. "
              "Re-collect with current WheeledLab for a correct attr.")
        config.elevation_map_size = 676
    config.elevation_grid_size = grid_from_size(config.elevation_map_size)

    print(f"  raw rows: {len(states):,}")
    print(f"  elevation map: last {config.elevation_map_size} dims "
          f"({config.elevation_grid_size}x{config.elevation_grid_size}, from H5 attrs)")

    boundary = terminated | truncated
    valid = ~boundary
    n_total = len(states)
    print(f"  boundary rows (terminated|truncated): {int(boundary.sum()):,} / {n_total:,} "
          f"({100*boundary.mean():.1f}%)")

    is_seq = config.horizon > 1

    if not is_seq:
        # ================= SINGLE-STEP PATH (original, unchanged) =================
        states = states[valid]; actions = actions[valid]; next_states = next_states[valid]
        pos_delta = next_states[:, :3] - states[:, :3]
        print(f"  boundary filter -> {len(states):,} rows kept")
        print(f"  Δpos (filtered): "
              f"|X| mean={np.abs(pos_delta[:,0]).mean():.3f} max={np.abs(pos_delta[:,0]).max():.3f} | "
              f"|Y| mean={np.abs(pos_delta[:,1]).mean():.3f} max={np.abs(pos_delta[:,1]).max():.3f} | "
              f"|Z| mean={np.abs(pos_delta[:,2]).mean():.3f} max={np.abs(pos_delta[:,2]).max():.3f}")

        n = len(states)
        n_train = int(n * (1 - config.val_split))
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]
        print(f"\nSplit: {len(train_idx):,} train, {len(val_idx):,} val (by row)")

        train_dataset = CNNDynamicsDataset(
            states[train_idx], actions[train_idx], next_states[train_idx], config,
        )
        val_dataset = CNNDynamicsDataset(
            states[val_idx], actions[val_idx], next_states[val_idx], config,
            target_stats={
                "core_target_mean": train_dataset.core_target_mean,
                "core_target_std":  train_dataset.core_target_std,
                "elevation_target_mean": train_dataset.elevation_target_mean,
                "elevation_target_std":  train_dataset.elevation_target_std,
            },
        )
    else:
        # ================= MULTI-STEP (SEQUENCE) PATH (roadmap #1) =================
        # Split by EPISODE (not row) so no window leaks across the train/val boundary.
        keys = env_ids.astype(np.int64) * 1_000_000 + episode_ids.astype(np.int64)
        uniq = np.unique(keys)
        perm = np.random.permutation(len(uniq))
        n_val_ep = int(len(uniq) * config.val_split)
        val_keys = set(uniq[perm[:n_val_ep]].tolist())
        val_row = np.fromiter((k in val_keys for k in keys), dtype=bool, count=len(keys))
        train_row = ~val_row
        print(f"\nSplit by episode: {len(uniq)-n_val_ep:,} train / {n_val_ep:,} val episodes "
              f"({int(train_row.sum()):,}/{int(val_row.sum()):,} rows)")

        # Stats computed ONCE over TRAIN valid transitions (same convention as single-step,
        # so the checkpoint's stats stay drop-in compatible with dynamics_api / eval tools).
        tr = train_row & valid
        elev = config.elevation_map_size
        c = states[tr][:, :config.core_state_dim].astype(np.float32)
        nc = next_states[tr][:, :config.core_state_dim].astype(np.float32)
        ct = nc - c
        e_in = states[tr][:, -elev:].astype(np.float32)
        e_in[np.isinf(e_in)] = -10.0
        e_nx = next_states[tr][:, -elev:].astype(np.float32)
        e_nx[np.isinf(e_nx)] = -10.0
        et = e_nx - e_in
        stats = {
            "core_state_mean": c.mean(axis=0), "core_state_std": c.std(axis=0) + 1e-8,
            "elevation_mean": e_in.mean(), "elevation_std": e_in.std() + 1e-8,
            "action_mean": actions[tr].mean(axis=0), "action_std": actions[tr].std(axis=0) + 1e-8,
            "core_target_mean": ct.mean(axis=0), "core_target_std": ct.std(axis=0) + 1e-6,
            "elevation_target_mean": et.mean(), "elevation_target_std": et.std() + 1e-6,
        }
        del c, nc, ct, e_in, e_nx, et
        print(f"  Δcore std[:3] (target scale): {stats['core_target_std'][:3]}")

        train_dataset = SequenceDynamicsDataset(
            states[train_row], actions[train_row], boundary[train_row],
            keys[train_row], timesteps[train_row], config, stats)
        val_dataset = SequenceDynamicsDataset(
            states[val_row], actions[val_row], boundary[val_row],
            keys[val_row], timesteps[val_row], config, stats)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=(config.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=(config.device == "cuda"),
    )

    # ---- Model ----
    print("\nInitializing model...")
    model = DynamicsCNN(
        core_state_dim=config.core_state_dim,
        action_dim=action_dim,
        elevation_latent_dim=config.elevation_latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        elevation_grid=config.elevation_grid_size,
    ).to(config.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # ---- Train ----
    print(f"\nTraining on {config.device}...")
    print("=" * 80)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        if is_seq:
            tf_prob = tf_prob_for_epoch(config, epoch)
            train_loss = train_epoch_seq(
                model, train_loader, optimizer, config.device, train_dataset, tf_prob)
            val_loss, step_err, step_naive = eval_epoch_seq(
                model, val_loader, config.device, train_dataset)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, config.device)
            val_loss, val_core_err, val_elev_err, val_true_delta = eval_epoch(
                model, val_loader, config.device, train_dataset,
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if is_seq:
                N = config.horizon
                print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                      f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | tf={tf_prob:.2f}")
                print(f"          {N}-step closed-loop Z MAE (cm): "
                      f"step1={step_err[0,2]*100:.2f}  step{N}={step_err[-1,2]*100:.2f}  "
                      f"(naive-hold step{N}={step_naive[-1,2]*100:.2f})")
                print(f"          final-step XY MAE (cm): "
                      f"X={step_err[-1,0]*100:.2f}  Y={step_err[-1,1]*100:.2f}")
            else:
                mae_per_axis = np.abs(val_core_err[:, :3]).mean(axis=0)
                naive_per_axis = np.abs(val_true_delta[:, :3]).mean(axis=0)
                print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                      f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")
                print(f"          Pos MAE  (m):    "
                      f"X={mae_per_axis[0]:.4f}  Y={mae_per_axis[1]:.4f}  Z={mae_per_axis[2]:.4f}")
                print(f"          Naive MAE(m):    "
                      f"X={naive_per_axis[0]:.4f}  Y={naive_per_axis[1]:.4f}  Z={naive_per_axis[2]:.4f}  "
                      f"(predict-zero baseline)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config_dict": {
                    "core_state_dim": config.core_state_dim,
                    "elevation_latent_dim": config.elevation_latent_dim,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "elevation_map_size": config.elevation_map_size,
                    "predict_delta": config.predict_delta,
                    "use_normalization": True,
                    "use_loss_weighting": False,
                    "target_only_normalization": True,
                    "trained_with_boundary_filter": True,
                    # roadmap #1: >1 means this checkpoint was trained on N-step rollouts
                    # (recorded terrain fed back + scheduled sampling). 1 = single-step.
                    "train_horizon": config.horizon,
                    "scheduled_sampling": bool(is_seq and config.scheduled_sampling),
                },
                "stats": {
                    "core_state_mean": train_dataset.core_state_mean,
                    "core_state_std": train_dataset.core_state_std,
                    "elevation_mean": train_dataset.elevation_mean,
                    "elevation_std": train_dataset.elevation_std,
                    "action_mean": train_dataset.action_mean,
                    "action_std": train_dataset.action_std,
                    "core_target_mean": train_dataset.core_target_mean,
                    "core_target_std": train_dataset.core_target_std,
                    "elevation_target_mean": train_dataset.elevation_target_mean,
                    "elevation_target_std": train_dataset.elevation_target_std,
                    "core_weights": train_dataset.core_weights,
                    "elevation_weight": train_dataset.elevation_weight,
                },
            }, config.save_dir / "best_model.pt")

        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, config.save_dir / f"checkpoint_ep{epoch+1}.pt")

    # ---- Loss curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train", linewidth=2)
    ax.plot(val_losses, label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (normalized-MSE)")
    ax.set_title("CNN Training Progress")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.save_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ---- Final eval ----
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION (best checkpoint, val set) | horizon={config.horizon}")
    print("=" * 80)

    ckpt = torch.load(config.save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if is_seq:
        _, step_err, step_naive = eval_epoch_seq(model, val_loader, config.device, train_dataset)
        N = config.horizon
        print(f"\n  {N}-step closed-loop (self-fed, recorded terrain), per-step |error| vs "
              f"'hold initial' baseline")
        print(f"  step     X MAE      Y MAE      Z MAE     | Z naive")
        print(f"  " + "-" * 56)
        for t in range(N):
            print(f"  {t+1:>3}   {step_err[t,0]*100:7.2f}cm {step_err[t,1]*100:7.2f}cm "
                  f"{step_err[t,2]*100:7.2f}cm | {step_naive[t,2]*100:7.2f}cm")
        print(f"\n  (single-step Z ~ step1; the point of #1 is that step{N} stays low too — "
              f"that's reduced exposure bias vs a single-step model.)")
    else:
        _, final_core_err, _, final_true_delta = eval_epoch(
            model, val_loader, config.device, train_dataset,
        )
        print(f"\n  axis      model MAE      naive MAE       improvement")
        print(f"  " + "-" * 56)
        for i, lbl in enumerate(["X", "Y", "Z"]):
            m = np.abs(final_core_err[:, i]).mean()
            n_ = np.abs(final_true_delta[:, i]).mean()
            improvement = "BETTER" if m < n_ else "WORSE"
            ratio = m / n_ if n_ > 1e-9 else float("inf")
            print(f"  {lbl:>4}    {m*100:7.2f}cm     {n_*100:7.2f}cm     "
                  f"{improvement} ({ratio:.2f}x naive)")
        print(f"\nIf model MAE >= naive MAE on any axis, the model is no better than "
              f"predicting 'no motion' on that axis.")
    print(f"\nModel saved: {config.save_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN dynamics model")
    parser.add_argument("--data_file", type=str, help="H5 data file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--save_dir", type=str, help="Output directory")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Rollout-training horizon N (roadmap #1). 1 = single-step "
                             "(default/original). >1 = unroll N steps, feed predictions back "
                             "with recorded terrain + scheduled sampling.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Window stride for horizon>1 (default 1 = max overlap).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (vary this per ensemble member, roadmap #3).")
    parser.add_argument("--no_scheduled_sampling", action="store_true",
                        help="Disable scheduled sampling (pure self-fed rollout) for horizon>1.")
    parser.add_argument("--tf_end", type=float, default=None,
                        help="Final teacher-forcing prob for scheduled sampling (default 0.0).")
    args = parser.parse_args()
    main(args)