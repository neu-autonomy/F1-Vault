"""
loss.py

Physics-informed (and optionally supervised) losses for F1-Vault.

The repository's `TerrainEncoder` predicts dense per-cell terrain properties from a
bird's-eye-view (BEV) elevation map. The predicted properties are continuous
2D grids, so the base task is *dense regression* (a "segmentation-like" output,
but with continuous targets).

This module provides a loss that can be used in two regimes:

1) Physics-informed / self-supervised:
   - Use a differentiable simulator (e.g., DPhysics) to roll out a trajectory
     under the sample's control inputs and compare it to the observed trajectory.

2) Supervised (for debugging or synthetic data):
   - If ground-truth terrain property maps are available, compare predicted maps
     to targets directly.

The loss is intentionally written to be simulator-agnostic: any callable that
returns a trajectory tensor (or a dict/tuple containing it) can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _wrap_to_pi(angle_rad: Tensor) -> Tensor:
    """Wrap angles to [-pi, pi]."""
    two_pi = 2.0 * math.pi
    return (angle_rad + math.pi) - torch.floor((angle_rad + math.pi) / two_pi) * two_pi - math.pi


def quat_wxyz_to_yaw(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert quaternion(s) (w, x, y, z) to yaw angle(s) in radians."""
    assert q.shape[-1] == 4, f"Expected (...,4) quaternion, got {tuple(q.shape)}"
    qw, qx, qy, qz = q.unbind(dim=-1)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp + eps)


def rotmat_to_yaw(R: Tensor, eps: float = 1e-8) -> Tensor:
    """Extract yaw from rotation matrix (...,3,3)."""
    assert R.shape[-2:] == (3, 3), f"Expected (...,3,3), got {tuple(R.shape)}"
    return torch.atan2(R[..., 1, 0], R[..., 0, 0] + eps)


def quat_wxyz_to_rotmat(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert quaternion(s) (w,x,y,z) to rotation matrix (...,3,3)."""
    assert q.shape[-1] == 4
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def total_variation_2d(x: Tensor, reduction: str = "mean") -> Tensor:
    """Total variation loss for (B,H,W) or (B,C,H,W)."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.dim() == 4, f"Expected (B,C,H,W) or (B,H,W), got {tuple(x.shape)}"

    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    tv = dx.abs().mean(dim=(-3, -2, -1)) + dy.abs().mean(dim=(-3, -2, -1))

    if reduction == "none":
        return tv
    if reduction == "sum":
        return tv.sum()
    if reduction == "mean":
        return tv.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


@dataclass
class LossWeights:
    traj_pos: float = 1.0
    traj_yaw: float = 0.1
    terrain_supervised: float = 0.0
    height_consistency: float = 0.05
    tv: float = 1e-4
    stiffness_prior: float = 0.0
    damping_prior: float = 0.0
    friction_prior: float = 0.0


class PhysicsInformedTerrainLoss(nn.Module):
    """
    Combined loss for terrain learning.

    - Physics-informed mode (if simulator and gt_trajectory are provided)
    - Optional supervised terrain loss (if terrain_gt is provided)
    - Regularizers: height consistency, TV, soft priors
    """

    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        pose_loss: str = "huber",
        huber_delta: float = 1.0,
        supervisor_reduction: str = "mean",
    ):
        super().__init__()
        self.weights = weights
        self.pose_loss = pose_loss.lower()
        self.huber_delta = float(huber_delta)
        self.supervisor_reduction = supervisor_reduction

    def _pointwise_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.pose_loss == "mse":
            return F.mse_loss(pred, target, reduction="mean")
        if self.pose_loss in ("l1", "mae"):
            return F.l1_loss(pred, target, reduction="mean")
        if self.pose_loss in ("smoothl1", "smooth_l1"):
            return F.smooth_l1_loss(pred, target, reduction="mean", beta=self.huber_delta)
        if self.pose_loss == "huber":
            return F.huber_loss(pred, target, reduction="mean", delta=self.huber_delta)
        raise ValueError(f"Unknown pose_loss: {self.pose_loss}")

    def _extract_controls(self, batch: Dict[str, Any]) -> Tensor:
        if "controls" in batch:
            return batch["controls"]
        if "actions" in batch:
            return batch["actions"]
        raise KeyError("Batch must include 'controls' or 'actions' with shape (B,T,2).")

    def _extract_gt_pose(self, batch: Dict[str, Any]) -> Tensor:
        if "gt_trajectory" in batch:
            return batch["gt_trajectory"]
        raise KeyError("Batch must include 'gt_trajectory' with shape (B,T,7).")

    def _extract_initial_pose(self, batch: Dict[str, Any]) -> Optional[Tensor]:
        if "initial_pose" in batch:
            return batch["initial_pose"]
        return None

    def _default_state0_from_pose(self, initial_pose: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x0 = initial_pose[:, 0:3]
        q0 = initial_pose[:, 3:7]
        R0 = quat_wxyz_to_rotmat(q0)
        zeros = torch.zeros_like(x0)
        return (x0, zeros, R0, zeros)

    def forward(
        self,
        preds: Dict[str, Tensor],
        batch: Dict[str, Any],
        simulator: Optional[Callable[..., Any]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        logs: Dict[str, float] = {}

        for k in ("height", "stiffness", "damping", "friction"):
            if k not in preds:
                raise KeyError(f"Model output is missing key '{k}'")

        height = preds["height"]
        stiffness = preds["stiffness"]
        damping = preds["damping"]
        friction = preds["friction"]

        total = torch.zeros((), device=height.device, dtype=height.dtype)

        # Optional supervised terrain targets (debug/synthetic)
        if "terrain_gt" in batch and isinstance(batch["terrain_gt"], dict):
            gt = batch["terrain_gt"]
            sup_terms = []
            for k in ("height", "stiffness", "damping", "friction"):
                if k in gt:
                    sup_terms.append(F.mse_loss(preds[k], gt[k], reduction=self.supervisor_reduction))
            if sup_terms:
                sup = torch.stack(sup_terms).mean()
                total = total + self.weights.terrain_supervised * sup
                logs["loss/terrain_supervised"] = float(sup.detach().cpu())
            else:
                logs["loss/terrain_supervised"] = 0.0

        # Height consistency to input BEV
        if "bev_map" in batch:
            bev = batch["bev_map"]
            if bev.dim() == 4 and bev.shape[1] == 1:
                bev2d = bev[:, 0]
                hc = F.mse_loss(height, bev2d, reduction="mean")
                total = total + self.weights.height_consistency * hc
                logs["loss/height_consistency"] = float(hc.detach().cpu())

        # Smoothness / TV
        if self.weights.tv > 0:
            tv = (
                total_variation_2d(height)
                + total_variation_2d(stiffness)
                + total_variation_2d(damping)
                + total_variation_2d(friction)
            ) / 4.0
            total = total + self.weights.tv * tv
            logs["loss/tv"] = float(tv.detach().cpu())

        # Optional priors
        if self.weights.stiffness_prior > 0:
            prior = torch.log(stiffness.clamp_min(1e-6)).pow(2).mean()
            total = total + self.weights.stiffness_prior * prior
            logs["loss/stiffness_prior"] = float(prior.detach().cpu())

        if self.weights.damping_prior > 0:
            prior = torch.log(damping.clamp_min(1e-6)).pow(2).mean()
            total = total + self.weights.damping_prior * prior
            logs["loss/damping_prior"] = float(prior.detach().cpu())

        if self.weights.friction_prior > 0:
            fr = (friction / 2.0).clamp(1e-4, 1 - 1e-4)
            prior = (-(fr * torch.log(fr) + (1 - fr) * torch.log(1 - fr))).mean()
            total = total + self.weights.friction_prior * prior
            logs["loss/friction_prior"] = float(prior.detach().cpu())

        # Physics-informed trajectory loss
        if simulator is not None and ("gt_trajectory" in batch):
            controls = self._extract_controls(batch)
            gt_pose = self._extract_gt_pose(batch)
            gt_pos = gt_pose[..., 0:3]
            gt_yaw = quat_wxyz_to_yaw(gt_pose[..., 3:7])

            state0 = batch.get("state0", None)
            if state0 is None:
                init_pose = self._extract_initial_pose(batch)
                if init_pose is not None:
                    state0 = self._default_state0_from_pose(init_pose)
            # Clamp predicted terrain to sane ranges before physics rollout
            height_sim = height.clamp(-0.25, 0.25)          # start tight; widen later
            friction_sim = friction.clamp(0.2, 2.0)         # must be positive

            # stiffness/damping aren't used by your DPhysConfig (it uses constants), but keep them finite anyway
            # stiffness_sim = stiffness.clamp_min(1e-3)
            # damping_sim = damping.clamp_min(1e-3)

            sim_out = simulator(z_grid=height_sim, controls=controls, state=state0, friction=friction_sim)
            # sim_out = simulator(z_grid=height, controls=controls, state=state0, friction=friction)
            pred_pos, pred_yaw = self._parse_simulator_output(sim_out)
            if not torch.isfinite(pred_pos).all():
                raise RuntimeError("pred_pos has NaNs/Infs (simulator blew up)")
            if not torch.isfinite(gt_pos).all():
                raise RuntimeError("gt_pos has NaNs/Infs")
            
            T = min(pred_pos.shape[1], gt_pos.shape[1])
            pred_pos = pred_pos[:, :T]
            gt_pos = gt_pos[:, :T]
            pred_yaw = pred_yaw[:, :T]
            gt_yaw = gt_yaw[:, :T]

            traj_pos_loss = self._pointwise_loss(pred_pos, gt_pos)
            traj_yaw_loss = self._pointwise_loss(_wrap_to_pi(pred_yaw - gt_yaw), torch.zeros_like(gt_yaw))

            total = total + self.weights.traj_pos * traj_pos_loss
            total = total + self.weights.traj_yaw * traj_yaw_loss

            logs["loss/traj_pos"] = float(traj_pos_loss.detach().cpu())
            logs["loss/traj_yaw"] = float(traj_yaw_loss.detach().cpu())

            with torch.no_grad():
                rmse = torch.sqrt(((pred_pos - gt_pos) ** 2).mean()).item()
                logs["metric/pos_rmse"] = float(rmse)

        logs["loss/total"] = float(total.detach().cpu())
        return total, logs

    @staticmethod
    def _parse_simulator_output(sim_out: Any) -> Tuple[Tensor, Tensor]:
        if isinstance(sim_out, dict):
            if "positions" not in sim_out:
                raise KeyError("Simulator dict output must contain key 'positions'.")
            pred_pos = sim_out["positions"]
            if "yaws" in sim_out:
                pred_yaw = sim_out["yaws"]
            elif "rotmats" in sim_out:
                pred_yaw = rotmat_to_yaw(sim_out["rotmats"])
            else:
                pred_yaw = torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=pred_pos.device, dtype=pred_pos.dtype)
            return pred_pos, pred_yaw

        if isinstance(sim_out, (tuple, list)):
            if len(sim_out) == 2 and isinstance(sim_out[0], (tuple, list)) and len(sim_out[0]) >= 3:
                states = sim_out[0]
                Xs = states[0]
                Rs = states[2]
                return Xs, rotmat_to_yaw(Rs)
            if len(sim_out) >= 3 and torch.is_tensor(sim_out[0]) and torch.is_tensor(sim_out[2]):
                Xs = sim_out[0]
                Rs = sim_out[2]
                return Xs, rotmat_to_yaw(Rs)

        if torch.is_tensor(sim_out):
            pred_pos = sim_out
            pred_yaw = torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device=pred_pos.device, dtype=pred_pos.dtype)
            return pred_pos, pred_yaw

        raise TypeError(f"Unsupported simulator output type: {type(sim_out)}")
