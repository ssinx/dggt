"""Differentiable cubemap geometry using an OpenCV camera convention."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# Face order is +X, -X, +Y, -Y, +Z, -Z.
CUBEMAP_FACE_NAMES = ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z")


def cubemap_directions(
    face_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized world directions with shape ``[6, H, W, 3]``."""
    if face_size < 1:
        raise ValueError(f"face_size must be positive, got {face_size}")

    axis = (torch.arange(face_size, device=device, dtype=dtype) + 0.5) * (2.0 / face_size) - 1.0
    v, u = torch.meshgrid(axis, axis, indexing="ij")
    one = torch.ones_like(u)
    directions = torch.stack(
        (
            torch.stack((one, v, -u), dim=-1),       # +X
            torch.stack((-one, v, u), dim=-1),       # -X
            torch.stack((u, one, -v), dim=-1),       # +Y
            torch.stack((u, -one, v), dim=-1),       # -Y
            torch.stack((u, v, one), dim=-1),        # +Z
            torch.stack((-u, v, -one), dim=-1),      # -Z
        ),
        dim=0,
    )
    return F.normalize(directions, dim=-1)


def direction_to_face_uv(directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map directions to a face index and normalized ``grid_sample`` coordinates."""
    if directions.shape[-1] != 3:
        raise ValueError(f"Expected directions[..., 3], got {tuple(directions.shape)}")

    directions = F.normalize(directions, dim=-1)
    x, y, z = directions.unbind(dim=-1)
    abs_directions = directions.abs()
    major_axis = abs_directions.argmax(dim=-1)
    major = abs_directions.gather(-1, major_axis.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)

    face = torch.empty_like(major_axis)
    u = torch.empty_like(x)
    v = torch.empty_like(y)

    x_major = major_axis == 0
    pos = x_major & (x >= 0)
    neg = x_major & ~pos
    face[pos], u[pos], v[pos] = 0, -z[pos] / major[pos], y[pos] / major[pos]
    face[neg], u[neg], v[neg] = 1, z[neg] / major[neg], y[neg] / major[neg]

    y_major = major_axis == 1
    pos = y_major & (y >= 0)
    neg = y_major & ~pos
    face[pos], u[pos], v[pos] = 2, x[pos] / major[pos], -z[pos] / major[pos]
    face[neg], u[neg], v[neg] = 3, x[neg] / major[neg], z[neg] / major[neg]

    z_major = major_axis == 2
    pos = z_major & (z >= 0)
    neg = z_major & ~pos
    face[pos], u[pos], v[pos] = 4, x[pos] / major[pos], y[pos] / major[pos]
    face[neg], u[neg], v[neg] = 5, -x[neg] / major[neg], y[neg] / major[neg]
    return face, torch.stack((u, v), dim=-1).clamp(-1.0, 1.0)


def camera_world_rays(
    intrinsics: torch.Tensor,
    camera_to_worlds: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Create per-pixel world rays, returning ``[..., H, W, 3]``."""
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics[..., 3, 3], got {tuple(intrinsics.shape)}")
    if camera_to_worlds.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera_to_worlds[..., 4, 4], got {tuple(camera_to_worlds.shape)}")
    if intrinsics.shape[:-2] != camera_to_worlds.shape[:-2]:
        raise ValueError("Intrinsics and camera poses must have identical leading dimensions")

    dtype, device = intrinsics.dtype, intrinsics.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    camera_rays = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(intrinsics), pixels)
    world_rays = torch.einsum("...ij,...hwj->...hwi", camera_to_worlds[..., :3, :3], camera_rays)
    return F.normalize(world_rays, dim=-1)


def sample_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sample ``[B, 6, C, Hf, Wf]`` cubemaps at ``[B, ..., 3]`` directions.

    The returned tensor has shape ``[B, ..., C]``. Face boundaries use border
    padding; a seam-consistency loss can be added separately during training.
    """
    if cubemap.ndim != 5 or cubemap.shape[1] != 6:
        raise ValueError(f"Expected cubemap [B, 6, C, H, W], got {tuple(cubemap.shape)}")
    if directions.ndim < 2 or directions.shape[0] != cubemap.shape[0] or directions.shape[-1] != 3:
        raise ValueError("Directions must have shape [B, ..., 3] and match the cubemap batch")

    batch_size, _, channels, _, _ = cubemap.shape
    spatial_shape = directions.shape[1:-1]
    flat_directions = directions.reshape(batch_size, -1, 3)
    face, uv = direction_to_face_uv(flat_directions)
    grid = uv.unsqueeze(2)

    samples = []
    for face_index in range(6):
        face_sample = F.grid_sample(
            cubemap[:, face_index], grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        samples.append(face_sample.squeeze(-1).transpose(1, 2))
    all_samples = torch.stack(samples, dim=2)
    selected = all_samples.gather(
        2, face[..., None, None].expand(-1, -1, 1, channels)
    ).squeeze(2)
    return selected.reshape(batch_size, *spatial_shape, channels)


def render_cubemap(
    cubemap: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_worlds: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render a cubemap into cameras, returning ``[B, S, H, W, C]``."""
    rays = camera_world_rays(intrinsics, camera_to_worlds, height, width)
    return sample_cubemap(cubemap, rays)
