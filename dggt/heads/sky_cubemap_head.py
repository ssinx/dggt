"""Scene-level cubemap sky decoder inspired by Instant NuRec."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dggt.heads.utils import create_uv_grid, position_grid_to_embed
from dggt.utils.cubemap import camera_world_rays, cubemap_directions


class RayEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 4) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies), persistent=False)
        self.output_dim = 3 * (1 + 2 * num_frequencies)

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        encoded = [directions]
        for frequency in self.frequencies.to(dtype=directions.dtype):
            encoded.extend((torch.sin(math.pi * frequency * directions), torch.cos(math.pi * frequency * directions)))
        return torch.cat(encoded, dim=-1)


class ResidualConvBlock(nn.Module):
    """Two convolutions with a residual connection, as used by DPT fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(features)
        residual = self.conv1(residual)
        residual = self.conv2(F.gelu(residual))
        return features + residual


class DPTFusionBlock(nn.Module):
    """Merge a lateral feature into a coarse feature and upsample it."""

    def __init__(self, channels: int, has_lateral: bool = True) -> None:
        super().__init__()
        self.lateral_block = ResidualConvBlock(channels) if has_lateral else None
        self.output_block = ResidualConvBlock(channels)
        self.output_projection = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        features: torch.Tensor,
        lateral: torch.Tensor | None,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if self.lateral_block is not None:
            if lateral is None:
                raise ValueError("A lateral feature is required by this fusion block")
            features = features + self.lateral_block(lateral)
        features = self.output_block(features)
        features = F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)
        return self.output_projection(features)


class CubemapDPTUpsampler(nn.Module):
    """Reassemble patch queries at four scales and fuse them into RGB faces."""

    def __init__(self, input_dim: int, feature_dim: int, output_size: int) -> None:
        super().__init__()
        branch_dims = (
            max(input_dim // 8, 16),
            max(input_dim // 4, 16),
            max(input_dim // 2, 16),
            input_dim,
        )
        self.projections = nn.ModuleList([nn.Conv2d(input_dim, dim, 1) for dim in branch_dims])
        self.resizers = nn.ModuleList(
            [
                nn.ConvTranspose2d(branch_dims[0], branch_dims[0], 4, stride=4),
                nn.ConvTranspose2d(branch_dims[1], branch_dims[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(branch_dims[3], branch_dims[3], 3, stride=2, padding=1),
            ]
        )
        self.lateral_projections = nn.ModuleList(
            [nn.Conv2d(dim, feature_dim, 3, padding=1, bias=False) for dim in branch_dims]
        )
        self.fusion4 = DPTFusionBlock(feature_dim, has_lateral=False)
        self.fusion3 = DPTFusionBlock(feature_dim)
        self.fusion2 = DPTFusionBlock(feature_dim)
        self.fusion1 = DPTFusionBlock(feature_dim)
        self.output_size = output_size
        self.output = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )

    @staticmethod
    def _add_position_embedding(features: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
        height, width = features.shape[-2:]
        grid = create_uv_grid(width, height, dtype=features.dtype, device=features.device)
        embedding = position_grid_to_embed(grid, features.shape[1]).to(dtype=features.dtype)
        embedding = embedding.permute(2, 0, 1).unsqueeze(0)
        return features + strength * embedding

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        query_features = self._add_position_embedding(query_features)
        layers = [
            lateral(resize(project(query_features)))
            for project, resize, lateral in zip(
                self.projections, self.resizers, self.lateral_projections
            )
        ]

        path4 = self.fusion4(layers[3], None, layers[2].shape[-2:])
        path3 = self.fusion3(path4, layers[2], layers[1].shape[-2:])
        path2 = self.fusion2(path3, layers[1], layers[0].shape[-2:])
        finest_size = (layers[0].shape[-2] * 2, layers[0].shape[-1] * 2)
        path1 = self.fusion1(path2, layers[0], finest_size)
        if path1.shape[-2:] != (self.output_size, self.output_size):
            path1 = F.interpolate(
                path1,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        path1 = self._add_position_embedding(path1)
        return self.output(path1)


class SkyCubemapHead(nn.Module):
    """Decode one world-aligned RGB cubemap for each input scene.

    Backbone patch tokens carry scene context. Raw RGB patches preserve high
    frequency appearance, while world-ray embeddings align observations from
    different cameras and timesteps.
    """

    def __init__(
        self,
        token_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 1,
        patch_size: int = 14,
        cubemap_size: int = 224,
        query_size: int = 16,
        ray_frequencies: int = 4,
        dpt_features: int = 128,
        decoder_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if cubemap_size < query_size:
            raise ValueError("cubemap_size must be at least query_size")
        if cubemap_size % query_size:
            raise ValueError("cubemap_size must be divisible by query_size")
        if query_size < 2:
            raise ValueError("query_size must be at least 2 for four-scale DPT reassembly")

        self.patch_size = patch_size
        self.cubemap_size = cubemap_size
        self.query_size = query_size
        self.decoder_checkpointing = decoder_checkpointing
        self.ray_encoding = RayEncoding(ray_frequencies)

        self.token_norm = nn.LayerNorm(token_dim)
        self.token_projection = nn.Linear(token_dim, embed_dim)
        self.rgb_patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.input_ray_projection = nn.Linear(self.ray_encoding.output_dim, embed_dim)
        query_patch_size = cubemap_size // query_size
        self.query_ray_patch = nn.Conv2d(
            self.ray_encoding.output_dim,
            embed_dim,
            kernel_size=query_patch_size,
            stride=query_patch_size,
        )
        self.register_buffer(
            "query_directions", cubemap_directions(cubemap_size), persistent=False
        )
        self.face_embedding = nn.Parameter(torch.zeros(1, 6, 1, embed_dim))
        self.scene_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.cross_attention = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(depth)]
        )
        self.query_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.kv_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, 4 * embed_dim),
                    nn.GELU(),
                    nn.Linear(4 * embed_dim, embed_dim),
                )
                for _ in range(depth)
            ]
        )
        self.upsampler = CubemapDPTUpsampler(
            input_dim=embed_dim,
            feature_dim=dpt_features,
            output_size=cubemap_size,
        )
        nn.init.normal_(self.face_embedding, std=0.02)
        nn.init.normal_(self.scene_token, std=0.02)

    def _build_key_values(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_worlds: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        patch_tokens = tokens[:, :, patch_start_idx:]
        expected_patches = patch_h * patch_w
        if patch_tokens.shape[2] != expected_patches:
            raise ValueError(
                f"Backbone produced {patch_tokens.shape[2]} patches, expected {expected_patches} for {height}x{width}"
            )

        token_features = self.token_projection(self.token_norm(patch_tokens))
        rgb_features = self.rgb_patch(images.flatten(0, 1)).flatten(2).transpose(1, 2)
        rgb_features = rgb_features.view(batch_size, sequence_length, expected_patches, -1)

        rays = camera_world_rays(intrinsics, camera_to_worlds, height, width)
        rays = rays.permute(0, 1, 4, 2, 3).flatten(0, 1)
        rays = F.adaptive_avg_pool2d(rays, (patch_h, patch_w))
        rays = F.normalize(rays, dim=1).flatten(2).transpose(1, 2)
        rays = rays.view(batch_size, sequence_length, expected_patches, 3)
        ray_features = self.input_ray_projection(self.ray_encoding(rays))

        key_values = token_features + rgb_features + ray_features
        return key_values.flatten(1, 2)

    def forward(
        self,
        image_tokens_list: Sequence[torch.Tensor],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_worlds: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        if not image_tokens_list:
            raise ValueError("image_tokens_list cannot be empty")
        if intrinsics.shape[:2] != images.shape[:2] or camera_to_worlds.shape[:2] != images.shape[:2]:
            raise ValueError("Calibration batch and sequence dimensions must match images")

        key_values = self._build_key_values(
            image_tokens_list[-1], images, intrinsics, camera_to_worlds, patch_start_idx
        )
        batch_size = images.shape[0]
        query_directions = self.query_directions.to(dtype=images.dtype)
        encoded_query_rays = self.ray_encoding(query_directions).permute(0, 3, 1, 2)
        queries = self.query_ray_patch(encoded_query_rays).flatten(2).transpose(1, 2)
        queries = queries.view(1, 6, self.query_size * self.query_size, -1)
        queries = queries + self.face_embedding
        queries = queries.expand(batch_size, -1, -1, -1)
        scene_token = self.scene_token.expand(batch_size, -1, -1)
        key_values = torch.cat((scene_token, key_values), dim=1)

        decoded_faces = []
        for face_index in range(6):
            face_queries = queries[:, face_index]
            for attention, query_norm, kv_norm, ffn_norm, ffn in zip(
                self.cross_attention, self.query_norms, self.kv_norms, self.ffn_norms, self.ffns
            ):
                normalized_key_values = kv_norm(key_values)
                attended, _ = attention(
                    query_norm(face_queries), normalized_key_values, normalized_key_values, need_weights=False
                )
                face_queries = face_queries + attended
                face_queries = face_queries + ffn(ffn_norm(face_queries))
            decoded_faces.append(face_queries)

        features = torch.stack(decoded_faces, dim=1)
        features = features.view(batch_size * 6, self.query_size, self.query_size, -1).permute(0, 3, 1, 2)
        if self.decoder_checkpointing and self.training and features.requires_grad:
            rgb_logits = checkpoint(self.upsampler, features, use_reentrant=False)
        else:
            rgb_logits = self.upsampler(features)
        rgb = torch.sigmoid(rgb_logits)
        return rgb.view(batch_size, 6, 3, self.cubemap_size, self.cubemap_size)
