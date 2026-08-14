"""Scene-level cubemap sky decoder inspired by Instant NuRec."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if cubemap_size < query_size:
            raise ValueError("cubemap_size must be at least query_size")

        self.patch_size = patch_size
        self.cubemap_size = cubemap_size
        self.query_size = query_size
        self.ray_encoding = RayEncoding(ray_frequencies)

        self.token_norm = nn.LayerNorm(token_dim)
        self.token_projection = nn.Linear(token_dim, embed_dim)
        self.rgb_patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.input_ray_projection = nn.Linear(self.ray_encoding.output_dim, embed_dim)
        self.query_projection = nn.Linear(self.ray_encoding.output_dim, embed_dim)
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
        self.pre_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),
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
        query_directions = cubemap_directions(
            self.query_size, device=images.device, dtype=images.dtype
        )
        queries = self.query_projection(self.ray_encoding(query_directions))
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
                attended, _ = attention(
                    query_norm(face_queries), kv_norm(key_values), kv_norm(key_values), need_weights=False
                )
                face_queries = face_queries + attended
                face_queries = face_queries + ffn(ffn_norm(face_queries))
            decoded_faces.append(face_queries)

        features = torch.stack(decoded_faces, dim=1)
        features = features.view(batch_size * 6, self.query_size, self.query_size, -1).permute(0, 3, 1, 2)
        features = self.pre_upsample(features)
        features = F.interpolate(
            features, size=(self.cubemap_size, self.cubemap_size), mode="bilinear", align_corners=False
        )
        rgb = torch.sigmoid(self.decoder(features))
        return rgb.view(batch_size, 6, 3, self.cubemap_size, self.cubemap_size)
