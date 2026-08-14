import torch

from dggt.heads.sky_cubemap_head import SkyCubemapHead


def test_sky_cubemap_head_shape_and_gradient():
    batch_size, sequence_length = 1, 2
    height = width = 28
    patch_start_idx = 2
    token_dim = 24
    num_patches = (height // 14) * (width // 14)
    tokens = torch.randn(batch_size, sequence_length, patch_start_idx + num_patches, token_dim)
    images = torch.rand(batch_size, sequence_length, 3, height, width)
    intrinsics = torch.tensor(
        [[[[20.0, 0.0, 14.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]]] * sequence_length]
    )
    camera_to_worlds = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, sequence_length, 1, 1)
    head = SkyCubemapHead(
        token_dim=token_dim,
        embed_dim=32,
        num_heads=4,
        depth=1,
        cubemap_size=16,
        query_size=2,
    )

    cubemap = head([tokens], images, intrinsics, camera_to_worlds, patch_start_idx)
    assert cubemap.shape == (batch_size, 6, 3, 16, 16)
    assert cubemap.min() >= 0 and cubemap.max() <= 1
    cubemap.mean().backward()
    assert head.token_projection.weight.grad is not None
