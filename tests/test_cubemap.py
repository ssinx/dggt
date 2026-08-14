import torch

from dggt.utils.cubemap import (
    camera_world_rays,
    cubemap_directions,
    direction_to_face_uv,
    sample_cubemap,
)


def test_cubemap_face_centers_round_trip():
    directions = cubemap_directions(1).reshape(6, 3)
    faces, uv = direction_to_face_uv(directions)
    assert torch.equal(faces, torch.arange(6))
    assert torch.allclose(uv, torch.zeros_like(uv))


def test_sample_cubemap_selects_expected_faces_and_backpropagates():
    cubemap = torch.arange(6.0).view(1, 6, 1, 1, 1).requires_grad_()
    directions = cubemap_directions(1).reshape(1, 6, 3)
    sampled = sample_cubemap(cubemap, directions)
    assert torch.allclose(sampled[0, :, 0], torch.arange(6.0))
    sampled.sum().backward()
    assert torch.allclose(cubemap.grad, torch.ones_like(cubemap))


def test_camera_world_rays_uses_opencv_axes_and_world_rotation():
    intrinsics = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]])
    camera_to_world = torch.eye(4).unsqueeze(0)
    rays = camera_world_rays(intrinsics, camera_to_world, 1, 1)
    assert torch.allclose(rays[0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))

    rotated = camera_to_world.clone()
    rotated[0, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rays = camera_world_rays(intrinsics, rotated, 1, 1)
    assert torch.allclose(rays[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
