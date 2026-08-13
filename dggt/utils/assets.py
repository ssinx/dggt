"""External 3D Gaussian asset loading and scene placement helpers."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


ASSET_HARVESTER_FIELDS = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
SH_C0 = 0.28209479177387814
GAUSSIAN_FIELDS = ("means", "colors", "opacities", "scales", "quats")


def _read_asset_harvester_ply(path: Path) -> dict[str, torch.Tensor]:
    """Read a complete compatible Asset Harvester Gaussian PLY file."""
    with path.open("rb") as file:
        if file.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")

        file_format = None
        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex_element = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            line = line.decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                file_format = parts[1]
            elif parts[0] == "element":
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex_element:
                if len(parts) != 3 or parts[1] not in ("float", "float32"):
                    raise ValueError(f"Only float vertex properties are supported in {path}")
                properties.append((parts[2], parts[1]))

        if file_format != "binary_little_endian":
            raise ValueError(f"Expected binary_little_endian PLY, got {file_format!r}: {path}")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        property_names = tuple(name for name, _ in properties)
        if set(property_names) != set(ASSET_HARVESTER_FIELDS) or len(properties) != len(ASSET_HARVESTER_FIELDS):
            raise ValueError(
                f"Unsupported Gaussian PLY schema in {path}. Expected {ASSET_HARVESTER_FIELDS}, got {property_names}."
            )

        bytes_per_vertex = struct.calcsize("<" + "f" * len(properties))
        expected_payload_size = vertex_count * bytes_per_vertex
        payload = file.read(expected_payload_size)
        if len(payload) != expected_payload_size or file.read(1):
            raise ValueError(
                f"Incomplete or oversized PLY payload in {path}: expected {expected_payload_size} bytes, "
                f"read {len(payload)} bytes."
            )

    dtype = np.dtype([(name, "<f4") for name in property_names])
    vertices = np.frombuffer(payload, dtype=dtype, count=vertex_count)
    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).copy()
    colors = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=-1).copy()
    opacities = vertices["opacity"].copy()
    scales = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=-1).copy()
    quats = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=-1).copy()

    means[:, 1:] *= -1
    quats = np.stack([-quats[:, 1], quats[:, 0], -quats[:, 3], quats[:, 2]], axis=-1)
    colors = np.clip(colors * SH_C0 + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + np.exp(-opacities))
    scales = np.exp(scales)

    return {
        "means": torch.from_numpy(means).float(),
        "colors": torch.from_numpy(colors).float(),
        "opacities": torch.from_numpy(opacities).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": torch.from_numpy(quats).float(),
    }


def _normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    if torch.any(norms <= 1e-8):
        raise ValueError("Gaussian asset contains a zero-length quaternion.")
    return quats / norms


def _validate_gaussians(gaussians: dict[str, torch.Tensor], source: str) -> dict[str, torch.Tensor]:
    missing = set(GAUSSIAN_FIELDS) - set(gaussians)
    if missing:
        raise ValueError(f"Gaussian asset {source} is missing fields: {sorted(missing)}")

    validated = {name: torch.as_tensor(gaussians[name], dtype=torch.float32).contiguous() for name in GAUSSIAN_FIELDS}
    count = validated["means"].shape[0]
    expected_shapes = {
        "means": (count, 3),
        "colors": (count, 3),
        "opacities": (count,),
        "scales": (count, 3),
        "quats": (count, 4),
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(validated[name].shape) != expected_shape:
            raise ValueError(f"Gaussian asset {source} has {name} shape {tuple(validated[name].shape)}, expected {expected_shape}.")
        if not torch.isfinite(validated[name]).all():
            raise ValueError(f"Gaussian asset {source} contains non-finite {name} values.")
    if count == 0:
        raise ValueError(f"Gaussian asset {source} has no Gaussians.")
    if torch.any(validated["scales"] <= 0):
        raise ValueError(f"Gaussian asset {source} contains non-positive scales.")

    validated["colors"] = validated["colors"].clamp(0, 1)
    validated["opacities"] = validated["opacities"].clamp(0, 1)
    validated["quats"] = _normalize_quaternions(validated["quats"])
    return validated


def _read_lwh(path: Path) -> torch.Tensor:
    values = [float(value) for value in path.read_text().split()]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"Expected three positive length/width/height values in {path}")
    return torch.tensor(values, dtype=torch.float32)


def _ground_center_asset(gaussians: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Center horizontal axes and put the robust lower Y bound at zero."""
    mask = gaussians["opacities"] >= 0.05
    reference_means = gaussians["means"][mask] if mask.any() else gaussians["means"]
    lower = torch.quantile(reference_means, 0.01, dim=0)
    upper = torch.quantile(reference_means, 0.99, dim=0)
    origin = torch.stack(
        [
            (lower[0] + upper[0]) * 0.5,
            lower[1],
            (lower[2] + upper[2]) * 0.5,
        ]
    )
    centered = dict(gaussians)
    centered["means"] = gaussians["means"] - origin
    return centered, origin


def convert_asset_harvester_ply(
    ply_path: str | Path,
    lwh_path: str | Path | None = None,
    center_ground: bool = True,
    max_gaussians: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Convert Asset Harvester PLY output into a meter-based DGGT Gaussian asset."""
    ply_path = Path(ply_path).resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply_path}")
    if lwh_path is None:
        lwh_path = ply_path.parent / "multiview" / "lwh.txt"
    lwh_path = Path(lwh_path).resolve()
    if not lwh_path.is_file():
        raise FileNotFoundError(f"L/W/H metadata not found: {lwh_path}")

    gaussians = _validate_gaussians(_read_asset_harvester_ply(ply_path), str(ply_path))
    lwh_m = _read_lwh(lwh_path)
    meters_per_normalized_unit = float(lwh_m.max())
    gaussians["means"] = gaussians["means"] * meters_per_normalized_unit
    gaussians["scales"] = gaussians["scales"] * meters_per_normalized_unit
    local_origin_m = torch.zeros(3, dtype=torch.float32)
    if center_ground:
        gaussians, local_origin_m = _ground_center_asset(gaussians)

    if max_gaussians is not None:
        if max_gaussians <= 0:
            raise ValueError("max_gaussians must be positive when provided")
        count = gaussians["means"].shape[0]
        if count > max_gaussians:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(count, generator=generator)[:max_gaussians]
            gaussians = {name: values[indices] for name, values in gaussians.items()}

    return {
        "format": "dggt_gaussian_asset_v1",
        **gaussians,
        "lwh_m": lwh_m,
        "local_origin_m": local_origin_m,
        "meters_per_normalized_unit": torch.tensor(meters_per_normalized_unit, dtype=torch.float32),
    }


def save_gaussian_asset(asset: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asset, output_path)


def load_asset_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("assets"), list):
        raise ValueError(f"Asset manifest must be a JSON object containing an assets list: {manifest_path}")

    assets = []
    for index, spec in enumerate(manifest["assets"]):
        if not isinstance(spec, dict):
            raise ValueError(f"Asset manifest entry {index} must be an object.")
        if "asset_path" not in spec:
            raise ValueError(f"Asset manifest entry {index} is missing asset_path.")
        resolved_spec = dict(spec)
        asset_path = Path(spec["asset_path"])
        resolved_spec["_asset_path"] = str((manifest_path.parent / asset_path).resolve()) if not asset_path.is_absolute() else str(asset_path)
        resolved_spec.setdefault("id", f"asset_{index}")
        resolved_spec.setdefault("scale", 1.0)
        resolved_spec.setdefault("opacity_scale", 1.0)
        resolved_spec.setdefault("start_frame", 0)
        resolved_spec.setdefault("end_frame", None)
        resolved_spec.setdefault("ground_snap", True)
        resolved_spec.setdefault("contact_opacity_threshold", 0.05)
        resolved_spec.setdefault("contact_quantile", 0.99)
        resolved_spec.setdefault("ground_opacity_threshold", 0.05)
        resolved_spec.setdefault("ground_quantile", 0.75)
        resolved_spec.setdefault("ground_search_radius", 1.5)
        resolved_spec.setdefault("ground_vertical_band", 1.5)
        resolved_spec.setdefault("ground_min_points", 32)
        resolved_spec.setdefault("ground_clearance_m", 0.0)
        resolved_spec.setdefault("vlm_orientation_refine", True)
        resolved_spec.setdefault("orientation_prompt", "")
        resolved_spec.setdefault("orientation_top_k", 3)
        resolved_spec.setdefault("orientation_projection_max_points", 3000)
        resolved_spec.setdefault("max_yaw_delta_deg", 90.0)
        resolved_spec.setdefault("min_scale_factor", 0.5)
        resolved_spec.setdefault("max_scale_factor", 2.0)
        if float(resolved_spec["scale"]) <= 0:
            raise ValueError(f"Asset {resolved_spec['id']} has a non-positive scale.")
        if float(resolved_spec["opacity_scale"]) < 0:
            raise ValueError(f"Asset {resolved_spec['id']} has a negative opacity_scale.")
        for field_name in ("contact_opacity_threshold", "ground_opacity_threshold"):
            if not 0 <= float(resolved_spec[field_name]) <= 1:
                raise ValueError(f"Asset {resolved_spec['id']} field {field_name} must be in [0, 1].")
        for field_name in ("contact_quantile", "ground_quantile"):
            if not 0 < float(resolved_spec[field_name]) <= 1:
                raise ValueError(f"Asset {resolved_spec['id']} field {field_name} must be in (0, 1].")
        for field_name in ("ground_search_radius", "ground_vertical_band"):
            if float(resolved_spec[field_name]) <= 0:
                raise ValueError(f"Asset {resolved_spec['id']} field {field_name} must be positive.")
        if int(resolved_spec["ground_min_points"]) < 3:
            raise ValueError(f"Asset {resolved_spec['id']} field ground_min_points must be at least 3.")
        if int(resolved_spec["orientation_top_k"]) < 1:
            raise ValueError(f"Asset {resolved_spec['id']} field orientation_top_k must be at least 1.")
        if int(resolved_spec["orientation_projection_max_points"]) < 32:
            raise ValueError(
                f"Asset {resolved_spec['id']} field orientation_projection_max_points must be at least 32."
            )
        if not 0 < float(resolved_spec["max_yaw_delta_deg"]) <= 180:
            raise ValueError(f"Asset {resolved_spec['id']} field max_yaw_delta_deg must be in (0, 180].")
        if float(resolved_spec["min_scale_factor"]) <= 0:
            raise ValueError(f"Asset {resolved_spec['id']} field min_scale_factor must be positive.")
        if float(resolved_spec["max_scale_factor"]) < float(resolved_spec["min_scale_factor"]):
            raise ValueError(
                f"Asset {resolved_spec['id']} field max_scale_factor must be at least min_scale_factor."
            )
        assets.append(resolved_spec)
    manifest["assets"] = assets
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def load_manifest_assets(manifest: dict[str, Any], device: str | torch.device) -> dict[str, dict[str, torch.Tensor]]:
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for spec in manifest["assets"]:
        asset_path = spec["_asset_path"]
        if asset_path in cache:
            continue
        try:
            payload = torch.load(asset_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(asset_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"DGGT Gaussian asset must contain a dictionary: {asset_path}")
        cache[asset_path] = {name: value.to(device) for name, value in _validate_gaussians(payload, asset_path).items()}
    return cache


def _matrix_from_manifest(value: Any, asset_id: str, field_name: str, device: torch.device) -> torch.Tensor:
    matrix = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tuple(matrix.shape) != (4, 4):
        raise ValueError(f"Asset {asset_id} field {field_name} must be a 4x4 matrix.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"Asset {asset_id} field {field_name} contains non-finite values.")
    if not torch.allclose(matrix[3], torch.tensor([0, 0, 0, 1], device=device, dtype=matrix.dtype), atol=1e-4):
        raise ValueError(f"Asset {asset_id} field {field_name} must have bottom row [0, 0, 0, 1].")
    return matrix


def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a proper 3x3 rotation matrix to a wxyz quaternion."""
    trace = rotation.trace()
    if trace > 0:
        scale = torch.sqrt(trace + 1.0) * 2.0
        quat = torch.stack(
            [
                scale * 0.25,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal = torch.diagonal(rotation)
        index = int(torch.argmax(diagonal).item())
        if index == 0:
            scale = torch.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = torch.stack(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    scale * 0.25,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = torch.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = torch.stack(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    scale * 0.25,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = torch.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = torch.stack(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    scale * 0.25,
                ]
            )
    return _normalize_quaternions(quat.unsqueeze(0))[0]


def _quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_w, lhs_x, lhs_y, lhs_z = lhs.unbind(dim=-1)
    rhs_w, rhs_x, rhs_y, rhs_z = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lhs_w * rhs_w - lhs_x * rhs_x - lhs_y * rhs_y - lhs_z * rhs_z,
            lhs_w * rhs_x + lhs_x * rhs_w + lhs_y * rhs_z - lhs_z * rhs_y,
            lhs_w * rhs_y - lhs_x * rhs_z + lhs_y * rhs_w + lhs_z * rhs_x,
            lhs_w * rhs_z + lhs_x * rhs_y - lhs_y * rhs_x + lhs_z * rhs_w,
        ],
        dim=-1,
    )


def _asset_transform_for_frame(spec: dict[str, Any], frame_index: int, device: torch.device) -> torch.Tensor | None:
    if frame_index < int(spec["start_frame"]):
        return None
    end_frame = spec["end_frame"]
    if end_frame is not None and frame_index > int(end_frame):
        return None

    frame_transforms = spec.get("frame_transforms")
    if frame_transforms is not None:
        if not isinstance(frame_transforms, dict) or str(frame_index) not in frame_transforms:
            return None
        return _matrix_from_manifest(frame_transforms[str(frame_index)], spec["id"], "frame_transforms", device)
    if "asset_to_world" not in spec:
        raise ValueError(f"Asset {spec['id']} needs asset_to_world for static placement.")
    return _matrix_from_manifest(spec["asset_to_world"], spec["id"], "asset_to_world", device)


def _scene_matches(spec: dict[str, Any], scene_name: str) -> bool:
    target_scenes = spec.get("scene_names")
    if target_scenes is None:
        return True
    if isinstance(target_scenes, str):
        return scene_name == target_scenes
    if isinstance(target_scenes, list):
        return scene_name in {str(name) for name in target_scenes}
    raise ValueError(f"Asset {spec['id']} scene_names must be a string or a list of strings.")


def rotation_matrix_about_axis(axis: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """Return a right-handed rotation about an arbitrary world-space axis."""
    axis = torch.as_tensor(axis)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.deg2rad(torch.as_tensor(angle_degrees, device=axis.device, dtype=axis.dtype))
    x, y, z = axis
    zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return identity + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)


def asset_rotation_for_frame(
    spec: dict[str, Any], frame_index: int, device: torch.device
) -> torch.Tensor:
    """Return the manifest rotation active in a frame."""
    transform = _asset_transform_for_frame(spec, frame_index, device)
    if transform is None:
        raise ValueError(f"Asset {spec['id']} is not active in frame {frame_index}.")
    return transform[:3, :3]


def select_asset_orientation_frames(
    asset: dict[str, torch.Tensor],
    spec: dict[str, Any],
    frame_extrinsics: torch.Tensor,
    frame_intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    placement_position_world: torch.Tensor,
    scene_units_per_meter: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select clear, diverse views using only asset projection geometry."""
    device = asset["means"].device
    position = torch.as_tensor(placement_position_world, device=device, dtype=asset["means"].dtype)
    opaque_mask = asset["opacities"] >= float(spec["contact_opacity_threshold"])
    local_means = asset["means"][opaque_mask]
    if local_means.shape[0] < 32:
        local_means = asset["means"]
    maximum_points = int(spec["orientation_projection_max_points"])
    if local_means.shape[0] > maximum_points:
        sample_indices = torch.linspace(
            0, local_means.shape[0] - 1, maximum_points, device=device
        ).long()
        local_means = local_means[sample_indices]

    frame_scores: list[dict[str, Any]] = []
    camera_centers: dict[int, torch.Tensor] = {}
    for frame_index in range(frame_extrinsics.shape[0]):
        transform = _asset_transform_for_frame(spec, frame_index, device)
        if transform is None:
            continue
        rotation = transform[:3, :3]
        world_means = (
            local_means * float(spec["scale"]) @ rotation.transpose(0, 1)
        ) * float(scene_units_per_meter) + position
        extrinsic = frame_extrinsics[frame_index]
        intrinsic = frame_intrinsics[frame_index]
        camera_means = world_means @ extrinsic[:3, :3].transpose(0, 1) + extrinsic[:3, 3]
        positive_depth = camera_means[:, 2] > 1e-6
        projected = camera_means @ intrinsic.transpose(0, 1)
        pixels = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        finite = torch.isfinite(pixels).all(dim=-1) & positive_depth
        if not finite.any():
            frame_scores.append({"frame_index": frame_index, "score": 0.0, "status": "behind_camera"})
            continue
        valid_pixels = pixels[finite]
        in_frame = (
            (valid_pixels[:, 0] >= 0)
            & (valid_pixels[:, 0] <= image_width - 1)
            & (valid_pixels[:, 1] >= 0)
            & (valid_pixels[:, 1] <= image_height - 1)
        )
        minimum = valid_pixels.amin(dim=0)
        maximum = valid_pixels.amax(dim=0)
        clipped_minimum = torch.maximum(minimum, minimum.new_tensor([0.0, 0.0]))
        clipped_maximum = torch.minimum(
            maximum, maximum.new_tensor([float(image_width - 1), float(image_height - 1)])
        )
        full_size = (maximum - minimum).clamp_min(0)
        clipped_size = (clipped_maximum - clipped_minimum).clamp_min(0)
        full_area = full_size.prod()
        clipped_area = clipped_size.prod()
        image_area = float(image_height * image_width)
        projected_area_ratio = float((clipped_area / image_area).item())
        in_frame_ratio = float(in_frame.float().mean().item())
        crop_completeness = float((clipped_area / full_area.clamp_min(1.0)).clamp(0, 1).item())
        resolution_score = min(projected_area_ratio / 0.08, 1.0)
        score = (
            0.45 * resolution_score
            + 0.35 * in_frame_ratio
            + 0.20 * crop_completeness
        )
        frame_scores.append(
            {
                "frame_index": frame_index,
                "score": score,
                "projected_area_ratio": projected_area_ratio,
                "in_frame_ratio": in_frame_ratio,
                "crop_completeness": crop_completeness,
                "bbox_pixels_xyxy": [
                    float(clipped_minimum[0].item()),
                    float(clipped_minimum[1].item()),
                    float(clipped_maximum[0].item()),
                    float(clipped_maximum[1].item()),
                ],
                "sampled_asset_point_count": int(local_means.shape[0]),
                "status": "scored",
            }
        )
        camera_centers[frame_index] = -(
            extrinsic[:3, :3].transpose(0, 1) @ extrinsic[:3, 3]
        )

    viable = [entry for entry in frame_scores if entry["score"] > 0 and entry["frame_index"] in camera_centers]
    viable.sort(key=lambda entry: entry["score"], reverse=True)
    selected: list[int] = []
    top_k = min(int(spec["orientation_top_k"]), len(viable))
    while len(selected) < top_k:
        best_entry = None
        best_selection_score = -1.0
        for entry in viable:
            frame_index = entry["frame_index"]
            if frame_index in selected:
                continue
            diversity = 1.0
            if selected:
                candidate_ray = camera_centers[frame_index] - position
                candidate_ray = candidate_ray / torch.linalg.vector_norm(candidate_ray)
                angles = []
                for selected_index in selected:
                    selected_ray = camera_centers[selected_index] - position
                    selected_ray = selected_ray / torch.linalg.vector_norm(selected_ray)
                    cosine = torch.dot(candidate_ray, selected_ray).clamp(-1, 1)
                    angles.append(float(torch.acos(cosine).item() / torch.pi))
                diversity = min(angles)
            selection_score = float(entry["score"]) * (0.75 + 0.25 * diversity)
            if selection_score > best_selection_score:
                best_entry = entry
                best_selection_score = selection_score
        if best_entry is None:
            break
        best_entry["selection_score"] = best_selection_score
        selected.append(best_entry["frame_index"])
    return selected, frame_scores


def transform_asset_gaussians(
    asset: dict[str, torch.Tensor],
    asset_to_world_m: torch.Tensor,
    asset_scale: float,
    scene_units_per_meter: float,
) -> dict[str, torch.Tensor]:
    """Place a meter-based local asset into the DGGT world coordinate system."""
    if scene_units_per_meter <= 0:
        raise ValueError("scene_units_per_meter must be positive")
    rotation = asset_to_world_m[:3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    if not torch.allclose(rotation.transpose(0, 1) @ rotation, identity, atol=1e-3, rtol=1e-3):
        raise ValueError("asset_to_world must contain rotation only; use the manifest scale field for uniform scale.")
    if torch.linalg.det(rotation) <= 0:
        raise ValueError("asset_to_world rotation must have a positive determinant.")

    local_scale = float(asset_scale)
    scene_scale = float(scene_units_per_meter)
    means_m = asset["means"] * local_scale
    means = (means_m @ rotation.transpose(0, 1) + asset_to_world_m[:3, 3]) * scene_scale
    pose_quat = _rotation_matrix_to_quaternion(rotation).expand_as(asset["quats"])
    quats = _normalize_quaternions(_quaternion_multiply(pose_quat, asset["quats"]))
    return {
        "means": means,
        "colors": asset["colors"],
        "opacities": asset["opacities"],
        "scales": asset["scales"] * local_scale * scene_scale,
        "quats": quats,
    }


def snap_asset_placement_to_ground(
    asset: dict[str, torch.Tensor],
    spec: dict[str, Any],
    frame_index: int,
    initial_position_world: torch.Tensor,
    scene_points_world: torch.Tensor,
    scene_opacities: torch.Tensor,
    down_direction_world: torch.Tensor,
    scene_units_per_meter: float,
    rotation_delta_world: torch.Tensor | None = None,
    asset_scale_factor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Move an asset only vertically until its opaque lower contour touches local ground."""
    initial_position = torch.as_tensor(
        initial_position_world,
        device=asset["means"].device,
        dtype=asset["means"].dtype,
    )
    down = torch.as_tensor(
        down_direction_world,
        device=initial_position.device,
        dtype=initial_position.dtype,
    )
    down_norm = torch.linalg.vector_norm(down)
    if not torch.isfinite(down_norm) or down_norm <= 1e-8:
        raise ValueError("Ground snapping requires a finite non-zero down direction.")
    down = down / down_norm
    diagnostics: dict[str, Any] = {
        "enabled": bool(spec["ground_snap"]),
        "initial_world_coordinate_xyz": initial_position.detach().cpu().tolist(),
        "down_direction_world_xyz": down.detach().cpu().tolist(),
        "effective_asset_scale": float(spec["scale"]) * float(asset_scale_factor),
    }
    if not spec["ground_snap"]:
        diagnostics.update({"status": "disabled", "final_world_coordinate_xyz": diagnostics["initial_world_coordinate_xyz"]})
        return initial_position, diagnostics

    scene_scale = float(scene_units_per_meter)
    scene_points = torch.as_tensor(scene_points_world, device=initial_position.device, dtype=initial_position.dtype)
    opacities = torch.as_tensor(scene_opacities, device=initial_position.device, dtype=initial_position.dtype).reshape(-1)
    if scene_points.ndim != 2 or scene_points.shape[1] != 3 or scene_points.shape[0] != opacities.shape[0]:
        raise ValueError("Scene points and opacities must have shapes [N, 3] and [N].")

    offsets = scene_points - initial_position
    vertical_offsets = offsets @ down
    horizontal_offsets = offsets - vertical_offsets[:, None] * down
    horizontal_distances = torch.linalg.vector_norm(horizontal_offsets, dim=-1)
    local_mask = (
        torch.isfinite(scene_points).all(dim=-1)
        & torch.isfinite(opacities)
        & (opacities >= float(spec["ground_opacity_threshold"]))
        & (horizontal_distances <= float(spec["ground_search_radius"]) * scene_scale)
        & (vertical_offsets.abs() <= float(spec["ground_vertical_band"]) * scene_scale)
    )
    local_points = scene_points[local_mask]
    minimum_points = int(spec["ground_min_points"])
    diagnostics["local_scene_point_count"] = int(local_points.shape[0])
    if local_points.shape[0] < minimum_points:
        diagnostics.update(
            {
                "status": "fallback_insufficient_local_ground_points",
                "final_world_coordinate_xyz": diagnostics["initial_world_coordinate_xyz"],
            }
        )
        return initial_position, diagnostics

    local_down_coordinates = local_points @ down
    ground_cutoff = torch.quantile(local_down_coordinates, float(spec["ground_quantile"]))
    ground_points = local_points[local_down_coordinates >= ground_cutoff]
    if ground_points.shape[0] < 3:
        diagnostics.update(
            {
                "status": "fallback_insufficient_ground_candidates",
                "ground_candidate_count": int(ground_points.shape[0]),
                "final_world_coordinate_xyz": diagnostics["initial_world_coordinate_xyz"],
            }
        )
        return initial_position, diagnostics

    reference_axis = torch.eye(3, device=down.device, dtype=down.dtype)[torch.argmin(down.abs())]
    horizontal_x = torch.linalg.cross(down, reference_axis)
    horizontal_x = horizontal_x / torch.linalg.vector_norm(horizontal_x)
    horizontal_y = torch.linalg.cross(down, horizontal_x)
    centered_ground = ground_points - initial_position
    design = torch.stack(
        (centered_ground @ horizontal_x, centered_ground @ horizontal_y, torch.ones_like(centered_ground[:, 0])),
        dim=-1,
    )
    targets = ground_points @ down
    inliers = torch.ones(ground_points.shape[0], dtype=torch.bool, device=ground_points.device)
    coefficients = torch.linalg.lstsq(design, targets).solution
    for _ in range(3):
        residuals = targets - design @ coefficients
        centered_residuals = residuals - torch.median(residuals[inliers])
        mad = torch.median(centered_residuals[inliers].abs())
        tolerance = torch.maximum(
            3.0 * 1.4826 * mad,
            torch.tensor(0.02 * scene_scale, device=mad.device, dtype=mad.dtype),
        )
        next_inliers = centered_residuals.abs() <= tolerance
        if next_inliers.sum() < 3 or torch.equal(next_inliers, inliers):
            break
        inliers = next_inliers
        coefficients = torch.linalg.lstsq(design[inliers], targets[inliers]).solution
    ground_coordinate_down = coefficients[2]
    ground_residuals = targets[inliers] - design[inliers] @ coefficients
    if not torch.isfinite(ground_coordinate_down) or not torch.isfinite(ground_residuals).all():
        diagnostics.update(
            {
                "status": "fallback_nonfinite_ground_fit",
                "ground_candidate_count": int(ground_points.shape[0]),
                "ground_inlier_count": int(inliers.sum().item()),
                "final_world_coordinate_xyz": diagnostics["initial_world_coordinate_xyz"],
            }
        )
        return initial_position, diagnostics

    transform = _asset_transform_for_frame(spec, frame_index, initial_position.device)
    if transform is None:
        raise ValueError(f"Asset {spec['id']} is not active in localization frame {frame_index}.")
    asset_mask = asset["opacities"] >= float(spec["contact_opacity_threshold"])
    if not asset_mask.any():
        asset_mask = torch.ones_like(asset["opacities"], dtype=torch.bool)
    rotation = transform[:3, :3]
    if rotation_delta_world is not None:
        rotation = torch.as_tensor(
            rotation_delta_world, device=rotation.device, dtype=rotation.dtype
        ) @ rotation
    relative_asset_means = (
        asset["means"][asset_mask]
        * float(spec["scale"])
        * float(asset_scale_factor)
        * scene_scale
    ) @ rotation.transpose(0, 1)
    asset_down_coordinates = relative_asset_means @ down
    asset_bottom_offset = torch.quantile(asset_down_coordinates, float(spec["contact_quantile"]))
    clearance = float(spec["ground_clearance_m"]) * scene_scale
    current_bottom_coordinate = initial_position @ down + asset_bottom_offset
    vertical_adjustment = ground_coordinate_down - clearance - current_bottom_coordinate
    corrected_position = initial_position + vertical_adjustment * down

    diagnostics.update(
        {
            "status": "snapped",
            "ground_candidate_count": int(ground_points.shape[0]),
            "ground_inlier_count": int(inliers.sum().item()),
            "ground_coordinate_along_down": float(ground_coordinate_down.item()),
            "ground_fit_rmse": float(torch.sqrt(torch.mean(ground_residuals.square())).item()),
            "asset_contact_point_count": int(asset_mask.sum().item()),
            "asset_bottom_offset_along_down": float(asset_bottom_offset.item()),
            "vertical_adjustment_world_units": float(vertical_adjustment.item()),
            "vertical_adjustment_m": float(vertical_adjustment.item() / scene_scale),
            "ground_clearance_m": float(spec["ground_clearance_m"]),
            "final_world_coordinate_xyz": corrected_position.detach().cpu().tolist(),
        }
    )
    return corrected_position, diagnostics


def get_assets_for_frame(
    manifest: dict[str, Any],
    asset_cache: dict[str, dict[str, torch.Tensor]],
    scene_name: str,
    frame_index: int,
    scene_units_per_meter: float,
    placement_points_world: dict[str, torch.Tensor] | None = None,
    rotation_deltas_world: dict[str, torch.Tensor] | None = None,
    scale_factors: dict[str, float] | None = None,
    only_asset_ids: set[str] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Return all configured external assets that should be rendered in one frame."""
    transformed_assets = []
    for spec in manifest["assets"]:
        if only_asset_ids is not None and spec["id"] not in only_asset_ids:
            continue
        if not _scene_matches(spec, scene_name):
            continue
        asset = asset_cache[spec["_asset_path"]]
        transform = _asset_transform_for_frame(spec, frame_index, asset["means"].device)
        if transform is None:
            continue
        if placement_points_world is not None and spec["id"] in placement_points_world:
            transform = transform.clone()
            transform[:3, 3] = torch.as_tensor(
                placement_points_world[spec["id"]],
                device=transform.device,
                dtype=transform.dtype,
            ) / float(scene_units_per_meter)
        if rotation_deltas_world is not None and spec["id"] in rotation_deltas_world:
            rotation_delta = torch.as_tensor(
                rotation_deltas_world[spec["id"]],
                device=transform.device,
                dtype=transform.dtype,
            )
            transform[:3, :3] = rotation_delta @ transform[:3, :3]
        transformed = transform_asset_gaussians(
            asset,
            transform,
            asset_scale=float(spec["scale"])
            * (float(scale_factors[spec["id"]]) if scale_factors and spec["id"] in scale_factors else 1.0),
            scene_units_per_meter=scene_units_per_meter,
        )
        opacity_scale = float(spec["opacity_scale"])
        if opacity_scale != 1.0:
            transformed["opacities"] = (transformed["opacities"] * opacity_scale).clamp(0, 1)
        transformed_assets.append(transformed)

    if not transformed_assets:
        return None
    return {name: torch.cat([asset[name] for asset in transformed_assets], dim=0) for name in GAUSSIAN_FIELDS}
