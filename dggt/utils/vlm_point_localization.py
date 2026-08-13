"""Locate prompt-defined points in rendered-video frames with a Qwen VLM."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Iterable

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


COORDINATE_RANGE = 1000


def localize_corresponding_points_in_frame(
    client: Any,
    model: str,
    prompt: str,
    front_frame: Any,
    birds_eye_frame: Any,
    front_extrinsic: torch.Tensor,
    front_intrinsic: torch.Tensor,
    birds_eye_extrinsic: torch.Tensor,
    birds_eye_intrinsic: torch.Tensor,
    output_path: str | Path,
    frame_index: int,
    retries: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Locate corresponding points in one frame and triangulate their world positions."""
    front_frame = _ensure_rgb_frame(front_frame)
    birds_eye_frame = _ensure_rgb_frame(birds_eye_frame)
    front_height, front_width = front_frame.shape[:2]
    birds_height, birds_width = birds_eye_frame.shape[:2]
    front_detections = _request_point_detections(
        client,
        model,
        prompt,
        front_frame,
        retries,
        enable_thinking,
        selection_mode="empty_placement_point",
    )
    front_points = [_to_pixel_coordinates(point, front_width, front_height) for point in front_detections]

    correspondences = []
    for front_point in front_points:
        front_pixel = front_point["coordinate_2d_pixels_xy"]
        line = epipolar_line_from_pixel(
            front_pixel,
            front_extrinsic,
            front_intrinsic,
            birds_eye_extrinsic,
            birds_eye_intrinsic,
        )
        annotated_birds_eye = _draw_constraint_line(birds_eye_frame, line)
        bird_prompt = (
            f"Find the same EMPTY PLACEMENT POINT labeled {front_point['label']!r} that was selected in the front view. "
            "This is a vacant ground location where a new asset will be inserted, not the location of an existing object. "
            "The valid location is restricted to the red epipolar line drawn on this bird's-eye image. "
            "Return exactly one point on that red line. Original request: " + prompt
        )
        detections = _request_point_detections(
            client,
            model,
            bird_prompt,
            annotated_birds_eye,
            retries,
            enable_thinking,
            selection_mode="empty_placement_point",
        )
        if len(detections) != 1:
            raise ValueError(
                f"Expected exactly one bird's-eye correspondence for {front_point['label']!r}, got {len(detections)}."
            )
        bird_point = _to_pixel_coordinates(detections[0], birds_width, birds_height)
        constrained_pixel = project_pixel_to_line(bird_point["coordinate_2d_pixels_xy"], line)
        bird_point["vlm_coordinate_2d_pixels_xy"] = bird_point["coordinate_2d_pixels_xy"]
        bird_point["coordinate_2d_pixels_xy"] = [float(value) for value in constrained_pixel]
        world_point, ray_error = triangulate_pixels(
            front_pixel,
            constrained_pixel,
            front_extrinsic,
            front_intrinsic,
            birds_eye_extrinsic,
            birds_eye_intrinsic,
        )
        correspondences.append(
            {
                "label": front_point["label"],
                "front_point": front_point,
                "birds_eye_point": bird_point,
                "birds_eye_epipolar_line": [float(value) for value in line],
                "world_coordinate_xyz": [float(value) for value in world_point],
                "triangulation_ray_error": float(ray_error),
            }
        )

    output_path = Path(output_path)
    visualization_dir = output_path.parent / "vlm_point_visualizations"
    frame_suffix = f"{frame_index:04d}"
    _save_point_visualization(
        front_frame,
        front_points,
        visualization_dir / f"front_frame_{frame_suffix}.png",
    )
    _save_constrained_visualization(
        birds_eye_frame,
        correspondences,
        visualization_dir / f"birds_eye_frame_{frame_suffix}.png",
    )
    results = {
        "schema_version": "vlm_stereo_point_localization_v1",
        "prompt": prompt,
        "frame_index": frame_index,
        "coordinate_frame": "dggt_world_xyz",
        "correspondences": correspondences,
    }
    save_localization_results(results, output_path)
    return results


def save_localization_results(results: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refine_asset_orientation(
    client: Any,
    model: str,
    user_prompt: str,
    asset_id: str,
    preview_frames: list[tuple[str, int, Any]],
    debug_dir: str | Path,
    max_yaw_delta_deg: float,
    min_scale_factor: float,
    max_scale_factor: float,
    retries: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Ask a VLM for constrained yaw and uniform-scale corrections from several previews."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved_images = []
    content: list[dict[str, Any]] = []
    instruction = (
        "You inspect an initially inserted 3D Gaussian asset in a driving scene and decide whether its "
        "horizontal orientation and uniform physical scale need correction. Images are labeled as front-view "
        "or bird's-eye previews. "
        "Use all views together and follow the user's placement request. The asset's local coordinate axes are "
        "arbitrary and have no semantic meaning. Return a yaw correction about the scene's vertical axis and a "
        "uniform scale factor relative to the currently rendered asset. Do not propose pitch, roll, translation, "
        "or non-uniform scaling. Positive yaw means CLOCKWISE when viewed in the "
        "bird's-eye image, and negative yaw means counter-clockwise. If the current orientation already meets "
        "the request, return zero. A scale_factor below 1 makes the asset smaller and above 1 makes it larger; "
        "return 1 when its size is already plausible. Judge physical size from comparable scene objects, lane "
        "width, road geometry, and all views together. Do not shrink an asset merely because it is far from the "
        "camera. Return only one JSON object with no Markdown: "
        '{"yaw_delta_deg":0.0,"scale_factor":1.0,"confidence":0.0,"reason":"string"}. '
        f"The yaw correction must be in [-{max_yaw_delta_deg:.3f}, {max_yaw_delta_deg:.3f}] degrees and "
        f"scale_factor must be in [{min_scale_factor:.4f}, {max_scale_factor:.4f}].\n\n"
        f"Asset id: {asset_id}\nUser request: {user_prompt}"
    )
    content.append({"type": "input_text", "text": instruction})
    for image_index, (view_name, frame_index, frame) in enumerate(preview_frames):
        rgb_frame = _ensure_rgb_frame(frame)
        filename = f"input_{image_index:02d}_{view_name}_frame_{frame_index:04d}.png"
        image_path = debug_dir / filename
        Image.fromarray(rgb_frame).save(image_path)
        saved_images.append(str(image_path))
        content.append(
            {
                "type": "input_text",
                "text": f"Image {image_index + 1}: {view_name}, sequence frame {frame_index}.",
            }
        )
        content.append({"type": "input_image", "image_url": _frame_to_png_data_url(rgb_frame)})

    image_descriptions = [
        f"Image {index + 1}: {view_name}, sequence frame {frame_index}."
        for index, (view_name, frame_index, _) in enumerate(preview_frames)
    ]
    (debug_dir / "request.txt").write_text(
        instruction + "\n\n" + "\n".join(image_descriptions) + "\n",
        encoding="utf-8",
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                extra_body={"enable_thinking": enable_thinking},
            )
            raw_response = _response_text(response)
            (debug_dir / "response_raw.txt").write_text(raw_response + "\n", encoding="utf-8")
            parsed = json.loads(_strip_json_fence(raw_response))
            if not isinstance(parsed, dict):
                raise ValueError("Orientation response is not a JSON object.")
            yaw = parsed.get("yaw_delta_deg")
            scale_factor = parsed.get("scale_factor")
            confidence = parsed.get("confidence")
            reason = parsed.get("reason")
            if not isinstance(yaw, (int, float)) or not np.isfinite(yaw):
                raise ValueError("Orientation response has invalid yaw_delta_deg.")
            if not isinstance(scale_factor, (int, float)) or not np.isfinite(scale_factor):
                raise ValueError("Orientation response has invalid scale_factor.")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("Orientation response has invalid confidence.")
            if not isinstance(reason, str):
                raise ValueError("Orientation response has invalid reason.")
            requested_yaw = float(yaw)
            applied_yaw = float(np.clip(requested_yaw, -max_yaw_delta_deg, max_yaw_delta_deg))
            requested_scale_factor = float(scale_factor)
            applied_scale_factor = float(
                np.clip(requested_scale_factor, min_scale_factor, max_scale_factor)
            )
            result = {
                "status": "refined",
                "asset_id": asset_id,
                "yaw_convention": "positive_is_clockwise_in_birds_eye_view",
                "requested_yaw_delta_deg": requested_yaw,
                "applied_yaw_delta_deg": applied_yaw,
                "was_clamped": (
                    requested_yaw != applied_yaw
                    or requested_scale_factor != applied_scale_factor
                ),
                "requested_scale_factor": requested_scale_factor,
                "applied_scale_factor": applied_scale_factor,
                "yaw_was_clamped": requested_yaw != applied_yaw,
                "scale_was_clamped": requested_scale_factor != applied_scale_factor,
                "confidence": float(confidence),
                "reason": reason,
                "input_images": saved_images,
                "request_path": str(debug_dir / "request.txt"),
                "raw_response_path": str(debug_dir / "response_raw.txt"),
            }
            (debug_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return result
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(2**attempt)
    failure = {
        "status": "fallback_vlm_error",
        "asset_id": asset_id,
        "applied_yaw_delta_deg": 0.0,
        "applied_scale_factor": 1.0,
        "error": f"{type(last_error).__name__}: {last_error}",
        "input_images": saved_images,
        "request_path": str(debug_dir / "request.txt"),
    }
    (debug_dir / "result.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return failure


def _pixel_ray(pixel: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_h = torch.tensor([float(pixel[0]), float(pixel[1]), 1.0], device=extrinsic.device, dtype=extrinsic.dtype)
    direction_camera = torch.linalg.solve(intrinsic, pixel_h)
    rotation = extrinsic[:3, :3]
    origin = -(rotation.transpose(0, 1) @ extrinsic[:3, 3])
    direction = rotation.transpose(0, 1) @ direction_camera
    return origin, direction / torch.linalg.vector_norm(direction)


def epipolar_line_from_pixel(
    pixel: Any,
    source_extrinsic: torch.Tensor,
    source_intrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
) -> np.ndarray:
    """Return target-image line ax + by + c = 0 for a source pixel."""
    origin, direction = _pixel_ray(pixel, source_extrinsic, source_intrinsic)
    depths = torch.tensor([0.1, 1000.0], device=origin.device, dtype=origin.dtype)
    world_points = origin[None] + depths[:, None] * direction[None]
    camera_points = world_points @ target_extrinsic[:3, :3].transpose(0, 1) + target_extrinsic[:3, 3]
    projected = camera_points @ target_intrinsic.transpose(0, 1)
    projected = projected[:, :2] / projected[:, 2:3]
    point_a, point_b = projected.detach().cpu().double().numpy()
    line = np.cross(np.array([*point_a, 1.0]), np.array([*point_b, 1.0]))
    norm = np.linalg.norm(line[:2])
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Cannot construct a valid bird's-eye epipolar line for the selected front point.")
    return line / norm


def project_pixel_to_line(pixel: Any, line: Any) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64)
    point = np.asarray(pixel, dtype=np.float64)
    return point - line[:2] * (line[0] * point[0] + line[1] * point[1] + line[2])


def triangulate_pixels(
    source_pixel: Any,
    target_pixel: Any,
    source_extrinsic: torch.Tensor,
    source_intrinsic: torch.Tensor,
    target_extrinsic: torch.Tensor,
    target_intrinsic: torch.Tensor,
) -> tuple[np.ndarray, float]:
    """Triangulate as the midpoint of the closest points on two camera rays."""
    origin_a, direction_a = _pixel_ray(source_pixel, source_extrinsic, source_intrinsic)
    origin_b, direction_b = _pixel_ray(target_pixel, target_extrinsic, target_intrinsic)
    rhs = origin_b - origin_a
    system = torch.stack((direction_a, -direction_b), dim=1)
    distances = torch.linalg.lstsq(system, rhs).solution
    point_a = origin_a + distances[0] * direction_a
    point_b = origin_b + distances[1] * direction_b
    if torch.any(distances <= 0):
        raise ValueError("Triangulated point lies behind one of the selected-frame cameras.")
    midpoint = (point_a + point_b) * 0.5
    return midpoint.detach().cpu().numpy(), torch.linalg.vector_norm(point_a - point_b).item()


def _line_segment_in_image(line: Any, width: int, height: int) -> list[tuple[float, float]]:
    a, b, c = (float(value) for value in line)
    candidates = []
    if abs(b) > 1e-8:
        candidates.extend([(0.0, -c / b), (float(width - 1), -(a * (width - 1) + c) / b)])
    if abs(a) > 1e-8:
        candidates.extend([(-c / a, 0.0), (-(b * (height - 1) + c) / a, float(height - 1))])
    valid = [(x, y) for x, y in candidates if 0 <= x < width and 0 <= y < height]
    unique = []
    for point in valid:
        if not any(np.linalg.norm(np.subtract(point, other)) < 1e-4 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        raise ValueError("The front-view point's epipolar line does not cross the bird's-eye image.")
    return unique[:2]


def _draw_constraint_line(frame: Any, line: Any) -> np.ndarray:
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    draw.line(_line_segment_in_image(line, *image.size), fill="#ff1744", width=max(3, min(image.size) // 150))
    return np.asarray(image)


def _save_constrained_visualization(frame: Any, correspondences: list[dict[str, Any]], output_path: Path) -> None:
    annotated = frame.copy()
    for correspondence in correspondences:
        annotated = _draw_constraint_line(annotated, correspondence["birds_eye_epipolar_line"])
    points = [correspondence["birds_eye_point"] for correspondence in correspondences]
    _save_point_visualization(annotated, points, output_path)


def create_qwen_client(api_key_env: str, base_url: str) -> Any:
    """Create a Qwen OpenAI-compatible client using the configured API key."""
    import os

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Set the {api_key_env} environment variable before enabling VLM point localization.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Missing optional VLM dependency. Run: pip install -r requirements_annotation.txt"
        ) from error
    return OpenAI(api_key=api_key, base_url=base_url)


def localize_points_in_videos(
    client: Any,
    model: str,
    prompt: str,
    video_paths: Iterable[str | Path],
    frame_stride: int,
    output_path: str | Path,
    retries: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Sample rendered videos and write prompt-matching point detections to JSON."""
    if frame_stride < 1:
        raise ValueError("VLM frame stride must be positive.")
    if retries < 0:
        raise ValueError("VLM retries must be zero or greater.")

    results = {
        "schema_version": "vlm_point_localization_v2",
        "prompt": prompt,
        "frame_stride": frame_stride,
        "coordinate_frame": "image_2d_pixels_xy_origin_top_left",
        "normalization_coordinate_frame": "image_2d_normalized_xy_1000",
        "videos": [],
    }
    visualization_dir = Path(output_path).parent / "vlm_point_visualizations"
    for video_path in video_paths:
        video_path = Path(video_path)
        results["videos"].append(
            _localize_points_in_video(
                client,
                model,
                prompt,
                video_path,
                frame_stride,
                retries,
                enable_thinking,
                visualization_dir / video_path.stem,
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def _localize_points_in_video(
    client: Any,
    model: str,
    prompt: str,
    video_path: Path,
    frame_stride: int,
    retries: int,
    enable_thinking: bool,
    visualization_dir: Path,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Rendered video was not created: {video_path}")

    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    frame_results = []
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index % frame_stride:
                continue
            frame = _ensure_rgb_frame(frame)
            height, width = frame.shape[:2]
            detections = _request_point_detections(
                client,
                model,
                prompt,
                frame,
                retries,
                enable_thinking,
            )
            points = [_to_pixel_coordinates(point, width, height) for point in detections]
            visualization_path = visualization_dir / f"frame_{frame_index:04d}.png"
            _save_point_visualization(frame, points, visualization_path)
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "image_size": {"width": width, "height": height},
                    "visualization_path": str(visualization_path),
                    "points": points,
                }
            )
    finally:
        reader.close()

    return {
        "video_path": str(video_path),
        "fps": metadata.get("fps"),
        "sampled_frames": frame_results,
    }


def _ensure_rgb_frame(frame: Any) -> Any:
    if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
        raise ValueError(f"Expected an RGB(A) video frame, got shape {frame.shape}.")
    return frame[:, :, :3]


def _request_point_detections(
    client: Any,
    model: str,
    user_prompt: str,
    frame: Any,
    retries: int,
    enable_thinking: bool,
    selection_mode: str = "visible_object_or_feature",
) -> list[dict[str, Any]]:
    image_data_url = _frame_to_png_data_url(frame)
    if selection_mode == "empty_placement_point":
        task_instruction = (
            "This is an ASSET INSERTION placement task, not an object-detection task. Infer how many new "
            "objects the user asks to add, and return exactly one placement point for each requested new object. "
            "Each returned point must be a currently EMPTY, visible ground contact location where the new object "
            "can be placed. NEVER return the center, wheels, footprint, or any point belonging to an existing "
            "vehicle, person, or obstacle. Existing objects may only be used as context for understanding roads, "
            "lanes, spacing, and alignment. Put the point on the supporting road/ground surface, approximately at "
            "the center of the future asset's ground footprint. The label must describe the vacant placement "
            "location (for example, 'empty parking position on the left roadside'), not an existing object's "
            "appearance or color. If no safe visible empty location satisfies the request, return an empty list. "
            "In Chinese: 请选择一个可添加新物体的空置地面点，绝对不要选择画面中已有物体上的点。"
        )
    elif selection_mode == "visible_object_or_feature":
        task_instruction = (
            "Find every visible point that satisfies the user's request. For each result, return the center of "
            "the matched object or the exact requested feature point. Do not infer occluded or off-image locations."
        )
    else:
        raise ValueError(f"Unknown VLM point selection mode: {selection_mode}")
    instruction = (
        "You locate visual points in one rendered driving-scene image. "
        + task_instruction
        + " Coordinates are image-plane [x, y] integers normalized to [0, 1000], "
        "where [0, 0] is the top-left corner. Do not infer occluded or off-image locations. "
        "Return only a JSON object in this exact format, with no Markdown or explanation: "
        '{"coordinate_frame":"image_2d_normalized_xy_1000","points":['
        '{"label":"string","point_2d_1000":[0,0],"confidence":0.0}]}. '
        "Return an empty points list when nothing matches.\n\n"
        f"User request: {user_prompt}"
    )

    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {"type": "input_image", "image_url": image_data_url},
                        ],
                    }
                ],
                extra_body={"enable_thinking": enable_thinking},
            )
            return _parse_point_response(_response_text(response))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _frame_to_png_data_url(frame: Any) -> str:
    from io import BytesIO

    image_bytes = BytesIO()
    Image.fromarray(frame).save(image_bytes, format="PNG")
    encoded_image = base64.b64encode(image_bytes.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def _response_text(response: Any) -> str:
    texts = []
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            text = getattr(content, "text", None)
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError("Qwen returned no final message text.")
    return "\n".join(texts)


def _parse_point_response(response_text: str) -> list[dict[str, Any]]:
    response = json.loads(_strip_json_fence(response_text))
    if not isinstance(response, dict):
        raise ValueError("VLM response is not a JSON object.")
    if response.get("coordinate_frame") != "image_2d_normalized_xy_1000":
        raise ValueError("VLM response uses an unexpected coordinate frame.")
    points = response.get("points")
    if not isinstance(points, list):
        raise ValueError("VLM response has no points list.")
    for point in points:
        _validate_point(point)
    return points


def _strip_json_fence(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _validate_point(point: Any) -> None:
    if not isinstance(point, dict):
        raise ValueError("VLM point is not a JSON object.")
    required_keys = {"label", "point_2d_1000", "confidence"}
    if required_keys.difference(point):
        raise ValueError(f"VLM point misses keys: {sorted(required_keys.difference(point))}")
    if not isinstance(point["label"], str):
        raise ValueError("VLM point label must be a string.")
    coordinates = point["point_2d_1000"]
    if (
        not isinstance(coordinates, list)
        or len(coordinates) != 2
        or not all(isinstance(value, int) and 0 <= value <= COORDINATE_RANGE for value in coordinates)
    ):
        raise ValueError(f"VLM point has invalid normalized coordinates: {coordinates!r}")
    confidence = point["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"VLM point has invalid confidence: {confidence!r}")


def _to_pixel_coordinates(point: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    normalized_x, normalized_y = point["point_2d_1000"]
    return {
        "label": point["label"],
        "confidence": point["confidence"],
        "coordinate_2d_normalized_xy_1000": [normalized_x, normalized_y],
        "coordinate_2d_pixels_xy": [
            round(normalized_x * max(width - 1, 0) / COORDINATE_RANGE),
            round(normalized_y * max(height - 1, 0) / COORDINATE_RANGE),
        ],
    }


def _save_point_visualization(frame: Any, points: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    marker_radius = max(4, round(min(width, height) * 0.012))

    for point in points:
        x_coordinate, y_coordinate = point["coordinate_2d_pixels_xy"]
        draw.ellipse(
            (
                x_coordinate - marker_radius,
                y_coordinate - marker_radius,
                x_coordinate + marker_radius,
                y_coordinate + marker_radius,
            ),
            fill="#ff1744",
            outline="#ffffff",
            width=max(1, marker_radius // 3),
        )
        label = f"{point['label']} {point['confidence']:.2f}"
        text_anchor = (
            min(x_coordinate + marker_radius + 3, max(width - 1, 0)),
            max(y_coordinate - marker_radius - 3, 0),
        )
        text_box = draw.textbbox(text_anchor, label)
        draw.rectangle(text_box, fill="#000000")
        draw.text(text_anchor, label, fill="#ffffff")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
