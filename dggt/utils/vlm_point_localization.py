"""Locate prompt-defined points in rendered-video frames with a Qwen VLM."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Iterable

import imageio
from PIL import Image, ImageDraw


COORDINATE_RANGE = 1000


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
) -> list[dict[str, Any]]:
    image_data_url = _frame_to_png_data_url(frame)
    instruction = (
        "You locate visual points in one rendered driving-scene image. Find every visible point that "
        "satisfies the user's request. For each result, return the center of the matched object or the "
        "exact requested feature point. Coordinates are image-plane [x, y] integers normalized to [0, 1000], "
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
