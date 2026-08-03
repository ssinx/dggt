#!/usr/bin/env python3
"""Generate structured weak labels for driving-scene images with Gemini.

Install the small, separate dependency set first:
    pip install -r requirements_annotation.txt

Set an API key without placing it in source code:
    export GEMINI_API_KEY='...'

Annotate one image:
    python scripts/gemini_scene_annotate.py path/to/frame.jpg --output-dir outputs/gemini

Annotate every supported image in a directory:
    python scripts/gemini_scene_annotate.py path/to/images --output-dir outputs/gemini

Render visualization files for annotations that already exist:
    python scripts/gemini_scene_annotate.py path/to/images --output-dir outputs/gemini --visualize-only

The generated labels are image-plane weak annotations.  Lane points and boxes
use the [0, 1000] normalized image coordinate system, not metric BEV or 3D
coordinates.  Use a calibrated geometry stage to convert them to BEV/3D.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import ImageDraw, ImageFont


SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
COORDINATE_RANGE = 1000
LANE_COLORS = {
    "toward_image_top": "#00e5ff",
    "toward_image_bottom": "#00ff73",
    "left_to_right": "#ff9d00",
    "right_to_left": "#ff4d6d",
    "unknown": "#ffffff",
}
OBJECT_COLORS = {
    "car": "#00b4d8",
    "truck": "#ff9f1c",
    "bus": "#f15bb5",
    "motorcycle": "#9b5de5",
    "bicycle": "#00f5d4",
    "pedestrian": "#f9c74f",
    "emergency_vehicle": "#ef233c",
    "other": "#ffffff",
}
LIGHT_COLORS = {"red": "#ff1744", "yellow": "#ffea00", "green": "#00e676", "off": "#b0bec5", "unknown": "#ffffff"}

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "coordinate_frame", "scene_summary", "lanes", "traffic_lights", "objects", "notes"],
    "properties": {
        "schema_version": {"type": "string", "enum": ["driving_scene_v2"]},
        "coordinate_frame": {"type": "string", "enum": ["image_2d_normalized_1000"]},
        "scene_summary": {"type": "string"},
        "lanes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "image_region",
                    "flow_direction_in_image",
                    "maneuver_indications",
                    "evidence",
                    "centerline_points_2d_1000",
                    "visibility",
                    "confidence",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "image_region": {
                        "type": "string",
                        "enum": ["left_image", "center_image", "right_image", "spans_image", "unknown"],
                    },
                    "flow_direction_in_image": {
                        "type": "string",
                        "enum": ["toward_image_top", "toward_image_bottom", "left_to_right", "right_to_left", "unknown"],
                    },
                    "maneuver_indications": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["left", "right", "straight", "u_turn", "none", "unknown"]},
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["lane_markings", "road_edge", "arrow_marking", "vehicles", "traffic_signal", "other", "unknown"]},
                    },
                    "centerline_points_2d_1000": {"$ref": "#/$defs/polyline"},
                    "visibility": {"type": "string", "enum": ["visible", "partially_occluded", "unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "traffic_lights": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "bbox_2d_1000", "state", "associated_lane_ids", "association_confidence", "visibility", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "bbox_2d_1000": {"$ref": "#/$defs/bbox"},
                    "state": {"type": "string", "enum": ["red", "yellow", "green", "off", "unknown"]},
                    "associated_lane_ids": {"type": "array", "items": {"type": "string"}},
                    "association_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "visibility": {"type": "string", "enum": ["visible", "partially_occluded", "unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "category", "bbox_2d_1000", "motion_cue", "visibility", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "emergency_vehicle", "other"],
                    },
                    "bbox_2d_1000": {"$ref": "#/$defs/bbox"},
                    "motion_cue": {"type": "string", "enum": ["likely_moving", "likely_stationary", "unknown"]},
                    "visibility": {"type": "string", "enum": ["visible", "partially_occluded", "unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "point": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "integer", "minimum": 0, "maximum": COORDINATE_RANGE},
        },
        "polyline": {"type": "array", "minItems": 2, "items": {"$ref": "#/$defs/point"}},
        "bbox": {
            "type": "array",
            "description": "[ymin, xmin, ymax, xmax] in normalized image coordinates [0, 1000].",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 0, "maximum": COORDINATE_RANGE},
        },
    },
}

PROMPT = """You annotate one driving-scene camera image for a dataset.
Return only JSON that conforms to the supplied schema.

Rules:
- The camera may face forward, rearward, sideways, or obliquely. Its calibration, vehicle-relative orientation, ego pose, and temporal context are unavailable. Do not assume that the camera is forward-facing or that the ego vehicle is visible.
- This is a 2D image annotation task. Do not infer metric BEV coordinates, 3D boxes, distances, hidden lanes, or ego-frame lane positions.
- Coordinates use the original image plane, normalized to integers in [0, 1000].
- A bounding box is exactly [ymin, xmin, ymax, xmax].
- Annotate each lane as an image-local visible road segment. IDs are unique only within this image. Order its centerline points along the visible segment; do not apply a near/far convention.
- `image_region` is only left/center/right in the image plane. Never use it to infer left/right of the ego vehicle.
- `flow_direction_in_image` describes only the projected direction visually suggested in the image. It is not same/opposite direction relative to the ego vehicle. Use `unknown` if lane direction cannot be supported by markings, arrows, or traffic behavior.
- `maneuver_indications` records only a visible arrow marking or strongly supported lane geometry, expressed relative to travel along that lane. It does not assert a legal permission; use `none` when no maneuver indication is visible and `unknown` when it is unreadable.
- `evidence` must list the visual cues that support each lane segment.
- Associate a traffic light with a lane only when the image provides strong evidence that it faces or is relevant to that lane. `associated_lane_ids` is not a claim that the light legally controls the lane; otherwise emit an empty list and association_confidence 0.
- Detect salient traffic participants only. A single image cannot establish actual motion: use `motion_cue=unknown` unless there is strong visible evidence for likely_moving or likely_stationary.
- Mark occluded, unreadable, or uncertain content as `unknown` / `partially_occluded` and lower confidence.
- Do not add prose or Markdown outside the JSON response.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create structured Gemini weak labels for driving images.")
    parser.add_argument("input", type=Path, nargs="?", help="An image file or a directory containing images.")
    parser.add_argument("--output-dir", type=Path, help="Directory for one JSON file per image.")
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model name; override if your account uses another model.")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--list-models", action="store_true", help="List Gemini models visible to this API key, then exit.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover images when input is a directory.")
    parser.add_argument("--max-images", type=int, help="Optional cap for a directory input.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate JSON files that already exist.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after a transient API error.")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Do not save the overlay image alongside JSON.")
    parser.add_argument("--visualize-only", action="store_true", help="Render existing JSON annotations without calling Gemini.")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()

    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
    if not args.list_models and args.output_dir is None:
        parser.error("--output-dir is required unless --list-models is used")
    return args


def discover_images(input_path: Path, recursive: bool, max_images: int | None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise ValueError(f"No supported images found in {input_path}")
    return images


def image_mime_type(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Cannot infer image MIME type for {path}")
    return mime_type


def output_path_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        relative_path = image_path.relative_to(input_root)
        return (output_dir / relative_path).with_suffix(".json")
    return output_dir / f"{image_path.stem}.json"


def visualization_path_for(annotation_path: Path) -> Path:
    return annotation_path.with_suffix(".vis.png")


def parse_json_response(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text[:-3]
    annotation = json.loads(text)
    if not isinstance(annotation, dict):
        raise ValueError("Gemini response is not a JSON object")
    return annotation


def validate_annotation(annotation: dict[str, Any]) -> None:
    required_keys = {"schema_version", "coordinate_frame", "scene_summary", "lanes", "traffic_lights", "objects", "notes"}
    missing_keys = required_keys.difference(annotation)
    if missing_keys:
        raise ValueError(f"Gemini response misses required keys: {sorted(missing_keys)}")
    if annotation["schema_version"] != "driving_scene_v2":
        raise ValueError(f"Unexpected schema_version: {annotation['schema_version']!r}")
    if annotation["coordinate_frame"] != "image_2d_normalized_1000":
        raise ValueError(f"Unexpected coordinate_frame: {annotation['coordinate_frame']!r}")
    for lane in annotation["lanes"]:
        validate_entity_keys(
            lane,
            {"id", "image_region", "flow_direction_in_image", "maneuver_indications", "evidence", "centerline_points_2d_1000", "visibility", "confidence"},
            "lane",
        )
        validate_polyline(lane["centerline_points_2d_1000"], f"lane {lane['id']}")
    for traffic_light in annotation["traffic_lights"]:
        validate_entity_keys(
            traffic_light,
            {"id", "bbox_2d_1000", "state", "associated_lane_ids", "association_confidence", "visibility", "confidence"},
            "traffic_light",
        )
        validate_bbox(traffic_light["bbox_2d_1000"], f"traffic_light {traffic_light['id']}")
    for obj in annotation["objects"]:
        validate_entity_keys(obj, {"id", "category", "bbox_2d_1000", "motion_cue", "visibility", "confidence"}, "object")
        validate_bbox(obj["bbox_2d_1000"], f"object {obj['id']}")


def validate_entity_keys(entity: Any, required_keys: set[str], label: str) -> None:
    if not isinstance(entity, dict):
        raise ValueError(f"{label} is not a JSON object")
    missing_keys = required_keys.difference(entity)
    if missing_keys:
        raise ValueError(f"{label} misses required keys: {sorted(missing_keys)}")


def validate_polyline(points: Any, label: str) -> None:
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"{label} must contain at least two polyline points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or not all(isinstance(value, int) for value in point):
            raise ValueError(f"{label} contains an invalid point: {point!r}")
        if not all(0 <= value <= COORDINATE_RANGE for value in point):
            raise ValueError(f"{label} point is outside [0, {COORDINATE_RANGE}]: {point!r}")


def validate_bbox(bbox: Any, label: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, int) for value in bbox):
        raise ValueError(f"{label} has an invalid bounding box: {bbox!r}")
    ymin, xmin, ymax, xmax = bbox
    if not (0 <= ymin < ymax <= COORDINATE_RANGE and 0 <= xmin < xmax <= COORDINATE_RANGE):
        raise ValueError(f"{label} has an invalid box extent: {bbox!r}")


def normalized_point_to_pixel(point: list[int], width: int, height: int) -> tuple[int, int]:
    x, y = point
    return round(x / COORDINATE_RANGE * width), round(y / COORDINATE_RANGE * height)


def normalized_bbox_to_pixels(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    x1, y1 = normalized_point_to_pixel([xmin, ymin], width, height)
    x2, y2 = normalized_point_to_pixel([xmax, ymax], width, height)
    return x1, y1, x2, y2


def draw_label(draw: ImageDraw.ImageDraw, anchor: tuple[int, int], text: str, color: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = draw.textbbox(anchor, text, font=font)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill="#000000cc")
    draw.text(anchor, text, fill=color, font=font)


def render_visualization(image_path: Path, annotation: dict[str, Any], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Missing Pillow. Run: pip install -r requirements_annotation.txt") from error

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(2, round(min(width, height) / 480))

    for lane in annotation["lanes"]:
        color = LANE_COLORS[lane["flow_direction_in_image"]]
        points = [normalized_point_to_pixel(point, width, height) for point in lane["centerline_points_2d_1000"]]
        draw.line(points, fill=color, width=line_width * 2)
        maneuvers = "/".join(lane["maneuver_indications"])
        label = f"{lane['id']} {lane['flow_direction_in_image']} {maneuvers} {lane['confidence']:.2f}"
        draw_label(draw, points[0], label, color, font)

    for traffic_light in annotation["traffic_lights"]:
        color = LIGHT_COLORS[traffic_light["state"]]
        x1, y1, x2, y2 = normalized_bbox_to_pixels(traffic_light["bbox_2d_1000"], width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width * 2)
        lane_ids = ",".join(traffic_light["associated_lane_ids"]) or "unlinked"
        label = f"light:{traffic_light['state']} ~ {lane_ids} {traffic_light['association_confidence']:.2f}"
        draw_label(draw, (x1, max(0, y1 - 12)), label, color, font)

    for obj in annotation["objects"]:
        color = OBJECT_COLORS[obj["category"]]
        x1, y1, x2, y2 = normalized_bbox_to_pixels(obj["bbox_2d_1000"], width, height)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width * 2)
        draw_label(draw, (x1, max(0, y1 - 12)), f"{obj['category']} {obj['motion_cue']} {obj['confidence']:.2f}", color, font)

    summary = f"lanes={len(annotation['lanes'])} lights={len(annotation['traffic_lights'])} objects={len(annotation['objects'])}"
    draw_label(draw, (8, 8), summary, "#ffffff", font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def annotate_image(client: Any, model: str, image_path: Path, retries: int) -> dict[str, Any]:
    from google.genai import types

    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=image_mime_type(image_path))
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=SCENE_SCHEMA)

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=[image_part, PROMPT], config=config)
            if not response.text:
                raise RuntimeError("Gemini returned an empty response")
            annotation = parse_json_response(response.text)
            validate_annotation(annotation)
            return annotation
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)

    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if args.list_models:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Set the {args.api_key_env} environment variable before running this script.", file=sys.stderr)
            return 2
        try:
            from google import genai
        except ImportError:
            print("Missing dependency. Run: pip install -r requirements_annotation.txt", file=sys.stderr)
            return 2
        client = genai.Client(api_key=api_key)
        for model in client.models.list():
            if model.name and "gemini" in model.name:
                print(model.name)
        return 0
    if args.input is None:
        print("Provide an input image/directory, or use --list-models.", file=sys.stderr)
        return 2

    try:
        images = discover_images(args.input, args.recursive, args.max_images)
    except (FileNotFoundError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

    client = None
    if not args.visualize_only:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Set the {args.api_key_env} environment variable before running this script.", file=sys.stderr)
            return 2
        try:
            from google import genai
        except ImportError:
            print("Missing dependency. Run: pip install -r requirements_annotation.txt", file=sys.stderr)
            return 2
        client = genai.Client(api_key=api_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, image_path in enumerate(images, start=1):
        output_path = output_path_for(image_path, args.input, args.output_dir)
        visualization_path = visualization_path_for(output_path)
        if args.visualize_only:
            if not output_path.exists():
                failures += 1
                print(f"[{index}/{len(images)}] Missing annotation JSON: {output_path}", file=sys.stderr)
                continue
            try:
                annotation = json.loads(output_path.read_text(encoding="utf-8"))
                validate_annotation(annotation)
                render_visualization(image_path, annotation, visualization_path)
                print(f"[{index}/{len(images)}] Saved {visualization_path}")
            except Exception as error:
                failures += 1
                print(f"[{index}/{len(images)}] Visualization failed: {error}", file=sys.stderr)
            continue
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] Skipping existing {output_path}")
            continue

        print(f"[{index}/{len(images)}] Annotating {image_path}")
        try:
            assert client is not None
            annotation = annotate_image(client, args.model, image_path, args.retries)
            annotation["metadata"] = {
                "source_image": str(image_path),
                "model": args.model,
                "coordinate_system": "2D normalized image coordinates [0, 1000]",
                "ego_frame_semantics": "not inferred from an arbitrary-orientation single camera image",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  Saved {output_path}")
            if args.visualize:
                render_visualization(image_path, annotation, visualization_path)
                print(f"  Saved {visualization_path}")
        except Exception as error:
            failures += 1
            print(f"  Failed: {error}", file=sys.stderr)
            if "NOT_FOUND" in str(error) or "no longer available" in str(error):
                print(
                    f"  Model {args.model!r} is unavailable to this key. "
                    "Run with --list-models and pass an available ID with --model.",
                    file=sys.stderr,
                )

    if failures:
        print(f"Completed with {failures} failed image(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
