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
from typing import Any


SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
COORDINATE_RANGE = 1000

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "scene_summary", "lanes", "traffic_lights", "objects", "notes"],
    "properties": {
        "schema_version": {"type": "string", "enum": ["driving_scene_v1"]},
        "scene_summary": {"type": "string"},
        "lanes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "role",
                    "travel_direction",
                    "allowed_maneuvers",
                    "centerline_points_2d_1000",
                    "visibility",
                    "confidence",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["ego", "left_of_ego", "right_of_ego", "oncoming", "crossing", "unknown"],
                    },
                    "travel_direction": {"type": "string", "enum": ["same", "opposite", "crossing", "unknown"]},
                    "allowed_maneuvers": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["left", "right", "straight", "u_turn", "unknown"]},
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
                "required": ["id", "bbox_2d_1000", "state", "controlled_lane_ids", "visibility", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "bbox_2d_1000": {"$ref": "#/$defs/bbox"},
                    "state": {"type": "string", "enum": ["red", "yellow", "green", "off", "unknown"]},
                    "controlled_lane_ids": {"type": "array", "items": {"type": "string"}},
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
                "required": ["id", "category", "bbox_2d_1000", "dynamic", "visibility", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "emergency_vehicle", "other"],
                    },
                    "bbox_2d_1000": {"$ref": "#/$defs/bbox"},
                    "dynamic": {"type": "boolean"},
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

PROMPT = """You annotate a forward-facing driving-scene image for a dataset.
Return only JSON that conforms to the supplied schema.

Rules:
- This is a 2D image annotation task. Do not infer metric BEV coordinates, 3D boxes, distances, or hidden lanes.
- Coordinates use the original image plane, normalized to integers in [0, 1000].
- A bounding box is exactly [ymin, xmin, ymax, xmax].
- Lane centerlines are ordered from image bottom (near) toward image top (far). Include only clearly visible lane segments.
- `allowed_maneuvers` means the legal or visually indicated outgoing movement. Use `unknown` when evidence is insufficient.
- Associate a traffic light with a lane only when the association is visually well supported; otherwise emit an empty list.
- Detect salient traffic participants only. Mark occluded, unreadable, or uncertain content as `unknown` / `partially_occluded` and lower confidence.
- Do not add prose or Markdown outside the JSON response.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create structured Gemini weak labels for driving images.")
    parser.add_argument("input", type=Path, help="An image file or a directory containing images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for one JSON file per image.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name; override if your account uses another model.")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover images when input is a directory.")
    parser.add_argument("--max-images", type=int, help="Optional cap for a directory input.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate JSON files that already exist.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after a transient API error.")
    args = parser.parse_args()

    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.retries < 0:
        parser.error("--retries must be zero or greater")
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
    required_keys = {"schema_version", "scene_summary", "lanes", "traffic_lights", "objects", "notes"}
    missing_keys = required_keys.difference(annotation)
    if missing_keys:
        raise ValueError(f"Gemini response misses required keys: {sorted(missing_keys)}")
    if annotation["schema_version"] != "driving_scene_v1":
        raise ValueError(f"Unexpected schema_version: {annotation['schema_version']!r}")
    for lane in annotation["lanes"]:
        validate_polyline(lane["centerline_points_2d_1000"], f"lane {lane['id']}")
    for entity_type in ("traffic_lights", "objects"):
        for entity in annotation[entity_type]:
            validate_bbox(entity["bbox_2d_1000"], f"{entity_type} {entity['id']}")


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


def annotate_image(client: Any, model: str, image_path: Path, retries: int) -> dict[str, Any]:
    from google.genai import types

    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=image_mime_type(image_path))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=SCENE_SCHEMA,
        temperature=0.1,
    )

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
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Set the {args.api_key_env} environment variable before running this script.", file=sys.stderr)
        return 2

    try:
        from google import genai
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements_annotation.txt", file=sys.stderr)
        return 2

    try:
        images = discover_images(args.input, args.recursive, args.max_images)
    except (FileNotFoundError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)
    failures = 0
    for index, image_path in enumerate(images, start=1):
        output_path = output_path_for(image_path, args.input, args.output_dir)
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] Skipping existing {output_path}")
            continue

        print(f"[{index}/{len(images)}] Annotating {image_path}")
        try:
            annotation = annotate_image(client, args.model, image_path, args.retries)
            annotation["metadata"] = {
                "source_image": str(image_path),
                "model": args.model,
                "coordinate_system": "2D normalized image coordinates [0, 1000]",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  Saved {output_path}")
        except Exception as error:
            failures += 1
            print(f"  Failed: {error}", file=sys.stderr)

    if failures:
        print(f"Completed with {failures} failed image(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
