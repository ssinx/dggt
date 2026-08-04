#!/usr/bin/env python
"""Convert a complete Asset Harvester Gaussian PLY into a DGGT asset file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dggt.utils.assets import convert_asset_harvester_ply, save_gaussian_asset


def _identity_matrix() -> list[list[int]]:
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, help="Asset Harvester gaussians.ply path")
    parser.add_argument("--output", required=True, help="Output DGGT .pt asset path")
    parser.add_argument("--lwh-path", help="Optional lwh.txt path; defaults to <PLY parent>/multiview/lwh.txt")
    parser.add_argument("--max-gaussians", type=int, help="Optional deterministic uniform downsampling limit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for downsampling")
    parser.add_argument(
        "--keep-source-origin",
        action="store_true",
        help="Do not move the local origin to horizontal center and robust ground level",
    )
    parser.add_argument("--manifest-output", help="Optionally write a one-asset DGGT manifest JSON")
    parser.add_argument("--asset-id", default="external_asset", help="Asset ID used in the optional manifest")
    parser.add_argument("--scene-name", help="Optional DGGT scene name filter for the optional manifest")
    parser.add_argument(
        "--scene-units-per-meter",
        type=float,
        default=1.0,
        help="DGGT model-units-per-meter written to the optional manifest",
    )
    args = parser.parse_args()

    if args.scene_units_per_meter <= 0:
        parser.error("--scene-units-per-meter must be positive")

    asset = convert_asset_harvester_ply(
        args.input_ply,
        lwh_path=args.lwh_path,
        center_ground=not args.keep_source_origin,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
    )
    output_path = Path(args.output).resolve()
    save_gaussian_asset(asset, output_path)

    print(f"Saved {asset['means'].shape[0]:,} Gaussians to {output_path}")
    print("L/W/H (m):", ", ".join(f"{value:.3f}" for value in asset["lwh_m"].tolist()))
    print("Local origin moved by (m):", ", ".join(f"{value:.3f}" for value in asset["local_origin_m"].tolist()))

    if args.manifest_output:
        manifest_path = Path(args.manifest_output).resolve()
        asset_path = output_path.relative_to(manifest_path.parent) if output_path.is_relative_to(manifest_path.parent) else output_path
        asset_spec = {
            "id": args.asset_id,
            "asset_path": str(asset_path),
            "asset_to_world": _identity_matrix(),
            "scale": 1.0,
            "start_frame": 0,
        }
        if args.scene_name:
            asset_spec["scene_names"] = [args.scene_name]
        manifest = {
            "scene_units_per_meter": args.scene_units_per_meter,
            "assets": [asset_spec],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved example manifest to {manifest_path}")


if __name__ == "__main__":
    main()
