"""Extract binary sky masks from ``<data_root>/images`` with SegFormer.

Example:
    python datasets/tools/extract_sky_masks.py \
        --data_root /path/to/xxx \
        --segformer_path ../SegFormer \
        --checkpoint ../SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth

The output for ``<data_root>/images/frame.jpg`` is written to
``<data_root>/sky_masks/frame.png``. White pixels (255) denote the Cityscapes
``sky`` class (semantic ID 10), and black pixels (0) denote all other classes.
It also writes same-sized black placeholder masks to
``<data_root>/fine_dynamic_masks/{human,vehicle,all}/frame.png``.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

SKY_CLASS_ID = 10
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_SEGFORMER_PATH = Path(__file__).resolve().parents[3] / "SegFormer"
DYNAMIC_MASK_CLASSES = ("human", "vehicle", "all")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Directory containing the input images/ directory.",
    )
    parser.add_argument(
        "--segformer_path",
        type=Path,
        default=DEFAULT_SEGFORMER_PATH,
        help=f"SegFormer repository path (default: {DEFAULT_SEGFORMER_PATH}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="SegFormer Cityscapes configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SegFormer Cityscapes checkpoint file.",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process images in subdirectories of images/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks that already exist.",
    )
    return parser.parse_args()


def find_images(image_dir: Path, recursive: bool):
    paths = image_dir.rglob("*") if recursive else image_dir.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_paths(args):
    image_dir = args.data_root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Input image directory does not exist: {image_dir}")

    if args.config is None:
        args.config = args.segformer_path / "local_configs" / "segformer" / "B5" / "segformer.b5.1024x1024.city.160k.py"
    if args.checkpoint is None:
        args.checkpoint = args.segformer_path / "pretrained" / "segformer.b5.1024x1024.city.160k.pth"
    return image_dir, args.data_root / "sky_masks", args.data_root / "fine_dynamic_masks"


def validate_segformer_paths(args):
    for label, path in (("SegFormer repository", args.segformer_path), ("config", args.config), ("checkpoint", args.checkpoint)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} does not exist: {path}\n"
                "Install SegFormer and download the Cityscapes B5 checkpoint, "
                "or pass the correct --segformer_path, --config, and --checkpoint."
            )


def import_mmseg(segformer_path: Path):
    if str(segformer_path) not in sys.path:
        sys.path.insert(0, str(segformer_path))
    try:
        from mmseg.apis import inference_segmentor, init_segmentor
    except ImportError as error:
        raise ImportError(
            "Could not import SegFormer dependencies. Run this script in the "
            "SegFormer environment after installing its requirements."
        ) from error
    return inference_segmentor, init_segmentor


def main():
    args = parse_args()
    image_dir, sky_mask_dir, fine_dynamic_mask_dir = validate_paths(args)
    image_paths = find_images(image_dir, args.recursive)
    if not image_paths:
        print(f"No supported images found in {image_dir}")
        return

    try:
        import imageio.v2 as imageio
        import numpy as np
        from tqdm import tqdm
    except ImportError as error:
        raise ImportError(
            "Could not import imageio, numpy, or tqdm. Run this script in the "
            "SegFormer environment after installing its requirements."
        ) from error

    output_paths = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(image_dir).with_suffix(".png")
        sky_mask_path = sky_mask_dir / relative_path
        dynamic_mask_paths = {
            dynamic_class: fine_dynamic_mask_dir / dynamic_class / relative_path
            for dynamic_class in DYNAMIC_MASK_CLASSES
        }
        output_paths.append((image_path, sky_mask_path, dynamic_mask_paths))

    requires_segmentation = any(args.overwrite or not sky_mask_path.exists() for _, sky_mask_path, _ in output_paths)
    if requires_segmentation:
        validate_segformer_paths(args)
        inference_segmentor, init_segmentor = import_mmseg(args.segformer_path)
        model = init_segmentor(str(args.config), str(args.checkpoint), device=args.device)
    else:
        inference_segmentor = None
        model = None

    sky_written_count = 0
    sky_skipped_count = 0
    dynamic_written_count = 0
    dynamic_skipped_count = 0
    for image_path, sky_mask_path, dynamic_mask_paths in tqdm(output_paths, desc="Extracting sky masks"):
        write_sky_mask = args.overwrite or not sky_mask_path.exists()
        dynamic_paths_to_write = [
            dynamic_mask_path
            for dynamic_mask_path in dynamic_mask_paths.values()
            if args.overwrite or not dynamic_mask_path.exists()
        ]

        if write_sky_mask:
            sky_mask_path.parent.mkdir(parents=True, exist_ok=True)
            segmentation = np.asarray(inference_segmentor(model, str(image_path))[0])
            sky_mask = (segmentation == SKY_CLASS_ID).astype(np.uint8) * 255
            imageio.imwrite(sky_mask_path, sky_mask)
            sky_written_count += 1
        else:
            sky_skipped_count += 1

        if dynamic_paths_to_write:
            if write_sky_mask:
                black_mask = np.zeros_like(sky_mask)
            else:
                image = imageio.imread(image_path)
                black_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            for dynamic_mask_path in dynamic_paths_to_write:
                dynamic_mask_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(dynamic_mask_path, black_mask)
                dynamic_written_count += 1
        dynamic_skipped_count += len(dynamic_mask_paths) - len(dynamic_paths_to_write)

    print(
        f"Saved {sky_written_count} sky masks to {sky_mask_dir}; "
        f"skipped {sky_skipped_count} existing sky masks.\n"
        f"Saved {dynamic_written_count} black dynamic-mask placeholders to {fine_dynamic_mask_dir}; "
        f"skipped {dynamic_skipped_count} existing placeholders."
    )


if __name__ == "__main__":
    main()
