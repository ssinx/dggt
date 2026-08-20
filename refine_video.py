import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from third_party.difix.src.model import Difix


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refine every frame with DiFix and write a side-by-side comparison "
            "video (original on the left, refined on the right)."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/diffusion_model.pth"),
        help="DiFix checkpoint path (default: pretrained/diffusion_model.pth)",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=199,
        help="Single-step denoising timestep (default: 199)",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"DiFix checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("DiFix requires a CUDA-capable GPU.")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output video paths must be different.")


def refine_video(input_path, output_path, checkpoint_path, timestep):
    device = torch.device("cuda")
    model = Difix(
        pretrained_path=str(checkpoint_path),
        timestep=timestep,
        mv_unet=False,
    )
    model.set_eval()
    model.sched.set_timesteps(1, device=device)
    model.sched.timesteps = torch.tensor([model.timesteps.item()], device=device)

    reader = imageio.get_reader(str(input_path))
    writer = None
    try:
        metadata = reader.get_meta_data()
        fps = metadata.get("fps", 30)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )

        frame_count = metadata.get("nframes")
        total = frame_count if isinstance(frame_count, int) and frame_count > 0 else None
        for frame in tqdm(reader, total=total, desc="DiFix refinement"):
            input_image = Image.fromarray(frame).convert("RGB")
            width, height = input_image.size
            process_width = ((width + 7) // 8) * 8
            process_height = ((height + 7) // 8) * 8

            with torch.inference_mode():
                output_image = model.sample(
                    input_image,
                    width=process_width,
                    height=process_height,
                    prompt="remove degradation",
                )

            if output_image.size != (width, height):
                output_image = output_image.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            original_frame = np.asarray(input_image)
            refined_frame = np.asarray(output_image)
            comparison_frame = np.concatenate(
                (original_frame, refined_frame), axis=1
            )
            writer.append_data(comparison_frame)
    finally:
        reader.close()
        if writer is not None:
            writer.close()


def main():
    args = parse_args()
    validate_args(args)
    refine_video(args.input, args.output, args.checkpoint, args.timestep)
    print(f"Refined video written to: {args.output}")


if __name__ == "__main__":
    main()
