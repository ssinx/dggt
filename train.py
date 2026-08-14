import os
import argparse
import math
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import lpips

from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.cubemap import render_cubemap
from gsplat.rendering import rasterization
from datasets.dataset import WaymoOpenDataset, WaymoParquetSkyDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) /  ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default='pretrained/model_latest_waymo.pt')
    parser.add_argument('--log_dir', type=str, default='logs/xxx')
    parser.add_argument(
        '--sequence_length', type=int, default=4,
        help='Frames per sample for processed-data mode',
    )
    parser.add_argument(
        '--raw_sequence_lengths', type=int, nargs='+', default=[4, 8, 12],
        help='Candidate interval lengths sampled per raw Waymo scene (default: 4 8 12)',
    )
    parser.add_argument(
        '--raw_fixed_start', type=int,
        help='Fix the raw-scene interval start index; intended for deterministic overfit tests',
    )
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument(
        '--save_image', type=int, default=50,
        help='Save non-overwriting training visualizations every N optimizer steps (default: 50)',
    )
    parser.add_argument(
        '--save_ckpt', type=int, default=50,
        help='Update model_latest.pt and, when improved, model_best.pt every N optimizer steps (default: 50)',
    )
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--log_every', type=int, default=10, help='Print and record metrics every N optimizer steps')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--use_splatformer', type=bool, default=False)
    parser.add_argument('--downsample_3dgs', type=bool, default=False)
    parser.add_argument('--sky_model', choices=['cubemap', 'gaussian'], default='cubemap')
    parser.add_argument('--sky_rgb_weight', type=float, default=1.0)
    parser.add_argument('--scene_names', nargs='+', help='Processed Waymo scene directory names')
    parser.add_argument('--scene_names_file', help='Text file containing one processed scene name per line')
    parser.add_argument('--raw_waymo_dir', help='Waymo v2 split directory containing component Parquet folders')
    parser.add_argument(
        '--sky_mask_cache_dir', default='/scratch/junyizh3/waymo_sky_masks',
        help='Directory for persistent, uncompressed uint8 SegFormer sky masks',
    )
    parser.add_argument('--segformer_python', default='/data/user_data/junyizh3/miniconda3/envs/segformer/bin/python')
    parser.add_argument('--segformer_path', default='/data/user_data/junyizh3/projects/SegFormer')
    parser.add_argument('--segformer_config', default='/data/user_data/junyizh3/projects/SegFormer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py')
    parser.add_argument('--segformer_checkpoint', default='/data/user_data/junyizh3/projects/SegFormer/checkpoints/segformer.b5.1024x1024.city.160k.pth')
    return parser.parse_args()


def resolve_scene_names(args, default_all=False):
    if args.scene_names and args.scene_names_file:
        raise ValueError('Use only one of --scene_names and --scene_names_file')
    if args.scene_names_file:
        with open(args.scene_names_file, encoding='utf-8') as scene_file:
            scene_names = [line.strip() for line in scene_file if line.strip() and not line.lstrip().startswith('#')]
        if not scene_names:
            raise ValueError(f'No scene names found in {args.scene_names_file}')
        return scene_names
    if args.scene_names:
        return args.scene_names
    # Raw modular Waymo discovers every Parquet segment when no explicit
    # subset is requested. Keep the legacy processed-data default unchanged.
    return None if default_all else [str(scene_id).zfill(3) for scene_id in range(300, 600)]


def load_checkpoint_for_sky_training(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    model_state = model.state_dict()
    checkpoint_sky = {key: value for key, value in checkpoint.items() if key.startswith("sky_head.")}
    model_sky_keys = {key for key in model_state if key.startswith("sky_head.")}
    sky_is_compatible = set(checkpoint_sky) == model_sky_keys and all(
        model_state[key].shape == value.shape for key, value in checkpoint_sky.items()
    )
    if checkpoint_sky and not sky_is_compatible:
        # Architecture migrations intentionally restart only the sky decoder;
        # every frozen checkpoint tensor must still match below.
        checkpoint = {key: value for key, value in checkpoint.items() if not key.startswith("sky_head.")}

    incompatible = model.load_state_dict(checkpoint, strict=False)
    missing_sky = sorted(key for key in incompatible.missing_keys if key.startswith("sky_head."))
    missing_non_sky = sorted(key for key in incompatible.missing_keys if not key.startswith("sky_head."))
    unexpected_non_sky = sorted(
        key for key in incompatible.unexpected_keys if not key.startswith("sky_head.")
    )
    if missing_non_sky or unexpected_non_sky:
        raise RuntimeError(
            "Checkpoint is incompatible with frozen DGGT modules: "
            f"missing={missing_non_sky[:10]}, unexpected={unexpected_non_sky[:10]}"
        )
    return missing_sky


def configure_sky_only_training(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sky_head.parameters():
        parameter.requires_grad_(True)

    model.eval()
    model.sky_head.train()

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(not name.startswith("sky_head.") for name in trainable_names):
        raise RuntimeError(f"Expected only sky_head parameters to be trainable, got {trainable_names[:10]}")
    return trainable_names


def verify_sky_only_gradients(model):
    sky_has_gradient = False
    for name, parameter in model.named_parameters():
        if name.startswith("sky_head."):
            sky_has_gradient = sky_has_gradient or parameter.grad is not None
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter unexpectedly received a gradient: {name}")
    if not sky_has_gradient:
        raise RuntimeError("No sky_head parameter received a gradient")


def save_training_checkpoint(model, log_dir, global_iteration, checkpoint_name):
    """Atomically replace a named rolling checkpoint to limit disk usage."""
    checkpoint_dir = os.path.join(log_dir, "ckpt")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    temporary_path = f"{checkpoint_path}.tmp"
    torch.save(model.module.state_dict(), temporary_path)
    os.replace(temporary_path, checkpoint_path)
    print(f"[Checkpoint] Saved iteration {global_iteration} to {checkpoint_path}")


def save_training_images(log_dir, global_iteration, rendered_image, target_image,
                         bg_render, alphas, sky_cubemap):
    """Save one visualization pair without replacing artifacts from an earlier run."""
    images_dir = os.path.join(log_dir, "images")
    artifact_stem = f"iteration_{global_iteration:09d}"
    suffix = 0
    while True:
        unique_stem = artifact_stem if suffix == 0 else f"{artifact_stem}_{suffix}"
        frame_path = os.path.join(images_dir, f"{unique_stem}_frame.png")
        cubemap_path = os.path.join(images_dir, f"{unique_stem}_cubemap.png")
        if not os.path.exists(frame_path) and not os.path.exists(cubemap_path):
            break
        suffix += 1

    random_frame_idx = random.randint(0, rendered_image.shape[0] - 1)
    rendered = rendered_image[random_frame_idx].detach().cpu().clamp(0, 1)
    target = target_image[random_frame_idx].detach().cpu().clamp(0, 1)
    sky = bg_render[random_frame_idx].detach().cpu().permute(2, 0, 1).clamp(0, 1)
    alpha_rgb = alphas[random_frame_idx, ..., 0].unsqueeze(0).repeat(3, 1, 1).cpu()
    combined = torch.cat([target, rendered, sky, alpha_rgb], dim=-1)
    T.ToPILImage()(combined).save(frame_path)

    cubemap_strip = torch.cat(
        [face for face in sky_cubemap[0].detach().cpu().clamp(0, 1)], dim=-1
    )
    T.ToPILImage()(cubemap_strip).save(cubemap_path)
    print(f"[Images] Saved iteration {global_iteration} to {frame_path} and {cubemap_path}")

def main(args):
    dist.init_process_group(backend='nccl')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dtype = torch.float32
    
    scene_names = resolve_scene_names(args, default_all=args.raw_waymo_dir is not None)
    if args.raw_waymo_dir:
        dataset = WaymoParquetSkyDataset(
            args.raw_waymo_dir,
            scene_names=scene_names,
            sequence_lengths=args.raw_sequence_lengths,
            fixed_start=args.raw_fixed_start,
            segformer_python=args.segformer_python,
            segformer_path=args.segformer_path,
            segformer_config=args.segformer_config,
            segformer_checkpoint=args.segformer_checkpoint,
            segformer_device=f"cuda:{args.local_rank}",
            sky_mask_cache_dir=args.sky_mask_cache_dir,
        )
        # Windows are grouped by segment so each worker can reuse its small
        # in-memory Parquet cache. Every window is still visited every epoch.
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            mode=1,
            views=1,
        )
        sampler = DistributedSampler(dataset, shuffle=True)
    if args.local_rank == 0:
        if args.raw_waymo_dir:
            print(
                f"Training dataset: {len(dataset)} scenes/steps per epoch, "
                f"random interval length from {args.raw_sequence_lengths}"
            )
        else:
            print(f"Training dataset: {len(dataset)} samples, {args.sequence_length} frames per sample")
    # Online SegFormer owns one persistent CUDA subprocess and therefore must
    # be called from the main process, not forked DataLoader workers.
    dataloader_workers = 0 if args.raw_waymo_dir else 4
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=dataloader_workers)

    tensorboard_writer = None
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)
        metrics_path = os.path.join(args.log_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                metrics_file.write("iteration,epoch,loss,render_l1,sky_rgb,lpips,alpha_diagnostic,lr\n")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoard logging requires the `tensorboard` package. "
                "Install it with `pip install tensorboard`."
            ) from error
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))
        # Make the dashboard immediately observable while the checkpoint,
        # SegFormer, and first training batch are still loading.
        tensorboard_writer.add_scalar("run/started", 1.0, 0)
        tensorboard_writer.flush()

    if args.sky_model != "cubemap":
        raise ValueError("Sky-only training requires --sky_model cubemap")

    model = VGGT().to(device)
    missing_sky = load_checkpoint_for_sky_training(model, args.ckpt_path)
    trainable_names = configure_sky_only_training(model)
    if args.local_rank == 0:
        if missing_sky:
            print(f"Initialized {len(missing_sky)} sky_head tensors from scratch")
        else:
            print("Resuming sky_head weights from checkpoint")
        print(f"Training {len(trainable_names)} sky_head parameter tensors only")

    # requires_grad must be finalized before DDP constructs its gradient reducer.
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device).eval()
    for parameter in lpips_loss_fn.parameters():
        parameter.requires_grad_(False)

    optimizer = AdamW(model.module.sky_head.parameters(), lr=1e-4, weight_decay=1e-4)

    total_iterations = max(1, args.max_epoch * len(dataloader))
    warmup_iterations = min(1000, max(1, total_iterations // 20))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda iteration: min((iteration + 1) / warmup_iterations, 1.0) * 0.5 * (
            1 + math.cos(math.pi * min(iteration, total_iterations) / total_iterations)
        ),
    )

    global_iteration = 0
    best_checkpoint_loss = float("inf")
    progress = tqdm(
        total=total_iterations,
        desc="Training sky cubemap",
        unit="step",
        disable=args.local_rank != 0,
        dynamic_ncols=True,
    )
    for step in range(args.max_epoch):
        sampler.set_epoch(step)        
        for batch in dataloader:
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            sky_mask_valid = batch.get('sky_mask_valid')
            if sky_mask_valid is None:
                sky_mask_valid = torch.ones(sky_mask.shape[:2], device=device)
            else:
                sky_mask_valid = sky_mask_valid.to(device)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)
            gt_intrinsics = batch['intrinsics'].to(device)
            camera_to_worlds = batch['camera_to_worlds'].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(
                    images,
                    intrinsics=gt_intrinsics if args.sky_model == "cubemap" else None,
                    camera_to_worlds=camera_to_worlds if args.sky_model == "cubemap" else None,
                )
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]

                use_depth = True
                if use_depth:
                    depth_map = predictions["depth"][0]
                    point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None,...]
                    point_map = torch.from_numpy(point_map).to(device).float()
                else:      
                    point_map = predictions["world_points"]
                gs_map = predictions["gs_map"]
                gs_conf = predictions["gs_conf"]
                dy_map = predictions["dynamic_conf"].squeeze(-1) #B,H,W,1

                static_mask = torch.ones_like(bg_mask)
                static_points = point_map[static_mask].reshape(-1, 3)
                gs_dynamic_list = dy_map[static_mask].sigmoid() 
                static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                static_opacity = static_opacity * (1 - gs_dynamic_list)
                static_gs_conf = gs_conf[static_mask]
                frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                gs_timestamps = timestamps[frame_idx]     

                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid() 
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i
                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)
                    
                chunked_renders, chunked_alphas = [], []
                S = extrinsic.shape[0]
                for idx in range(S):
                    t0 = timestamps[idx]
                    static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                    static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                    if dynamic_points:
                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            static_gs_list,
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]#注释
                        )
                    renders_chunk, alphas_chunk, _ = rasterization(
                        means=world_points, 
                        quats=rotation, 
                        scales=scales, 
                        opacities=opacity, 
                        colors=rgbs,
                        viewmats=extrinsic[idx][None], 
                        Ks=intrinsic[idx][None],
                        width=W, 
                        height=H, 
                    )
                    chunked_renders.append(renders_chunk)
                    chunked_alphas.append(alphas_chunk)


                renders = torch.cat(chunked_renders, dim=0)
                alphas = torch.cat(chunked_alphas, dim=0)
                if args.sky_model == "cubemap":
                    bg_render = render_cubemap(
                        predictions["sky_cubemap"], gt_intrinsics, camera_to_worlds, H, W
                    )[0]
                else:
                    bg_render = model.module.sky_model(images, extrinsic, intrinsic)
                renders = alphas * renders + (1 - alphas) * bg_render

                rendered_image = renders.permute(0, 3, 1, 2)
                target_image = images[0]


                ####################### Loss ###########################


                render_l1_loss = F.l1_loss(rendered_image, target_image)

                sky_pixels = sky_mask[0, ..., :1].clamp(0, 1)
                sky_pixels = sky_pixels * sky_mask_valid[0, :, None, None, None]
                sky_rgb_error = (bg_render - target_image.permute(0, 2, 3, 1)).abs()
                sky_rgb_loss = (sky_rgb_error * sky_pixels).sum() / (3.0 * sky_pixels.sum().clamp_min(1.0))

                # Gaussian alpha is frozen; this metric diagnoses foreground leakage only.
                sky_mask_loss = F.l1_loss(alphas, 1 - sky_mask[0, ..., 0][..., None])
                lpips_val = lpips_loss_fn(rendered_image, target_image).mean()
                lpips_weight = 0.05 * min(global_iteration / 1000, 1.0)
                loss = render_l1_loss + args.sky_rgb_weight * sky_rgb_loss + lpips_weight * lpips_val

            loss.backward()
            if global_iteration == 0:
                verify_sky_only_gradients(model.module)
            torch.nn.utils.clip_grad_norm_(model.module.sky_head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_iteration += 1
            if args.local_rank == 0:
                progress.update(1)
                current_loss = loss.item()
                current_sky_loss = sky_rgb_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    sky=f"{current_sky_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    refresh=True,
                )
                tensorboard_writer.add_scalar("loss/total", current_loss, global_iteration)
                tensorboard_writer.add_scalar("loss/sky_rgb", current_sky_loss, global_iteration)
                tensorboard_writer.add_scalar("loss/render_l1", render_l1_loss.item(), global_iteration)
                tensorboard_writer.add_scalar("loss/lpips", lpips_val.item(), global_iteration)
                tensorboard_writer.add_scalar(
                    "diagnostic/alpha_sky_mask", sky_mask_loss.item(), global_iteration
                )
                tensorboard_writer.add_scalar("optimizer/learning_rate", current_lr, global_iteration)
                if global_iteration == 1 or (args.log_every > 0 and global_iteration % args.log_every == 0):
                    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                        metrics_file.write(
                            f"{global_iteration},{step},{current_loss:.8f},"
                            f"{render_l1_loss.item():.8f},{sky_rgb_loss.item():.8f},"
                            f"{lpips_val.item():.8f},{sky_mask_loss.item():.8f},{current_lr:.10g}\n"
                        )
                    tensorboard_writer.flush()
            if args.local_rank == 0:
                if args.save_image > 0 and global_iteration % args.save_image == 0:
                    save_training_images(
                        args.log_dir, global_iteration, rendered_image, target_image,
                        bg_render, alphas, predictions["sky_cubemap"],
                    )
                if args.save_ckpt > 0 and global_iteration % args.save_ckpt == 0:
                    save_training_checkpoint(model, args.log_dir, global_iteration, "model_latest.pt")
                    if current_loss < best_checkpoint_loss:
                        best_checkpoint_loss = current_loss
                        save_training_checkpoint(model, args.log_dir, global_iteration, "model_best.pt")
                        print(f"[Checkpoint] New best loss: {best_checkpoint_loss:.8f}")
        
    if args.local_rank == 0:
        progress.close()
        save_training_checkpoint(model, args.log_dir, global_iteration, "model_latest.pt")
        tensorboard_writer.flush()
        tensorboard_writer.close()
    if hasattr(dataset, "close"):
        dataset.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
