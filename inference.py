import argparse
import os
import random
import time
import numpy as np
from scipy.spatial import cKDTree
import scipy.spatial.transform
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import open3d as o3d
from third_party.difix.infer import process_images_with_difix
from third_party.TAPIP3D.utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_masked_gs, get_split_gs
from dggt.utils.visual_track import visualize_tracks_on_images
from gsplat.rendering import rasterization
from datasets.dataset import ImageDirectoryDataset, WaymoOpenDataset, load_and_preprocess_images
from utils.interplation import interp_all
from utils.video_maker import make_comparison_video_quad


OPENCV_TO_WAYMO = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
WAYMO_CAMERA_IDS = (0, 1, 2, 3, 4)
WAYMO_COMPARISON_CAMERA_ORDER = (3, 1, 0, 2, 4)
DEFAULT_WAYMO_CAMERA_INTRINSICS = {
    0: np.array([2056.2823692090583, 2056.2823692090583, 939.5779800048163, 641.1030804525235, 0.030161695455207253, -0.27366348163840665, 0.001099757907898257, -0.0019278126890982551, 0.0], dtype=np.float32),
    1: np.array([2063.9144000244905, 2063.9144000244905, 978.7598986481419, 639.3123770365744, 0.028426816000167728, -0.3168906331588955, 0.0009645062447413426, 0.0008992608844722939, 0.0], dtype=np.float32),
    2: np.array([2065.5315047381105, 2065.5315047381105, 932.3535100809372, 645.739378382839, 0.04046903603810112, -0.3500865311278198, -0.00016608180821456998, -0.0006035090111663273, 0.0], dtype=np.float32),
    3: np.array([2062.2195522170755, 2062.2195522170755, 974.2667860249346, 238.09466189172986, 0.041057267678638515, -0.33001393529516099, 0.0002816406257543573, -0.0002818074981018001, 0.0], dtype=np.float32),
    4: np.array([2057.3747280824628, 2057.3747280824628, 958.7077423615501, 262.8829315187602, 0.036860040361417955, -0.3123290638079368, 0.001720265718725565, -0.0011633157299501204, 0.0], dtype=np.float32),
}
DEFAULT_WAYMO_CAMERA_EXTRINSICS = {
    0: np.array([[0.9998362731902033, -0.006126490657911932, 0.01702624225551874, 1.538666713905355], [0.006212629204228849, 0.9999681466335748, -0.005010883812664572, -0.02493885762968132], [-0.01699500077951916, 0.005115841126518574, 0.999842486653809, 2.1153399048271977], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    1: np.array([[0.7178061976277986, -0.6961137197995401, 0.013414609721087198, 1.4939734510380249], [0.6960062510324471, 0.7179313259630274, 0.012243762686665767, 0.09112245742382508], [-0.01815381973249437, 0.0005480034822642641, 0.9998350556573337, 2.1151088154019067], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    2: np.array([[0.719643071473014, 0.6942976091985146, 0.008042359238015363, 1.4892976372888604], [-0.6942355150736363, 0.7196870899331542, -0.009356398558782966, -0.09461793927925179], [-0.012284107266275738, 0.0011499759887574406, 0.9999238862352954, 2.115639662241132], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    3: np.array([[0.008863678723313523, -0.9999469878494199, 0.0052399132017394495, 1.432758697942947], [0.9998136741398955, 0.008952114488320716, 0.017101948692201843, 0.11116961823274964], [-0.017147950384013396, 0.005087350691654994, 0.9998400205335688, 2.115181938652384], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    4: np.array([[0.005429609317539405, 0.9999852584639241, 0.00004687747692506315, 1.427151747978933], [-0.9998164036504379, 0.0054295539235595745, -0.018376042438339648, -0.11155890913778603], [-0.018376026071035875, 0.00005290586085128549, 0.9998311451774277, 2.1158184977254706], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
}


def load_waymo_camera_calibrations(calibration_dir):
    """Load raw Waymo camera-to-vehicle transforms and intrinsics."""
    camera_to_ego = {}
    camera_intrinsics = {}
    fallback_camera_ids = []
    for camera_id in WAYMO_CAMERA_IDS:
        extrinsic_path = os.path.join(calibration_dir, "extrinsics", f"{camera_id}.txt")
        intrinsic_path = os.path.join(calibration_dir, "intrinsics", f"{camera_id}.txt")
        if os.path.isfile(extrinsic_path) and os.path.isfile(intrinsic_path):
            extrinsic = np.loadtxt(extrinsic_path, dtype=np.float32)
            intrinsic = np.loadtxt(intrinsic_path, dtype=np.float32).reshape(-1)
        else:
            extrinsic = DEFAULT_WAYMO_CAMERA_EXTRINSICS[camera_id].copy()
            intrinsic = DEFAULT_WAYMO_CAMERA_INTRINSICS[camera_id].copy()
            fallback_camera_ids.append(camera_id)
        if extrinsic.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 extrinsic matrix in {extrinsic_path}, got {extrinsic.shape}.")
        if intrinsic.shape[0] < 4:
            raise ValueError(f"Expected at least [fx, fy, cx, cy] in {intrinsic_path}.")

        camera_to_ego[camera_id] = extrinsic @ OPENCV_TO_WAYMO
        camera_intrinsics[camera_id] = intrinsic[:4]

    if fallback_camera_ids:
        print(
            f"Using built-in scene-150 Waymo calibration fallback for cameras {fallback_camera_ids}; "
            "rendered geometry is approximate when the current vehicle rig differs."
        )
    return camera_to_ego, camera_intrinsics


def canonicalize_waymo_camera_calibrations(camera_to_ego, camera_intrinsics):
    headings = {}
    for stored_camera_id, camera_pose in camera_to_ego.items():
        camera_forward_in_vehicle = camera_pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        headings[stored_camera_id] = float(np.arctan2(camera_forward_in_vehicle[1], camera_forward_in_vehicle[0]))

    front_camera = min(headings, key=lambda camera_id: abs(headings[camera_id]))
    remaining_camera_ids = [camera_id for camera_id in WAYMO_CAMERA_IDS if camera_id != front_camera]
    left_cameras = sorted(
        (camera_id for camera_id in remaining_camera_ids if headings[camera_id] > 0),
        key=lambda camera_id: abs(headings[camera_id]),
    )
    right_cameras = sorted(
        (camera_id for camera_id in remaining_camera_ids if headings[camera_id] < 0),
        key=lambda camera_id: abs(headings[camera_id]),
    )
    if len(left_cameras) != 2 or len(right_cameras) != 2:
        heading_degrees = {camera_id: round(np.degrees(heading), 2) for camera_id, heading in headings.items()}
        raise ValueError(
            "Cannot recover canonical Waymo camera IDs from calibration headings: "
            f"{heading_degrees}."
        )

    canonical_to_stored = {
        0: front_camera,
        1: left_cameras[0],
        2: right_cameras[0],
        3: left_cameras[1],
        4: right_cameras[1],
    }
    canonical_camera_to_ego = {
        canonical_camera_id: camera_to_ego[stored_camera_id]
        for canonical_camera_id, stored_camera_id in canonical_to_stored.items()
    }
    canonical_camera_intrinsics = {
        canonical_camera_id: camera_intrinsics[stored_camera_id]
        for canonical_camera_id, stored_camera_id in canonical_to_stored.items()
    }
    return canonical_camera_to_ego, canonical_camera_intrinsics, canonical_to_stored


def format_camera_rig_directions(camera_to_ego, source_camera):
    source_camera_to_ego = camera_to_ego[source_camera]
    camera_directions = []
    for camera_id in WAYMO_CAMERA_IDS:
        target_camera_to_ego = camera_to_ego[camera_id]
        source_to_target = np.linalg.inv(target_camera_to_ego) @ source_camera_to_ego
        target_forward_in_source = source_to_target[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw_degrees = np.degrees(np.arctan2(target_forward_in_source[0], target_forward_in_source[2]))
        target_center_in_source = np.linalg.inv(source_to_target)[:3, 3]
        camera_directions.append(
            f"camera_{camera_id}: yaw={yaw_degrees:+.1f}deg, "
            f"center=({target_center_in_source[0]:+.2f}, {target_center_in_source[1]:+.2f}, {target_center_in_source[2]:+.2f})m"
        )
    return "; ".join(camera_directions)


def compute_source_point_frustum_coverage(point_map, render_extrinsic, render_intrinsic, image_sizes, stride=16):
    sampled_point_map = point_map[0, :, ::stride, ::stride].reshape(point_map.shape[1], -1, 3)
    coverage = []
    for camera_offset, camera_id in enumerate(WAYMO_CAMERA_IDS):
        height, width = image_sizes[camera_id]
        visible_points = 0
        total_points = 0
        for frame_idx, world_points in enumerate(sampled_point_map):
            render_idx = frame_idx * len(WAYMO_CAMERA_IDS) + camera_offset
            viewmat = render_extrinsic[render_idx]
            intrinsic = render_intrinsic[render_idx]
            camera_points = world_points @ viewmat[:3, :3].transpose(0, 1) + viewmat[:3, 3]
            depth = camera_points[:, 2]
            valid_depth = depth > 1e-5
            projected_x = intrinsic[0, 0] * camera_points[:, 0] / depth.clamp_min(1e-5) + intrinsic[0, 2]
            projected_y = intrinsic[1, 1] * camera_points[:, 1] / depth.clamp_min(1e-5) + intrinsic[1, 2]
            visible = valid_depth & (projected_x >= 0) & (projected_x < width) & (projected_y >= 0) & (projected_y < height)
            visible_points += visible.sum().item()
            total_points += visible.numel()
        coverage.append((camera_id, visible_points / max(total_points, 1)))
    return coverage


def find_waymo_image_path(scene_dir, frame_id, camera_id):
    image_stem = os.path.join(scene_dir, "images", f"{frame_id:03d}_{camera_id}")
    for extension in (".jpg", ".png"):
        image_path = image_stem + extension
        if os.path.isfile(image_path):
            return image_path
    raise FileNotFoundError(f"Missing image for Waymo frame {frame_id}, camera {camera_id} under {scene_dir}/images.")


def load_waymo_comparison_ground_truth(scene_dir, frame_id):
    return {
        camera_id: load_and_preprocess_images([
            find_waymo_image_path(scene_dir, frame_id, camera_id)
        ])[0]
        for camera_id in WAYMO_COMPARISON_CAMERA_ORDER
    }


def make_waymo_all_camera_comparison_frame(rendered_images, ground_truth_images, dynamic_mask):
    """Compose GT, rendered views, and source-camera dynamic mask into one video frame."""
    rendered_images = {
        camera_id: image.detach().cpu().clamp(0, 1)
        for camera_id, image in rendered_images.items()
    }
    ground_truth_images = {
        camera_id: image.detach().cpu().clamp(0, 1)
        for camera_id, image in ground_truth_images.items()
    }
    dynamic_mask = dynamic_mask.detach().cpu().sigmoid().clamp(0, 1)

    for camera_id in WAYMO_COMPARISON_CAMERA_ORDER:
        if rendered_images[camera_id].shape != ground_truth_images[camera_id].shape:
            raise ValueError(
                f"Rendered and ground-truth sizes differ for camera {camera_id}: "
                f"{tuple(rendered_images[camera_id].shape)} and {tuple(ground_truth_images[camera_id].shape)}."
            )
    if dynamic_mask.shape != rendered_images[0].shape[-2:]:
        raise ValueError(
            "Dynamic mask size does not match source camera 0 render: "
            f"{tuple(dynamic_mask.shape)} and {tuple(rendered_images[0].shape[-2:])}."
        )

    canvas_height = max(
        image.shape[-2]
        for images in (rendered_images, ground_truth_images)
        for image in images.values()
    )
    canvas_width = max(
        image.shape[-1]
        for images in (rendered_images, ground_truth_images)
        for image in images.values()
    )

    def pad_to_canvas(image):
        height_padding = canvas_height - image.shape[-2]
        width_padding = canvas_width - image.shape[-1]
        if height_padding < 0 or width_padding < 0:
            raise ValueError("Comparison canvas cannot be smaller than an input image.")
        return F.pad(
            image,
            (
                width_padding // 2,
                width_padding - width_padding // 2,
                height_padding,
                0,
            ),
        )

    def tile_row(images):
        return torch.cat([pad_to_canvas(images[camera_id]) for camera_id in WAYMO_COMPARISON_CAMERA_ORDER], dim=2)

    dynamic_mask = dynamic_mask.unsqueeze(0).expand(3, -1, -1)
    dynamic_tiles = {
        camera_id: torch.zeros_like(ground_truth_images[camera_id])
        for camera_id in WAYMO_COMPARISON_CAMERA_ORDER
    }
    dynamic_tiles[0] = dynamic_mask
    return torch.cat(
        [tile_row(ground_truth_images), tile_row(rendered_images), tile_row(dynamic_tiles)],
        dim=1,
    )


def build_preprocessed_waymo_intrinsic(image_path, raw_intrinsic, output_height, output_width):
    with Image.open(image_path) as image:
        original_width, original_height = image.size

    resize_width = 518
    if output_width != resize_width:
        raise ValueError(f"Expected model input width {resize_width}, got {output_width}.")
    resize_scale = resize_width / original_width
    resized_height = round(original_height * resize_scale / 14) * 14
    intrinsic = np.array(
        [
            [raw_intrinsic[0] * resize_scale, 0.0, raw_intrinsic[2] * resize_scale],
            [0.0, raw_intrinsic[1] * resize_scale, raw_intrinsic[3] * resize_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if resized_height > output_height:
        intrinsic[1, 2] -= (resized_height - output_height) // 2
    elif resized_height < output_height:
        intrinsic[1, 2] += (output_height - resized_height) // 2
    return intrinsic


def get_preprocessed_waymo_image_size(image_path):
    with Image.open(image_path) as image:
        original_width, original_height = image.size
    output_width = 518
    resized_height = round(original_height * (output_width / original_width) / 14) * 14
    return min(resized_height, output_width), output_width


def build_waymo_render_intrinsics(
    scene_dir,
    frame_ids,
    camera_intrinsics,
    source_camera,
    output_height,
    output_width,
    device,
    dtype,
):
    if len(frame_ids) == 0:
        raise ValueError("At least one Waymo frame is required to build calibrated intrinsics.")
    source_image_path = find_waymo_image_path(scene_dir, frame_ids[0], source_camera)
    render_intrinsics = {}
    render_image_sizes = {}
    missing_image_camera_ids = []
    for camera_id in WAYMO_CAMERA_IDS:
        try:
            image_path = find_waymo_image_path(scene_dir, frame_ids[0], camera_id)
        except FileNotFoundError:
            image_path = source_image_path
            missing_image_camera_ids.append(camera_id)
        camera_height, camera_width = get_preprocessed_waymo_image_size(image_path)
        if camera_id == source_camera and (camera_height, camera_width) != (output_height, output_width):
            raise ValueError(
                f"Source camera preprocessing size {(camera_height, camera_width)} does not match model input "
                f"size {(output_height, output_width)}."
            )
        intrinsic = build_preprocessed_waymo_intrinsic(
            image_path,
            camera_intrinsics[camera_id],
            camera_height,
            camera_width,
        )
        render_intrinsics[camera_id] = torch.as_tensor(intrinsic, device=device, dtype=dtype).unsqueeze(0).expand(
            len(frame_ids), -1, -1
        ).clone()
        render_image_sizes[camera_id] = (camera_height, camera_width)
    if missing_image_camera_ids:
        print(
            f"Using source camera {source_camera} image dimensions for missing target camera images "
            f"{missing_image_camera_ids}."
        )
    return render_intrinsics, render_image_sizes


def format_camera_fovs(intrinsics, image_sizes):
    fovs = []
    for camera_id in WAYMO_CAMERA_IDS:
        intrinsic = intrinsics[camera_id][0]
        height, width = image_sizes[camera_id]
        fov_x = torch.rad2deg(2 * torch.atan(torch.tensor(width, device=intrinsic.device, dtype=intrinsic.dtype) / (2 * intrinsic[0, 0])))
        fov_y = torch.rad2deg(2 * torch.atan(torch.tensor(height, device=intrinsic.device, dtype=intrinsic.dtype) / (2 * intrinsic[1, 1])))
        fovs.append(f"camera_{camera_id}=({fov_x.item():.1f}, {fov_y.item():.1f})deg")
    return ", ".join(fovs)


def build_all_camera_render_poses(
    source_extrinsics,
    camera_to_ego,
    render_intrinsics,
    source_camera,
    rig_translation_scale,
):
    """Build frame-major target poses for all Waymo cameras from one source camera."""
    if source_camera not in WAYMO_CAMERA_IDS:
        raise ValueError(f"source_camera must be one of {WAYMO_CAMERA_IDS}, got {source_camera}.")
    if rig_translation_scale <= 0:
        raise ValueError(f"rig_translation_scale must be positive, got {rig_translation_scale}.")

    device = source_extrinsics.device
    dtype = source_extrinsics.dtype
    source_camera_to_ego = torch.as_tensor(camera_to_ego[source_camera], device=device, dtype=dtype)

    relative_viewmats = []
    for camera_id in WAYMO_CAMERA_IDS:
        target_camera_to_ego = torch.as_tensor(camera_to_ego[camera_id], device=device, dtype=dtype)
        relative_viewmat = torch.linalg.inv(target_camera_to_ego) @ source_camera_to_ego
        relative_viewmat[:3, 3] *= rig_translation_scale
        relative_viewmats.append(relative_viewmat)

    relative_viewmats = torch.stack(relative_viewmats, dim=0)
    target_extrinsics = torch.matmul(relative_viewmats.unsqueeze(0), source_extrinsics.unsqueeze(1))
    target_intrinsics = torch.stack([render_intrinsics[camera_id] for camera_id in WAYMO_CAMERA_IDS], dim=1)
    return (
        target_extrinsics.flatten(0, 1),
        target_intrinsics.flatten(0, 1),
    )


def get_frame_ids_from_image_paths(image_paths):
    frame_ids = []
    for image_path in image_paths:
        while isinstance(image_path, (list, tuple)):
            if len(image_path) == 0:
                raise ValueError("Encountered an empty image path batch entry.")
            image_path = image_path[0]
        frame_name = os.path.basename(os.fspath(image_path))
        try:
            frame_ids.append(int(frame_name.split("_", maxsplit=1)[0]))
        except ValueError as error:
            raise ValueError(f"Cannot extract a Waymo frame ID from {frame_name}.") from error
    return frame_ids


def estimate_waymo_rig_translation_scale(scene_dir, image_paths, source_extrinsics):
    frame_ids = get_frame_ids_from_image_paths(image_paths)
    if len(frame_ids) != source_extrinsics.shape[0]:
        raise ValueError(
            f"Got {len(frame_ids)} input frame paths but {source_extrinsics.shape[0]} predicted poses."
        )

    ego_positions = []
    for frame_id in frame_ids:
        ego_pose_path = os.path.join(scene_dir, "ego_pose", f"{frame_id:03d}.txt")
        if not os.path.isfile(ego_pose_path):
            raise FileNotFoundError(
                f"Missing ego pose {ego_pose_path}; it is required to align the Waymo rig translation scale."
            )
        ego_pose = np.loadtxt(ego_pose_path, dtype=np.float32)
        if ego_pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 ego pose matrix in {ego_pose_path}, got {ego_pose.shape}.")
        ego_positions.append(ego_pose[:3, 3])

    source_rotations = source_extrinsics[:, :3, :3]
    source_translations = source_extrinsics[:, :3, 3]
    source_camera_positions = -torch.bmm(
        source_rotations.transpose(1, 2),
        source_translations.unsqueeze(-1),
    ).squeeze(-1).detach().float().cpu().numpy()
    ego_positions = np.stack(ego_positions, axis=0)

    model_displacements = np.linalg.norm(source_camera_positions[1:] - source_camera_positions[0], axis=1)
    ego_displacements = np.linalg.norm(ego_positions[1:] - ego_positions[0], axis=1)
    valid = (model_displacements > 1e-5) & (ego_displacements > 1e-5)
    if not np.any(valid):
        raise ValueError("Cannot estimate the Waymo rig scale because the selected sequence has no measurable motion.")

    rig_translation_scale = float(np.median(model_displacements[valid] / ego_displacements[valid]))
    if not np.isfinite(rig_translation_scale) or rig_translation_scale <= 0:
        raise ValueError(f"Invalid estimated Waymo rig translation scale: {rig_translation_scale}.")
    return rig_translation_scale


def alpha_t(t, t0, alpha, gamma0 = 1, gamma1 = 0.1):
    sigma = torch.log(torch.tensor(gamma1)).to(gamma0.device) / ((gamma0)**2 + 1e-6)
    conf = torch.exp(sigma*(t0-t)**2)
    alpha_ = alpha * conf
    return alpha_.float()

def compute_metrics(img1, img2, loss_fn):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(img1.shape[0]):
        im1 = img1[i].cpu().permute(1, 2, 0).numpy()
        im2 = img2[i].cpu().permute(1, 2, 0).numpy()
        psnr = peak_signal_noise_ratio(im1, im2, data_range=1.0)
        ssim = structural_similarity(im1, im2, channel_axis=2, data_range=1.0)
        lpips_val = loss_fn(img1[i].unsqueeze(0) * 2 - 1, img2[i].unsqueeze(0) * 2 - 1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_val.item())
    return sum(psnr_list) / len(psnr_list), sum(ssim_list) / len(ssim_list), sum(lpips_list) / len(lpips_list)

def calculate_scale_factor(P1, P2):
    distances_P1 = torch.norm(P1[1:], dim=1)  
    distances_P2 = torch.norm(P2[1:], dim=1)  
    avg_distance_P1 = torch.mean(distances_P1)
    if avg_distance_P1 < 0.1: #almost not move
        return 1
    avg_distance_P2 = torch.mean(distances_P2)
    scale_factor = avg_distance_P2 / avg_distance_P1
    return scale_factor

def save_video(images, path, fps=8):
    images = images.detach().cpu()  # Ensure it's on CPU
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.byte().permute(0, 2, 3, 1).numpy()  # [S, H, W, 3]
    
    imageio.mimwrite(path, images, fps=fps, codec='libx264')

def parse_scene_names(scene_names_str):
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        try:
            return [str(i).zfill(3) for i in range(int(start), int(end) + 1)]
        except ValueError as error:
            raise ValueError(
                "Scene ranges must use numeric IDs, for example '(3,7)'. "
                "Pass named scenes as space-separated values instead."
            ) from error

    scene_names = []
    for scene_name in scene_names_str.split():
        try:
            scene_names.append(str(int(scene_name)).zfill(3))
        except ValueError:
            scene_names.append(scene_name)
    return scene_names

def main():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_dir', type=str, help='Path to a processed Waymo-style dataset')
    input_group.add_argument('--plain_image_dir', type=str, help='Directory of ordinary images for unlabelled inference')
    parser.add_argument('--scene_names', type=str, nargs='+', help='Scene names, supports numeric IDs, named directories, or numeric ranges like (3,7)')
    parser.add_argument('--input_views', type=int, default=1, help='Number of input views')
    parser.add_argument('--input_camera', type=int, default=0, choices=WAYMO_CAMERA_IDS, help='Waymo camera ID used when --input_views=1')
    parser.add_argument('--sequence_length', type=int, default=4, help='Number of input frames')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting frame index')
    parser.add_argument('--mode', type=int, choices=[1,2,3], required=True, help='Processing mode')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for results')
    parser.add_argument('-images', action='store_true', help='Whether to output each frame image')
    parser.add_argument('-depth', action='store_true', help='Whether to output each frame depth as .npy')
    parser.add_argument('-metrics', action='store_true', help='Whether to output evaluation metrics')
    parser.add_argument('-diffusion', action='store_true', help='Whether to process images with diffusion model')
    parser.add_argument('--intervals', type=int, default=2, help='Interval for mode=3')
    parser.add_argument('--render_all_cameras', action='store_true', help='Render all five Waymo cameras from the selected single-camera input')
    parser.add_argument('--calibration_dir', type=str, help='Optional scene directory containing intrinsics/ and extrinsics/')
    parser.add_argument('--camera_rig_scale', type=float, help='Optional model-units-per-meter scale for Waymo camera rig translations')
    args = parser.parse_args()

    plain_image_inference = args.plain_image_dir is not None
    if plain_image_inference:
        if args.mode != 2:
            parser.error('--plain_image_dir supports only --mode 2')
        if args.input_views != 1:
            parser.error('--plain_image_dir supports only --input_views 1')
        if args.metrics:
            parser.error('-metrics requires ground-truth images and is unavailable with --plain_image_dir')
    elif not args.scene_names:
        parser.error('--scene_names is required when using --image_dir')

    if args.render_all_cameras:
        if plain_image_inference:
            parser.error('--render_all_cameras requires a processed Waymo dataset passed with --image_dir')
        if args.mode != 2:
            parser.error('--render_all_cameras currently supports only --mode 2')
        if args.input_views != 1:
            parser.error('--render_all_cameras requires --input_views 1')
        if args.metrics:
            parser.error('-metrics cannot compare five rendered cameras to one source-camera ground truth')
        if args.camera_rig_scale is not None and args.camera_rig_scale <= 0:
            parser.error('--camera_rig_scale must be positive')

    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    loss_fn = lpips.LPIPS(net='alex').to(device) if args.metrics else None

    if plain_image_inference:
        dataset = ImageDirectoryDataset(
            args.plain_image_dir,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
        )
    else:
        scene_names_str = ' '.join(args.scene_names)
        scene_names = parse_scene_names(scene_names_str)
    if not plain_image_inference and args.mode == 3:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views,
            intervals=args.intervals,
            input_camera=args.input_camera,
        )
    elif not plain_image_inference:
        dataset = WaymoOpenDataset(
            args.image_dir,
            scene_names=scene_names,
            sequence_length=args.sequence_length,
            start_idx=args.start_idx,
            mode=args.mode,
            views=args.input_views,
            input_camera=args.input_camera,
        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    if args.mode == 3:
        track_ckpt = 'path_to_track_model'
        track_model = load_model(track_ckpt)
        track_model.to(device)
        track_model.seq_len = 2
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    inference_time_list = []
    scene_idx = 1

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            scene_name = scene_names[batch_idx] if not plain_image_inference else str(scene_idx).zfill(3)
            images = batch['images'].to(device)
            sky_mask = batch['masks'].to(device).permute(0, 1, 3, 4, 2)
            if not plain_image_inference:
                gt_dy_map = batch['dynamic_mask'].to(device)
                gt_depth = batch['gt_depth'].to(device)

            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch['timestamps'][0].to(device)
            
            if args.mode == 3:
                target_images = batch['targets'].to(device)
                target_sky_masks = batch['target_masks'].to(device)

            start_time = time.time()
            dynamic = False
            if 'dynamic_mask' in batch:
                dynamic = True
                dynamic_masks = batch['dynamic_mask'].to(device)[:, :, 0, :, :]

            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                H, W = images.shape[-2:]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], (H, W))
                extrinsic = extrinsics[0]
                bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device).view(1, 1, 4).expand(extrinsic.shape[0], 1, 4)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                intrinsic = intrinsics[0]
                model_source_intrinsic = intrinsic.clone()
                intervals=args.intervals
                views=args.input_views

                if args.render_all_cameras:
                    scene_dir = args.calibration_dir or os.path.join(args.image_dir, scene_name)
                    camera_to_ego, camera_intrinsics = load_waymo_camera_calibrations(scene_dir)
                    camera_to_ego, camera_intrinsics, canonical_to_stored = canonicalize_waymo_camera_calibrations(
                        camera_to_ego,
                        camera_intrinsics,
                    )
                    frame_ids = get_frame_ids_from_image_paths(batch['image_paths'])
                    render_intrinsics, render_image_sizes = build_waymo_render_intrinsics(
                        scene_dir,
                        frame_ids,
                        camera_intrinsics,
                        args.input_camera,
                        H,
                        W,
                        device,
                        intrinsic.dtype,
                    )
                    intrinsic = render_intrinsics[args.input_camera]

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

                if args.mode == 2:
                    static_mask = (bg_mask & (dy_map < 0.5))
                    static_points = point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
                    static_opacity = static_opacity * (1 - gs_dynamic_list)
                    static_gs_conf = gs_conf[static_mask]
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                    gs_timestamps = timestamps[frame_idx]

                if args.mode == 3:
                    depth_map = depth_map.unsqueeze(0)
                    if args.input_views == 1:
                        (extrinsic, intrinsic, point_map, gs_map, dy_map, 
                        gs_conf, bg_mask, images, pred_flows, flow_masks,depth_interp) = interp_all(extrinsic, intrinsic, point_map, gs_map, dy_map, 
                                                                                gs_conf, bg_mask, images, target_images, depth_map, track_model,intervals,views)
                    # if args.input_views == 3:
                    #     (extrinsic, intrinsic, point_map, gs_map, dy_map, 
                    #     gs_conf, bg_mask, images, pred_flows, flow_masks,depth_interp) =  interp_all_3views(extrinsic, intrinsic, point_map, gs_map, dy_map, 
                    #                                                             gs_conf, bg_mask, images, target_images, depth_map, track_model,intervals,views)
                    I = intervals
                    bg_point_map = point_map[:, ::I, ...]
                    bg_bg_mask = bg_mask[:, ::I, ...]
                    bg_gs_map = gs_map[:, ::I, ...]
                    bg_dy_map = dy_map[:, ::I, ...]
                    bg_gs_conf = gs_conf[:, ::I, ...]

                    static_mask = (bg_bg_mask & (bg_dy_map < 0.5))
                    gs_conf = bg_gs_conf[static_mask]
                    static_points = bg_point_map[static_mask].reshape(-1, 3)
                    gs_dynamic_list = bg_dy_map[static_mask].sigmoid()
                    static_rgbs, static_opacity, static_scales, static_rotation = get_split_gs(bg_gs_map, static_mask)
                    frame_idx = torch.nonzero(static_mask, as_tuple=False)[:,1]
                    gs_timestamps = timestamps[frame_idx]
                    static_opacity = static_opacity * (1 - gs_dynamic_list)


                dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
                for i in range(dy_map.shape[1]):
                    point_map_i = point_map[:, i]
                    bg_mask_i = bg_mask[:, i]
                    dy_conf_i = dy_map[:, i].sigmoid()

                    dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
                    dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, i], bg_mask_i)
                    gs_dynamic_list_i = dy_map[:, i][bg_mask_i].sigmoid()
                    dynamic_opacity = dynamic_opacity * gs_dynamic_list_i

                    dynamic_points.append(dynamic_point)
                    dynamic_rgbs.append(dynamic_rgb)
                    dynamic_opacitys.append(dynamic_opacity)
                    dynamic_scales.append(dynamic_scale)
                    dynamic_rotations.append(dynamic_rotation)

                if args.render_all_cameras:
                    camera_renders = {camera_id: [] for camera_id in WAYMO_CAMERA_IDS}
                    camera_alphas = {camera_id: [] for camera_id in WAYMO_CAMERA_IDS}
                else:
                    chunked_renders, chunked_alphas = [], []
                source_extrinsic = extrinsic
                source_intrinsic = intrinsic
                if args.render_all_cameras:
                    rig_translation_scale = args.camera_rig_scale or estimate_waymo_rig_translation_scale(
                        scene_dir,
                        batch['image_paths'],
                        source_extrinsic,
                    )
                    print(f"Waymo rig translation scale for scene {scene_name}: {rig_translation_scale:.6f}")
                    print(f"Waymo calibration file mapping for scene {scene_name}: {canonical_to_stored}")
                    print(f"Waymo camera directions for scene {scene_name}: {format_camera_rig_directions(camera_to_ego, args.input_camera)}")
                    print(
                        f"Model source FOV for scene {scene_name}: "
                        f"({torch.rad2deg(2 * torch.atan(W / (2 * model_source_intrinsic[0, 0, 0]))).item():.1f}, "
                        f"{torch.rad2deg(2 * torch.atan(H / (2 * model_source_intrinsic[0, 1, 1]))).item():.1f})deg"
                    )
                    print(f"Calibrated camera FOV for scene {scene_name}: {format_camera_fovs(render_intrinsics, render_image_sizes)}")
                    render_extrinsic, render_intrinsic = build_all_camera_render_poses(
                        source_extrinsic,
                        camera_to_ego,
                        render_intrinsics,
                        args.input_camera,
                        rig_translation_scale,
                    )
                    frustum_coverage = compute_source_point_frustum_coverage(
                        point_map,
                        render_extrinsic,
                        render_intrinsic,
                        render_image_sizes,
                    )
                    print(
                        f"Source-depth frustum coverage for scene {scene_name}: "
                        + ", ".join(f"camera_{camera_id}={coverage:.4f}" for camera_id, coverage in frustum_coverage)
                    )
                if args.mode == 3:
                    origin_extrinsic = extrinsic
                    origin_intrinsic = intrinsic   
                for idx in range(dy_map.shape[1]):
                    if args.mode == 3:
                        I = intervals
                        t0 = timestamps[idx//I]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = gs_conf , gamma1 = 0.1)###

                        world_points, rgbs, opacity, scales, rotation = concat_list(
                            [static_points, static_rgbs, static_opacity_, static_scales, static_rotation],
                            [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                        )
                        for camera_offset in range(len(WAYMO_CAMERA_IDS)) if args.render_all_cameras else (0,):
                            camera_id = WAYMO_CAMERA_IDS[camera_offset]
                            render_idx = idx * len(WAYMO_CAMERA_IDS) + camera_offset
                            viewmats = render_extrinsic[render_idx:render_idx + 1] if args.render_all_cameras else extrinsic[idx:idx + 1]
                            Ks = render_intrinsic[render_idx:render_idx + 1] if args.render_all_cameras else intrinsic[idx:idx + 1]
                            render_height, render_width = render_image_sizes[camera_id] if args.render_all_cameras else (H, W)
                            renders_chunk, alphas_chunk, _ = rasterization(
                                means=world_points,
                                quats=rotation,
                                scales=scales,
                                opacities=opacity,
                                colors=rgbs,
                                viewmats=viewmats,
                                Ks=Ks,
                                width=render_width,
                                height=render_height,
                                render_mode='RGB+ED',
                            )
                            if args.render_all_cameras:
                                camera_renders[camera_id].append(renders_chunk)
                                camera_alphas[camera_id].append(alphas_chunk)
                            else:
                                chunked_renders.append(renders_chunk)
                                chunked_alphas.append(alphas_chunk)
                    if args.mode == 2:
                        t0 = timestamps[idx]
                        static_opacity_ = alpha_t(gs_timestamps, t0, static_opacity, gamma0 = static_gs_conf)
                        static_gs_list = [static_points, static_rgbs, static_opacity_, static_scales, static_rotations]
                        if dynamic_points:
                            world_points, rgbs, opacity, scales, rotation = concat_list(
                                static_gs_list,
                                [dynamic_points[idx], dynamic_rgbs[idx], dynamic_opacitys[idx], dynamic_scales[idx], dynamic_rotations[idx]]
                            )
                        else:
                            world_points, rgbs, opacity, scales, rotation = static_gs_list
                        height_offset = 0
                        for camera_offset in range(len(WAYMO_CAMERA_IDS)) if args.render_all_cameras else (0,):
                            camera_id = WAYMO_CAMERA_IDS[camera_offset]
                            render_idx = idx * len(WAYMO_CAMERA_IDS) + camera_offset
                            viewmats = render_extrinsic[render_idx:render_idx + 1] if args.render_all_cameras else extrinsic[idx:idx + 1]
                            Ks = render_intrinsic[render_idx:render_idx + 1] if args.render_all_cameras else intrinsic[idx:idx + 1]
                            render_height, render_width = render_image_sizes[camera_id] if args.render_all_cameras else (H, W)
                            renders_chunk, alphas_chunk, _ = rasterization(
                                means=world_points,
                                quats=rotation,
                                scales=scales,
                                opacities=opacity,
                                colors=rgbs,
                                viewmats=viewmats,
                                Ks=Ks,
                                width=render_width,
                                height=render_height,
                                render_mode='RGB+ED',
                            )
                            if args.render_all_cameras:
                                camera_renders[camera_id].append(renders_chunk)
                                camera_alphas[camera_id].append(alphas_chunk)
                            else:
                                chunked_renders.append(renders_chunk)
                                chunked_alphas.append(alphas_chunk)
                if args.render_all_cameras:
                    rendered_images_by_camera = {}
                    coverage = []
                    for camera_offset, camera_id in enumerate(WAYMO_CAMERA_IDS):
                        foreground = torch.cat(camera_renders[camera_id], dim=0)
                        alpha = torch.cat(camera_alphas[camera_id], dim=0)
                        foreground = foreground[..., :-1]
                        target_extrinsic = render_extrinsic[camera_offset::len(WAYMO_CAMERA_IDS)]
                        target_intrinsic = render_intrinsic[camera_offset::len(WAYMO_CAMERA_IDS)]
                        render_height, render_width = render_image_sizes[camera_id]
                        bg_render = model.sky_model.forward_with_new_pose(
                            images,
                            source_extrinsic,
                            source_intrinsic,
                            target_extrinsic,
                            target_intrinsic,
                            output_height=render_height,
                            output_width=render_width,
                        )
                        bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)
                        if bg_render.shape[0] != foreground.shape[0]:
                            raise RuntimeError(
                                f"Foreground and sky render counts differ for camera {camera_id}: "
                                f"{foreground.shape[0]} and {bg_render.shape[0]}."
                            )
                        rendered_images_by_camera[camera_id] = (
                            alpha * foreground + (1 - alpha) * bg_render
                        ).permute(0, 3, 1, 2)
                        coverage.append(f"camera_{camera_id}={alpha.mean().item():.4f}")
                    print(f"Gaussian alpha coverage for scene {scene_name}: {', '.join(coverage)}")
                else:
                    renders = torch.cat(chunked_renders, dim=0)
                    depth_maps = renders[..., -1]
                    renders = renders[..., :-1]
                    alphas = torch.cat(chunked_alphas, dim=0)
                if not args.render_all_cameras and args.mode == 3:
                    bg_render = model.sky_model.forward_with_new_pose(images,origin_extrinsic,origin_intrinsic, extrinsic, intrinsic)
                elif not args.render_all_cameras and args.mode == 2:
                    bg_render = model.sky_model(images, extrinsic, intrinsic)
                    bg_render = (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)  #
                if not args.render_all_cameras and bg_render.shape[0] != renders.shape[0]:
                    raise RuntimeError(
                        f"Foreground and sky render counts differ: {renders.shape[0]} and {bg_render.shape[0]}."
                    )
                if not args.render_all_cameras:
                    renders = alphas * renders + (1 - alphas) * bg_render
                    rendered_image = renders.permute(0, 3, 1, 2)
                    target_image = images[0]

            inference_time = time.time() - start_time
            inference_time_list.append(inference_time)
            if args.diffusion:
                if args.render_all_cameras:
                    for camera_id, camera_images in rendered_images_by_camera.items():
                        processed_frames = []
                        for frame in camera_images:
                            processed_frames.append(
                                process_images_with_difix(frame.detach().cpu().clamp(0, 1), "path_to_diffusion_model")
                            )
                        rendered_images_by_camera[camera_id] = torch.stack(processed_frames, dim=0).to(device)
                else:
                    processed_frames = []
                    for i in range(rendered_image.shape[0]):
                        frame = rendered_image[i].detach().cpu().clamp(0, 1)
                        processed_frame = process_images_with_difix(frame, "path_to_diffusion_model")
                        processed_frames.append(processed_frame)
                    rendered_image = torch.stack(processed_frames, dim=0).to(device)
            
            if args.metrics:
                psnr, ssim, lpip = compute_metrics(rendered_image, target_image, loss_fn)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpip)
            scene_idx += 1

            scene_out_dir = os.path.join(args.output_path, scene_name)
            os.makedirs(scene_out_dir, exist_ok=True)

            if args.images or args.render_all_cameras:
                if args.render_all_cameras:
                    for frame_idx in range(source_extrinsic.shape[0]):
                        for camera_id in WAYMO_CAMERA_IDS:
                            rendered = rendered_images_by_camera[camera_id][frame_idx].detach().cpu().clamp(0, 1)
                            camera_out_dir = os.path.join(scene_out_dir, f"camera_{camera_id}")
                            os.makedirs(camera_out_dir, exist_ok=True)
                            T.ToPILImage()(rendered).save(os.path.join(camera_out_dir, f"frame_{frame_idx:04d}.png"))
                    video_path = os.path.join(scene_out_dir, "all_camera_comparison.mp4")
                    with imageio.get_writer(video_path, fps=8, codec="libx264") as writer:
                        for frame_idx, frame_id in enumerate(frame_ids):
                            ground_truth_images = load_waymo_comparison_ground_truth(scene_dir, frame_id)
                            comparison_frame = make_waymo_all_camera_comparison_frame(
                                {
                                    camera_id: rendered_images_by_camera[camera_id][frame_idx]
                                    for camera_id in WAYMO_CAMERA_IDS
                                },
                                ground_truth_images,
                                dy_map[0, frame_idx],
                            )
                            writer.append_data(comparison_frame.permute(1, 2, 0).mul(255).byte().numpy())
                    print("Saved all-camera comparison video:", video_path)
                elif args.input_views == 1:
                    image_list = []
                    for i in range(rendered_image.shape[0]):
                        rendered = rendered_image[i].detach().cpu().clamp(0, 1)
                        image_path = os.path.join(scene_out_dir, f"view_{i}.png")
                        T.ToPILImage()(rendered).save(image_path)
                        image_list.append(rendered.permute(1, 2, 0).numpy())
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, (np.array(image_list) * 255).astype(np.uint8), fps=8, codec="libx264")
                elif args.input_views == 3:
                    T_total = rendered_image.shape[0]
                    groups = T_total // 3
                    video_list = []
                    for g in range(groups):
                        idx_center = 3 * g + 0
                        idx_left = 3 * g + 1
                        idx_right = 3 * g + 2
                        center = rendered_image[idx_center].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        left = rendered_image[idx_left].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        right = rendered_image[idx_right].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                        H, W = center.shape[0], center.shape[1]
                        # convert to uint8 HWC
                        def to_uint8(arr):
                            a = (arr * 255.0).astype(np.uint8)
                            if a.ndim == 2:
                                a = np.stack([a] * 3, axis=-1)
                            if a.shape[2] == 4:
                                a = a[:, :, :3]
                            return a
                        left_u = to_uint8(left)
                        center_u = to_uint8(center)
                        right_u = to_uint8(right)                            
                        white = np.ones((H,10, 3), dtype=np.uint8) * 255
                        composed = np.concatenate([left_u, white, center_u, white, right_u], axis=1)
                        # save image
                        image_path = os.path.join(scene_out_dir, f"view_{g:04d}.png")
                        Image.fromarray(composed).save(image_path)
                        video_list.append(composed)
                    video_path = os.path.join(scene_out_dir, "rendered_video.mp4")
                    imageio.mimwrite(video_path, np.array(video_list), fps=8, codec="libx264")

            if args.mode == 2:
                depth_frames = predictions["depth"][0].detach().cpu()
            if args.mode == 3:
                depth_frames = depth_interp[0].detach().cpu()

            if not plain_image_inference and not args.render_all_cameras:
                gt_frames = target_image.detach().cpu()
                pred_frames = rendered_image.detach().cpu()
                dyn_frames = dy_map[0].sigmoid().detach().cpu()
                gt_dy_map = gt_dy_map.mean(dim=2)
                gt_dy_map = gt_dy_map[0].sigmoid().detach().cpu()
                gt_depth = gt_depth[..., 0:1]
                gt_depth = gt_depth[0].squeeze(-1).detach().cpu()
                if args.mode == 2:
                    sky_mask = sky_mask.detach().cpu()
                if args.mode == 3:
                    sky_mask = target_sky_masks.permute(0, 1, 3, 4, 2).detach().cpu()
                out_video = os.path.join(scene_out_dir, "comparison.mp4")
                make_comparison_video_quad(
                    gt_frames,
                    pred_frames,
                    gt_dy_map,
                    dyn_frames,
                    gt_depth,
                    depth_frames,
                    sky_mask,
                    out_video,
                    fps=8,
                    views=args.input_views,
                )
                print("Saved comparison video:", out_video)

            if args.depth:
                S = depth_frames.shape[0]

                if args.input_views == 1:
                    for i in range(S):
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{i}.npy"), depth_i)
                elif args.input_views == 3:
                    for i in range(S):
                        view_id = i % 3
                        frame_id = i // 3
                        depth_i = depth_frames[i].numpy()
                        np.save(os.path.join(scene_out_dir, f"view_{frame_id:04d}_{view_id}.npy"), depth_i)
    if args.metrics:
        print("PSNR:", sum(psnr_list) / len(psnr_list))
        print("SSIM:", sum(ssim_list) / len(ssim_list))
        print("LPIPS:", sum(lpips_list) / len(lpips_list))
        print("Avg Inference Time (s):", sum(inference_time_list) / len(inference_time_list))

if __name__ == "__main__":
    main()
