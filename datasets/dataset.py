import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
import os
import re
import io
import pickle
import struct
import subprocess
from collections import OrderedDict
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
import numpy as np
from tqdm import tqdm


OPENCV2WAYMO = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)

def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()

    
def load_and_preprocess_flow(flow_path_list, extrinsic_paths, intrinsic_path, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 image is required")

    flows = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # print(f"[DEBUG] flow_path_list: {flow_path_list}")
    for i, flow_path in enumerate(flow_path_list):
        # print(f"[DEBUG] Processing flow_path[{i}]: {flow_path}")
        # print(f"[DEBUG] Is file: {os.path.isfile(flow_path) if flow_path else 'Empty/None'}")
        # print(f"[DEBUG] Is dir: {os.path.isdir(flow_path) if flow_path else 'Empty/None'}")
        depth_and_flow = np.load(flow_path)
        flow = depth_and_flow
        flow = torch.tensor(flow).float()
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    
    return torch.stack(flows)


def load_and_preprocess_images(image_path_list, mode="crop"):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Accept paths/file objects as before, plus already-decoded PIL images
        # used by the streaming Parquet dataset.
        img = image_path.copy() if isinstance(image_path, Image.Image) else Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


class WaymoParquetSkyDataset(Dataset):
    """Stream one random temporal interval per Waymo v2 scene and epoch.

    No decoded images or masks are written to disk. Interval length is sampled
    from ``sequence_lengths`` independently whenever a scene is fetched.
    """

    IMAGE_COLUMN = "[CameraImageComponent].image"
    SEGMENTATION_COLUMN = "[CameraSegmentationLabelComponent].panoptic_label"
    DIVISOR_COLUMN = "[CameraSegmentationLabelComponent].panoptic_label_divisor"
    POSE_COLUMN = "[VehiclePoseComponent].world_from_vehicle.transform"
    EXTRINSIC_COLUMN = "[CameraCalibrationComponent].extrinsic.transform"
    INTRINSIC_COLUMNS = tuple(
        f"[CameraCalibrationComponent].intrinsic.{name}" for name in ("f_u", "f_v", "c_u", "c_v")
    )
    TIMESTAMP_COLUMN = "key.frame_timestamp_micros"
    CAMERA_COLUMN = "key.camera_name"
    # Waymo camera segmentation enum: TYPE_SKY = 25.
    SKY_CLASS_ID = 25

    def __init__(
        self, raw_dir, scene_names=None, sequence_lengths=(4, 8, 12), fixed_start=None,
        camera_name=1, cache_segments=2,
        segformer_python=None, segformer_path=None, segformer_config=None,
        segformer_checkpoint=None, segformer_device="cuda:0", sky_mask_cache_dir=None,
    ):
        sequence_lengths = tuple(int(length) for length in sequence_lengths)
        if not sequence_lengths or any(length < 1 for length in sequence_lengths):
            raise ValueError("sequence_lengths must contain positive integers")
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError(
                "Raw Waymo v2 training requires pyarrow. Install it with `pip install pyarrow`."
            ) from error

        self.pq = pq
        self.raw_dir = os.path.abspath(raw_dir)
        self.sequence_lengths = sequence_lengths
        self.fixed_start = fixed_start
        if fixed_start is not None and fixed_start < 0:
            raise ValueError("fixed_start must be non-negative")
        self.camera_name = camera_name
        self.cache_segments = max(1, cache_segments)
        self._cache = OrderedDict()
        self.sky_mask_cache_dir = (
            os.path.abspath(sky_mask_cache_dir) if sky_mask_cache_dir else None
        )
        if self.sky_mask_cache_dir:
            os.makedirs(self.sky_mask_cache_dir, exist_ok=True)
        self._segformer_process = None
        self.segformer_command = None
        if segformer_python:
            required = (segformer_path, segformer_config, segformer_checkpoint)
            if any(value is None for value in required):
                raise ValueError("SegFormer python, repository, config, and checkpoint must be provided together")
            worker_script = os.path.join(os.path.dirname(__file__), "tools", "segformer_stream.py")
            self.segformer_command = [
                segformer_python, worker_script,
                "--segformer_path", segformer_path,
                "--config", segformer_config,
                "--checkpoint", segformer_checkpoint,
                "--device", segformer_device,
            ]

        component_dirs = ("camera_image", "camera_calibration", "vehicle_pose")
        for component in component_dirs:
            directory = os.path.join(self.raw_dir, component)
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Missing Waymo v2 component directory: {directory}")

        filenames = sorted(name for name in os.listdir(os.path.join(self.raw_dir, "camera_image")) if name.endswith(".parquet"))
        if scene_names:
            selected = []
            for name in scene_names:
                if str(name).isdigit() and len(str(name)) <= 3:
                    index = int(name)
                    if index >= len(filenames):
                        raise IndexError(f"Raw Waymo scene index {index} is outside 0..{len(filenames) - 1}")
                    selected.append(filenames[index])
                else:
                    filename = str(name) if str(name).endswith(".parquet") else f"{name}.parquet"
                    if filename not in filenames:
                        raise FileNotFoundError(f"Unknown Waymo segment: {name}")
                    selected.append(filename)
            filenames = selected
        if not filenames:
            raise ValueError(f"No Waymo Parquet segments selected under {self.raw_dir}")

        self.segments = []
        self.segment_timestamps = {}
        self.segment_frame_counts = {}
        for filename in tqdm(filenames, desc="Indexing Waymo Parquet segments", unit="segment"):
            image_path = self._path("camera_image", filename)
            image_keys = self._camera_timestamps(image_path)
            timestamps = sorted(image_keys)
            valid_lengths = tuple(length for length in self.sequence_lengths if length <= len(timestamps))
            if not valid_lengths:
                continue
            self.segments.append(filename)
            self.segment_timestamps[filename] = timestamps
            self.segment_frame_counts[filename] = len(timestamps)
        if not self.segments:
            raise ValueError("No Waymo scenes contain enough images for the requested sequence lengths")

    def _path(self, component, filename):
        return os.path.join(self.raw_dir, component, filename)

    def _camera_timestamps(self, path):
        table = self.pq.read_table(path, columns=[self.TIMESTAMP_COLUMN, self.CAMERA_COLUMN])
        timestamps = table[self.TIMESTAMP_COLUMN].to_pylist()
        cameras = table[self.CAMERA_COLUMN].to_pylist()
        return {timestamp for timestamp, camera in zip(timestamps, cameras) if camera == self.camera_name}

    def _load_segment(self, filename):
        if filename in self._cache:
            self._cache.move_to_end(filename)
            return self._cache[filename]

        def rows(component, columns, camera=True):
            requested = [self.TIMESTAMP_COLUMN, *columns]
            if camera:
                requested.append(self.CAMERA_COLUMN)
            table = self.pq.read_table(self._path(component, filename), columns=requested)
            result = {}
            for row_index, timestamp in enumerate(table[self.TIMESTAMP_COLUMN].to_pylist()):
                if not camera or table[self.CAMERA_COLUMN][row_index].as_py() == self.camera_name:
                    result[timestamp] = tuple(table[column][row_index].as_py() for column in columns)
            return result

        calibration_table = self.pq.read_table(
            self._path("camera_calibration", filename),
            columns=[self.CAMERA_COLUMN, *self.INTRINSIC_COLUMNS, self.EXTRINSIC_COLUMN],
        )
        calibration = None
        for row_index, camera in enumerate(calibration_table[self.CAMERA_COLUMN].to_pylist()):
            if camera == self.camera_name:
                intrinsic = [calibration_table[column][row_index].as_py() for column in self.INTRINSIC_COLUMNS]
                extrinsic = np.asarray(calibration_table[self.EXTRINSIC_COLUMN][row_index].as_py(), dtype=np.float64).reshape(4, 4)
                calibration = (intrinsic, extrinsic)
                break
        if calibration is None:
            raise RuntimeError(f"Camera {self.camera_name} has no calibration in {filename}")

        data = {
            "filename": filename,
            "images": rows("camera_image", [self.IMAGE_COLUMN]),
            "poses": rows("vehicle_pose", [self.POSE_COLUMN], camera=False),
            "calibration": calibration,
            "generated_sky_masks": {},
        }
        self._cache[filename] = data
        while len(self._cache) > self.cache_segments:
            self._cache.popitem(last=False)
        return data

    def _ensure_segformer(self):
        if self.segformer_command is None:
            raise RuntimeError("Online SegFormer was not configured")
        if self._segformer_process is None:
            self._segformer_process = subprocess.Popen(
                self.segformer_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0,
            )
        if self._segformer_process.poll() is not None:
            raise RuntimeError(f"SegFormer worker exited with code {self._segformer_process.returncode}")
        return self._segformer_process

    @staticmethod
    def _write_pipe_message(stream, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        stream.write(struct.pack("!Q", len(payload)))
        stream.write(payload)
        stream.flush()

    @staticmethod
    def _read_pipe_message(stream):
        def read_exact(size):
            chunks = []
            remaining = size
            while remaining:
                chunk = stream.read(remaining)
                if not chunk:
                    raise RuntimeError(f"SegFormer pipe closed with {remaining} bytes still expected")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        header = read_exact(8)
        if len(header) != 8:
            raise RuntimeError("SegFormer worker closed its output pipe")
        size = struct.unpack("!Q", header)[0]
        payload = read_exact(size)
        return pickle.loads(payload)

    def _generate_sky_masks(self, segment, timestamps):
        cache = segment["generated_sky_masks"]
        filename = segment["filename"]
        if self.sky_mask_cache_dir:
            segment_dir = os.path.join(
                self.sky_mask_cache_dir, os.path.splitext(filename)[0]
            )
            os.makedirs(segment_dir, exist_ok=True)
            for timestamp in timestamps:
                if timestamp in cache:
                    continue
                mask_path = os.path.join(segment_dir, f"{timestamp}.npy")
                if not os.path.isfile(mask_path):
                    continue
                try:
                    mask = np.load(mask_path, allow_pickle=False)
                    if mask.ndim != 2:
                        raise ValueError(f"expected a 2D mask, got {mask.shape}")
                    cache[timestamp] = (mask != 0).astype(np.uint8, copy=False)
                except (OSError, ValueError, EOFError) as error:
                    print(f"Ignoring invalid cached sky mask {mask_path}: {error}")

        missing = [timestamp for timestamp in timestamps if timestamp not in cache]
        # Keep protocol messages and peak host memory small. SegFormer itself
        # still runs one image at a time inside the persistent worker.
        request_size = 4
        for start in range(0, len(missing), request_size):
            requested = missing[start : start + request_size]
            process = self._ensure_segformer()
            self._write_pipe_message(process.stdin, [segment["images"][timestamp][0] for timestamp in requested])
            try:
                response = self._read_pipe_message(process.stdout)
            except RuntimeError as error:
                return_code = process.poll()
                raise RuntimeError(
                    f"SegFormer worker communication failed (exit_code={return_code}): {error}"
                ) from error
            if "error" in response:
                raise RuntimeError(f"Online SegFormer failed: {response['error']}")
            for timestamp, mask in zip(requested, response["masks"]):
                mask = (np.asarray(mask) != 0).astype(np.uint8, copy=False)
                cache[timestamp] = mask
                if self.sky_mask_cache_dir:
                    mask_path = os.path.join(segment_dir, f"{timestamp}.npy")
                    temporary_path = f"{mask_path}.tmp.{os.getpid()}"
                    try:
                        with open(temporary_path, "wb") as mask_file:
                            np.save(mask_file, mask, allow_pickle=False)
                            mask_file.flush()
                            os.fsync(mask_file.fileno())
                        os.replace(temporary_path, mask_path)
                    finally:
                        if os.path.exists(temporary_path):
                            os.unlink(temporary_path)
        return [cache[timestamp] for timestamp in timestamps]

    def close(self):
        process = getattr(self, "_segformer_process", None)
        if process is not None and process.poll() is None:
            try:
                self._write_pipe_message(process.stdin, "stop")
                process.wait(timeout=5)
            except Exception:
                process.terminate()

    def __del__(self):
        self.close()

    def __len__(self):
        # One randomly sampled temporal interval per scene and epoch.
        return len(self.segments)

    @staticmethod
    def _preprocess_calibration(intrinsic, image_size, output_height):
        width, height = image_size
        new_height = round(height * (518.0 / width) / 14) * 14
        crop_top = max((new_height - 518) // 2, 0)
        pad_top = (output_height - min(new_height, 518)) // 2
        fx, fy, cx, cy = intrinsic
        return np.array(
            [[fx * 518.0 / width, 0, cx * 518.0 / width],
             [0, fy * new_height / height, cy * new_height / height - crop_top + pad_top],
             [0, 0, 1]], dtype=np.float32,
        )

    def __getitem__(self, index):
        filename = self.segments[index]
        scene_timestamps = self.segment_timestamps[filename]
        valid_lengths = [length for length in self.sequence_lengths if length <= len(scene_timestamps)]
        sequence_length = random.choice(valid_lengths)
        if self.fixed_start is None:
            start = random.randint(0, len(scene_timestamps) - sequence_length)
        else:
            start = self.fixed_start
            if start + sequence_length > len(scene_timestamps):
                raise IndexError(
                    f"Fixed interval {start}:{start + sequence_length} exceeds "
                    f"the {len(scene_timestamps)} frames in {filename}"
                )
        timestamps = tuple(scene_timestamps[start : start + sequence_length])
        segment = self._load_segment(filename)
        image_objects = [Image.open(io.BytesIO(segment["images"][timestamp][0])).convert("RGB") for timestamp in timestamps]
        images = load_and_preprocess_images(image_objects)

        generated_masks = self._generate_sky_masks(segment, timestamps)
        mask_objects = [Image.fromarray(mask * 255, mode="L") for mask in generated_masks]
        masks = load_and_preprocess_images(mask_objects)

        intrinsic_values, camera_to_vehicle = segment["calibration"]
        output_height = images.shape[-2]
        intrinsics = np.stack([
            self._preprocess_calibration(intrinsic_values, image.size, output_height) for image in image_objects
        ])
        origin_pose = np.asarray(segment["poses"][timestamps[0]][0], dtype=np.float64).reshape(4, 4)
        camera_to_vehicle = camera_to_vehicle @ OPENCV2WAYMO
        camera_to_worlds = []
        for timestamp in timestamps:
            world_from_vehicle = np.asarray(segment["poses"][timestamp][0], dtype=np.float64).reshape(4, 4)
            camera_to_worlds.append((np.linalg.inv(origin_pose) @ world_from_vehicle @ camera_to_vehicle).astype(np.float32))

        relative_timestamps = np.asarray(timestamps, dtype=np.float64) - timestamps[0]
        relative_timestamps = relative_timestamps / max(relative_timestamps[-1], 1.0) * (sequence_length / 4)
        return {
            "images": images,
            "masks": masks,
            "sky_mask_valid": torch.ones(len(timestamps), dtype=torch.float32),
            "timestamps": torch.from_numpy(relative_timestamps.astype(np.float32)),
            "intrinsics": torch.from_numpy(intrinsics),
            "camera_to_worlds": torch.from_numpy(np.stack(camera_to_worlds)),
            "image_paths": [f"{filename}:{timestamp}:{self.camera_name}" for timestamp in timestamps],
        }


def load_waymo_calibrations(scene_dir, image_paths):
    """Load calibrations in the same flattened order and transform intrinsics with image preprocessing."""
    image_metadata = []
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        frame_id, camera_id = stem.rsplit("_", maxsplit=1)
        with Image.open(path) as image:
            width, height = image.size
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        output_height = min(new_height, 518)
        image_metadata.append((frame_id, int(camera_id), width, height, new_height, output_height))

    max_height = max(metadata[-1] for metadata in image_metadata)
    intrinsics, camera_to_worlds = [], []
    origin_frame = image_metadata[0][0]
    origin_ego_pose = np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{origin_frame}.txt"))

    for frame_id, camera_id, width, height, new_height, output_height in image_metadata:
        raw_intrinsics = np.loadtxt(os.path.join(scene_dir, "intrinsics", f"{camera_id}.txt"))
        fx, fy, cx, cy = raw_intrinsics[:4]
        scale_x = 518.0 / width
        scale_y = new_height / height
        crop_top = max((new_height - 518) // 2, 0)
        pad_top = (max_height - output_height) // 2
        intrinsic = np.array(
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y - crop_top + pad_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        camera_to_ego = np.loadtxt(os.path.join(scene_dir, "extrinsics", f"{camera_id}.txt"))
        camera_to_ego = camera_to_ego @ OPENCV2WAYMO
        ego_pose = np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{frame_id}.txt"))
        ego_to_local_world = np.linalg.inv(origin_ego_pose) @ ego_pose
        camera_to_world = ego_to_local_world @ camera_to_ego
        intrinsics.append(intrinsic)
        camera_to_worlds.append(camera_to_world.astype(np.float32))

    return torch.from_numpy(np.stack(intrinsics)), torch.from_numpy(np.stack(camera_to_worlds))


class ImageDirectoryDataset(Dataset):
    """Loads ordinary image files for unlabelled reconstruction inference."""

    image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(self, image_dir, sequence_length=4, start_idx=0):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        self.image_dir = os.path.abspath(image_dir)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

        image_paths = []
        for root, dirnames, filenames in os.walk(self.image_dir):
            dirnames.sort(key=self._natural_sort_key)
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in self.image_extensions:
                    image_paths.append(os.path.join(root, filename))
        image_paths.sort(key=lambda path: self._natural_sort_key(os.path.relpath(path, self.image_dir)))

        if not image_paths:
            raise ValueError(f"No supported images found under: {image_dir}")

        if start_idx < 0 or start_idx >= len(image_paths):
            raise ValueError(
                f"start_idx must be in [0, {max(len(image_paths) - 1, 0)}], got {start_idx}"
            )

        image_paths = image_paths[start_idx:]
        self.sequences = [
            image_paths[index : index + sequence_length]
            for index in range(0, len(image_paths), sequence_length)
        ]

    @staticmethod
    def _natural_sort_key(value):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_paths = self.sequences[idx]
        images = load_and_preprocess_images(image_paths)
        sequence_length, _, height, width = images.shape

        return {
            "images": images,
            "masks": torch.zeros_like(images),
            "dynamic_mask": torch.zeros(sequence_length, 1, height, width),
            "gt_depth": torch.zeros(sequence_length, height, width),
            "timestamps": torch.arange(sequence_length, dtype=torch.float32),
        }


class LegacyWaymoOpenDataset(Dataset):
    def __init__(self, image_dir, scene_names = None, sequence_length= None, start_idx = -1, mode=1, views=1, intervals=2, input_camera=0):
        #mode 1 : train
        #mode 2 : pure reconstruction
        #mode 3 : interplation
        
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        if mode == 1:
            interval = 1
        elif mode == 2:
            interval = 1
        elif mode == 3:
            interval = intervals
        else:
            interval = 1
        self.interval =  interval
        self.mode = mode
        if mode == 1:
            test_mode = False
            load_flow = False
        elif mode == 2:
            test_mode = True
            load_flow = False
        elif mode == 3:
            test_mode = True
            load_flow = True
        else:
            pass
        self.test_mode = test_mode
        self.load_flow = load_flow
        self.views = views
        if not 0 <= input_camera <= 4:
            raise ValueError(f"input_camera must be in [0, 4], got {input_camera}")
        self.input_camera = input_camera

        # Scan all scene folders and collect image paths
        if scene_names is None:
            scene_names = [] 
            scene_names_ = [f"{i:03d}" for i in range(0, 99)]
            scene_names = scene_names + scene_names_
        self.scenes = scene_names
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.semantic_mask_path = []
        self.depth_flow_paths = []
        self.ego_paths = []

        self.start_idx = start_idx

        for scene_name in scene_names:
            scene_path = os.path.join(image_dir, scene_name, "images")
            if os.path.isdir(scene_path):
                # image
                if self.views == 1:
                    image_paths = sorted(
                        [
                            os.path.join(scene_path, f)
                            for f in os.listdir(scene_path)
                            if f.endswith((f"_{self.input_camera}.jpg", f"_{self.input_camera}.png"))
                        ]
                    )
                    self.image_paths.append(image_paths)
                elif self.views == 3:
                    views_image_lists = []
                    for v in range(3):
                        suffixes = (f"_{v}.jpg", f"_{v}.png")
                        files_v = sorted(
                            [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(suffixes)]
                        )
                        views_image_lists.append(files_v)
                    lengths = [len(l) for l in views_image_lists]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(f"Inconsistent number of images across views in scene {scene_name}, lengths: {lengths}")
                    self.image_paths.append(views_image_lists)

                # sky_mask
                sky_mask_path = os.path.join(image_dir, scene_name, "sky_masks")
                if os.path.isdir(sky_mask_path):
                    if self.views == 1:
                        sky_mask_paths = sorted(
                            [os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith((f"_{self.input_camera}.jpg", f"_{self.input_camera}.png"))]
                        )
                        self.sky_mask_paths.append(sky_mask_paths)
                    elif self.views == 3:
                        views_sky_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(suffixes)])
                            views_sky_lists.append(files_v)
                        self.sky_mask_paths.append(views_sky_lists)
                else:
                    self.sky_mask_paths.append([] if self.views == 1 else [[] for _ in range(3)])

                # extrinsic
                extrinsic_path = os.path.join(image_dir, scene_name, "ego_pose")
                if os.path.isdir(extrinsic_path):
                    extrinsic_paths = sorted([
                        os.path.join(extrinsic_path, f)
                        for f in os.listdir(extrinsic_path)
                        if f.endswith(".txt")
                    ])
                    self.extrinsic_paths.append(extrinsic_paths)
                else:
                    self.extrinsic_paths.append([])

                # extrinsic
                ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                # ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                if os.path.isdir(ego_path):
                    ego_path = os.path.join(ego_path, f"{self.input_camera}.txt")
                    self.ego_paths.append(ego_path)

                # intrinsic
                intrinsic_path = os.path.join(image_dir, scene_name, "intrinsics")
                if os.path.isdir(intrinsic_path):
                    if self.views == 1:
                        intrinsic_paths = os.path.join(intrinsic_path, f"{self.input_camera}.txt")
                        self.intrinsic_paths.append(intrinsic_paths)
                    elif self.views == 3:
                        intrinsics_views = []
                        for v in range(3):
                            p = os.path.join(intrinsic_path, f"{v}.txt")
                            intrinsics_views.append(p if os.path.exists(p) else "")
                        self.intrinsic_paths.append(intrinsics_views)
                else:
                    self.intrinsic_paths.append("" if self.views == 1 else ["" for _ in range(3)])

                # dynamic mask
                dynamic_mask_path = os.path.join(image_dir, scene_name, "fine_dynamic_masks/all")
                if os.path.isdir(dynamic_mask_path):
                    if self.views == 1:
                        dynamic_mask_paths = sorted(
                            [os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith((f"_{self.input_camera}.jpg", f"_{self.input_camera}.png"))]
                        )
                        self.dynamic_mask_path.append(dynamic_mask_paths)
                    elif self.views == 3:
                        views_dyn_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(suffixes)])
                            views_dyn_lists.append(files_v)
                        self.dynamic_mask_path.append(views_dyn_lists)
                else:
                    self.dynamic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                # depth
                depth_path = os.path.join(image_dir, scene_name, "depth_flows_4")
                if os.path.isdir(depth_path):
                    if self.views == 1:
                        depth_paths = sorted(
                            [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(f"_{self.input_camera}.npy")]
                        )
                        self.depth_flow_paths.append(depth_paths)
                    elif self.views == 3:
                        views_depth_lists = []
                        for v in range(3):
                            suffix = f"_{v}.npy"
                            files_v = sorted(
                                [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(suffix)]
                            )
                            views_depth_lists.append(files_v)
                        self.depth_flow_paths.append(views_depth_lists)
                else:
                    self.depth_flow_paths.append([] if self.views == 1 else [[] for _ in range(3)])
                # semantic mask
                semantic_mask_path = os.path.join(image_dir, scene_name, "custom_masks")
                if os.path.isdir(semantic_mask_path):
                    if self.views == 1:
                        semantic_mask_paths = sorted(
                            [os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith((f"_{self.input_camera}.jpg", f"_{self.input_camera}.png"))]
                        )
                        self.semantic_mask_path.append(semantic_mask_paths)
                    elif self.views == 3:
                        views_sem_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(suffixes)])
                            views_sem_lists.append(files_v)
                        self.semantic_mask_path.append(views_sem_lists)
                else:
                    self.semantic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        sky_mask_paths = self.sky_mask_paths[idx]
        dynamic_mask_paths = self.dynamic_mask_path[idx]
        semantic_mask_paths = self.semantic_mask_path[idx]

        start_idx = random.randint(0, max(1, len(image_paths[0] if self.views == 3 else image_paths) - 21))

        if self.mode == 1:
            indices = [start_idx]
            intervals = sorted(random.sample(range(1, 20), self.sequence_length - 1))
            for interval in intervals:
                indices.append(start_idx + interval)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                "interval": intervals,
            }

        
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            
            # if len(semantic_mask_paths) > 0:
            #     if self.views == 1:
            #         sem_mask_seq = [semantic_mask_paths[i] for i in indices]
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S, C, H, W]
            #     elif self.views == 3:
            #         sem_mask_seq = []
            #         for i in indices:
            #             for v in range(3):
            #                 sem_mask_seq.append(semantic_mask_paths[v][i])
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S*3, C, H, W]
            #     semantic_mask = semantic_mask * 255 / 10
            #     semantic_mask = semantic_mask.int()
            #     semantic_mask[semantic_mask > 9] = 255
            #     input_dict["semantic_mask"] = semantic_mask

            return input_dict

        elif self.mode == 2: 
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]
                


            input_dict = {
                "images": images,
                "image_paths": seq,
                "masks": masks,
                "timestamps": timestamps,
                "interval": intervals,
            }
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                else:
                    depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = depth_data

            return input_dict

        else:  # self.mode == 3
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            target_indices = [start_idx + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

            timestamps = np.array(indices) - start_idx
            timestamps = timestamps / timestamps[-1] * (self.sequence_length / 4)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
            
            # images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
                target_seq = [image_paths[i] for i in target_indices]
                target_images = load_and_preprocess_images(target_seq)  # [T, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

                target_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_seq.append(image_paths[v][i])
                target_images = load_and_preprocess_images(target_seq)  # [T*3, C, H, W]

            # sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_images(mask_seq)  # [S, C, H, W]
                target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_images(mask_seq)  # [S*3, C, H, W]

                target_mask_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_mask_seq.append(sky_mask_paths[v][i])
                target_masks = load_and_preprocess_images(target_mask_seq)  # [T*3, C, H, W]

            input_dict = {
                "images": images,
                "targets": target_images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                # "target_timestamps": target_timestamps,
                "interval": intervals,
                "target_masks": target_masks,
            }

            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)  # [S, C, H, W]
                    target_dy_mask_seq = [dynamic_mask_paths[i] for i in target_indices]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    target_dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    for i in target_indices:
                        for v in range(3):
                            target_dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_images(dy_mask_seq)         # [S*3, C, H, W]
                    target_dynamic_mask = load_and_preprocess_images(target_dy_mask_seq)  # [T*3, C, H, W]
                input_dict["dynamic_mask"] = target_dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_seq = [self.depth_flow_paths[idx][i] for i in target_indices]
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        target_depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        for i in target_indices:
                            for v in range(3):
                                target_depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                else:
                    target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = target_depth_data
            return input_dict


class WaymoOpenDataset(Dataset):
    """Waymo dataset loader with an arbitrary, ordered list of input cameras."""

    def __init__(
        self,
        image_dir,
        scene_names=None,
        sequence_length=None,
        start_idx=-1,
        mode=1,
        views=1,
        intervals=2,
        frame_stride=1,
        input_camera=0,
    ):
        if sequence_length is None or sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be at least 1, got {frame_stride}")
        input_cameras = (input_camera,) if isinstance(input_camera, int) else tuple(input_camera)
        if not input_cameras:
            raise ValueError("At least one input camera must be specified.")
        if len(set(input_cameras)) != len(input_cameras):
            raise ValueError(f"Input cameras must be unique, got {input_cameras}.")
        if any(camera_id not in range(5) for camera_id in input_cameras):
            raise ValueError(f"Input cameras must be in [0, 4], got {input_cameras}.")
        if views != len(input_cameras):
            raise ValueError(
                f"views ({views}) must equal the number of input cameras ({len(input_cameras)})."
            )

        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.start_idx = start_idx
        self.mode = mode
        self.interval = intervals if mode == 3 else 1
        self.frame_stride = frame_stride
        self.views = len(input_cameras)
        self.input_cameras = input_cameras
        self.input_camera = input_cameras[0]
        self.scenes = scene_names or [f"{scene_id:03d}" for scene_id in range(99)]
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.depth_flow_paths = []

        for scene_name in self.scenes:
            scene_dir = os.path.join(image_dir, scene_name)
            images_dir = os.path.join(scene_dir, "images")
            self.image_paths.append(self._collect_view_paths(images_dir, (".jpg", ".png"), scene_name, "images"))
            self.sky_mask_paths.append(
                self._collect_optional_view_paths(
                    os.path.join(scene_dir, "sky_masks"), (".jpg", ".png"), scene_name, "sky masks"
                )
            )
            self.dynamic_mask_path.append(
                self._collect_optional_view_paths(
                    os.path.join(scene_dir, "fine_dynamic_masks", "all"),
                    (".jpg", ".png"),
                    scene_name,
                    "dynamic masks",
                )
            )
            self.depth_flow_paths.append(
                self._collect_optional_view_paths(
                    os.path.join(scene_dir, "depth_flows_4"), (".npy",), scene_name, "depth flows"
                )
            )

    def _collect_view_paths(self, directory, extensions, scene_name, label):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Missing {label} directory for scene {scene_name}: {directory}")
        filenames = os.listdir(directory)
        view_paths = []
        for camera_id in self.input_cameras:
            suffixes = tuple(f"_{camera_id}{extension}" for extension in extensions)
            paths = sorted(
                os.path.join(directory, filename)
                for filename in filenames
                if filename.endswith(suffixes)
            )
            if not paths:
                raise FileNotFoundError(
                    f"No {label} found for scene {scene_name}, camera {camera_id} in {directory}."
                )
            view_paths.append(paths)
        self._validate_frame_alignment(view_paths, scene_name, label)
        return view_paths

    def _collect_optional_view_paths(self, directory, extensions, scene_name, label):
        return [] if not os.path.isdir(directory) else self._collect_view_paths(directory, extensions, scene_name, label)

    @staticmethod
    def _frame_id(path):
        return os.path.basename(path).split("_", maxsplit=1)[0]

    def _validate_frame_alignment(self, view_paths, scene_name, label):
        frame_ids = [[self._frame_id(path) for path in paths] for paths in view_paths]
        if any(camera_frame_ids != frame_ids[0] for camera_frame_ids in frame_ids[1:]):
            raise RuntimeError(
                f"Inconsistent {label} frame IDs across cameras {self.input_cameras} in scene {scene_name}."
            )

    def _flatten_paths(self, view_paths, indices):
        return [view_paths[view_index][frame_index] for frame_index in indices for view_index in range(self.views)]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        num_frames = len(image_paths[0])
        if self.mode == 1:
            start_idx = random.randint(0, max(1, num_frames - 21))
            indices = [start_idx] + [start_idx + offset for offset in sorted(random.sample(range(1, 20), self.sequence_length - 1))]
            target_indices = None
            intervals = [1 for _ in range(self.sequence_length - 1)]
        elif self.mode == 2:
            start_idx = self.start_idx
            indices = [start_idx + frame_index * self.frame_stride for frame_index in range(self.sequence_length)]
            target_indices = None
            intervals = [self.frame_stride for _ in range(self.sequence_length - 1)]
        elif self.mode == 3:
            start_idx = self.start_idx
            indices = [start_idx + frame_index * self.interval for frame_index in range(self.sequence_length)]
            target_indices = [start_idx + frame_index for frame_index in range(self.sequence_length * self.interval - (self.interval - 1))]
            intervals = self.interval
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        required_indices = indices if target_indices is None else target_indices
        if min(indices) < 0 or max(required_indices) >= num_frames:
            raise IndexError(
                f"Requested frames {min(indices)}..{max(required_indices)} for scene {self.scenes[idx]}, "
                f"but only 0..{num_frames - 1} are available."
            )

        sequence_paths = self._flatten_paths(image_paths, indices)
        images = load_and_preprocess_images(sequence_paths)
        timestamps = np.repeat(np.asarray(indices, dtype=np.float32) - start_idx, self.views)
        timestamps = timestamps / max(timestamps[-1], 1.0) * (self.sequence_length / 4)

        sky_mask_paths = self.sky_mask_paths[idx]
        if not sky_mask_paths:
            raise FileNotFoundError(f"Missing sky masks for scene {self.scenes[idx]}.")
        input_dict = {
            "images": images,
            "masks": load_and_preprocess_images(self._flatten_paths(sky_mask_paths, indices)),
            "image_paths": sequence_paths,
            "timestamps": timestamps,
            "interval": intervals,
        }
        scene_dir = os.path.join(self.image_dir, self.scenes[idx])
        input_dict["intrinsics"], input_dict["camera_to_worlds"] = load_waymo_calibrations(
            scene_dir, sequence_paths
        )

        dynamic_mask_paths = self.dynamic_mask_path[idx]
        selected_indices = target_indices if target_indices is not None else indices
        if dynamic_mask_paths:
            input_dict["dynamic_mask"] = load_and_preprocess_images(
                self._flatten_paths(dynamic_mask_paths, selected_indices)
            )

        if target_indices is not None:
            target_paths = self._flatten_paths(image_paths, target_indices)
            input_dict["targets"] = load_and_preprocess_images(target_paths)
            input_dict["target_masks"] = load_and_preprocess_images(self._flatten_paths(sky_mask_paths, target_indices))
            input_dict["target_intrinsics"], input_dict["target_camera_to_worlds"] = load_waymo_calibrations(
                scene_dir, target_paths
            )

        depth_paths = self.depth_flow_paths[idx]
        if depth_paths:
            input_dict["gt_depth"] = load_and_preprocess_flow(
                self._flatten_paths(depth_paths, selected_indices), None, None, images.shape[2], images.shape[3]
            )
        else:
            input_dict["gt_depth"] = torch.zeros(len(selected_indices) * self.views, images.shape[2], images.shape[3])
        return input_dict
