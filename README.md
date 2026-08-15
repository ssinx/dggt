<div align="center">

# DGGT ：FEEDFORWARD 4D RECONSTRUCTION OF DYNAMIC DRIVING SCENES USING UNPOSED IMAGES
<a href="https://arxiv.org/abs/2512.03004" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv">
</a>
<a href="https://xiaomi-research.github.io/dggt/" target="_blank">
  <img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page">
</a>

**Xiaoxue Chen**¹,²*, **Ziyi Xiong**¹,²*, **Yuantao Chen**¹, **Gen Li**¹, **Nan Wang**¹,  
**Hongcheng Luo**², **Long Chen**², **Haiyang Sun**²†, **Bing Wang**², **Guang Chen**², **Hangjun Ye**²,✉,  
**Hongyang Li**³, **Ya-Qin Zhang**¹, **Hao Zhao**¹,⁴,✉

¹ AIR, Tsinghua University  
² Xiaomi EV  
³ The University of Hong Kong  
⁴ Beijing Academy of Artificial Intelligence  

\* These authors contributed equally
† Project leader

</div>



## Abstract

Our method introduces a fully pose-free feedforward framework **DGGT** for reconstructing dynamic driving scenes directly from unposed RGB images. The model predicts camera poses, 3D Gaussian maps, dynamic motion in a single pass — without per-scene optimization or camera calibration.

<details><summary>CLICK for the full abstract</summary>

> Autonomous driving needs fast, scalable 4D reconstruction and re-simulation for training and evaluation, yet most methods for dynamic driving scenes still rely on per-scene optimization, known camera calibration, or short frame windows, making them slow and impractical. We revisit this problem from a feedforward perspective and introduce **Driving Gaussian Grounded Transformer (DGGT)**, a unified framework for pose-free dynamic scene reconstruction. We note that the existing formulations, treating camera pose as a required input, limit flexibility and scalability. Instead, we reformulate pose as an output of the model, enabling reconstruction directly from sparse, unposed images and supporting an arbitrary number of views for long sequences. Our approach jointly predicts per-frame 3D Gaussian maps and camera parameters, disentangles dynamics with a lightweight dynamic head, and preserves temporal consistency with a lifespan head that modulates visibility over time. A diffusion-based rendering refinement further reduces motion/interpolation artifacts and improves novel-view quality under sparse inputs. The result is a single-pass, pose-free algorithm that achieves state-of-the-art performance and speed. Trained and evaluated on large-scale driving benchmarks (Waymo, nuScenes, Argoverse2), our method outperforms prior work both when trained on each dataset and in zero-shot transfer across datasets, and it scales well as the number of input frames increases.
</details>

## 🚧 Todo

- [√] Release pre-trained checkpoints on  Waymo, NuScenes and Argoverse2
- [√] Release the inference code of our model to facilitate further research and reproducibility.
- [√] Release the training code


### 🚗 Dataset Support
This codebase provides support for Waymo Open Dataset, Nuscenes and Argoverse2. We provide instructions and scripts on how to download and preprocess these datasets:
| Dataset | Instruction |
|---------|-------------|
| Waymo | [Data Process Instruction](datasets/Waymo.md) |
| Argoverse2 | [Data Process Instruction](datasets/ArgoVerse2.md) |
| NuScenes | [Data Process Instruction](datasets/NuScenes.md) |

## Installation
### Installing dependencies

1. Create conda environment
```bash
conda create -n dggt python=3.10
conda activate dggt

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
```

If this environment was created before the dependency pins were added, reinstall
the compatible inference dependencies before running inference:

```bash
pip install --upgrade --force-reinstall \
  "numpy<2" \
  "huggingface_hub==0.26.5" \
  "transformers==4.46.3" \
  "diffusers==0.30.3"
pip check
```

2. Compile pointops2

```bash
cd third_party/pointops2
python setup.py install
cd ../..
```


### Downloading checkpoints
Download our pretrained inference model (trained on Waymo Open Dataset, 1 views) checkpoint [here](https://huggingface.co/xiaomi-research/dggt/resolve/main/model_latest_waymo.pt?download=true) to `pretrained/model_latest_waymo.pth`.

Download our pretrained diffusion model checkpoint [here](https://huggingface.co/xiaomi-research/dggt/resolve/main/model_difix.pkl?download=true) to `pretrained/diffusion_model.pth`.

Download TAPIP3D model checkpoint [here](https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth) to `pretrained/tracking_model.pth`.

Other checkpoints will be coming soon.
## Usage
### Quick start
You can test existing models on the Waymo Open dataset.
```bash
python inference.py \
    --image_dir /path/to/images \
    --scene_names 3 5 7 \
    --input_camera 0 \
    --input_views 1 \
    --intervals 2 \
    --sequence_length 4 \
    --input_frame_stride 1 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/checkpoint.pth \
    --output_path /path/to/output \
    -images \
    -depth \
    -diffusion \
    -metrics 
```
    --image_dir <path>: Specifies the directory containing the input images (required).
    --scene_names <names>: A string representing the scene names to process, supporting formats like 3 5 7 or "(3,7)" (required).
    --mode <mode>: Specifies the processing mode, with acceptable values of 1--train, 2--reconstruction, or 3--interplation (required).
    --ckpt_path <path>: The path to the pre-trained model weights file (required).
    --output_path <path>: The directory where the output results will be saved (required).
    --input_camera <ids...>: One or more ordered Waymo camera IDs, for example `--input_camera 1 3 4` (required for `--image_dir`).
    --input_views <views>: Deprecated compatibility flag; when supplied it must match the number of `--input_camera` IDs.
    --intervals <interval>: The interval of interpolation frames when performing frame interpolation (mode=3), defaulting to 2 (optional).
    --sequence_length <length>: Defines the number of input frames to consider for each inference, defaulting to 4 (optional).
    --input_frame_stride <stride>: In Waymo reconstruction mode (`--mode 2`), sample each selected camera every `<stride>` frames; default is 1 for consecutive frames.
    --start_idx <index>: Indicates the starting index of the frames to process, defaulting to 0 (optional).
    -images: A flag that, when specified, enables the output of rendered images for each frame (optional).
    -depth: A flag that, when specified, enables the output of depth maps in .npy format for each frame (optional).
    -metrics: A flag that, when specified, enables the output of evaluation metrics (PSNR, SSIM, LPIPS) after processing (optional).
    -diffusion: Whether to use diffusion model to optimize the rendered images (time-consuming) (optional).

### Inference on an ordinary image directory

For an unlabelled image sequence, use `--plain_image_dir` instead of the processed-dataset arguments. The directory is searched recursively for `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, and `webp` files. Files are naturally sorted by their relative paths and split into non-overlapping groups of `--sequence_length` images. Put temporally adjacent views of the same scene in the same directory; unrelated images cannot form a meaningful shared 3D reconstruction.

```bash
python inference.py \
    --plain_image_dir /path/to/images \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path pretrained/model_latest_waymo.pth \
    --output_path /path/to/output \
    -images \
    -depth
```

This mode predicts poses, depth, and Gaussian attributes from the images themselves. It uses all-zero sky/dynamic masks because no annotations are available, so it does not generate `comparison.mp4` or support `-metrics`. Rendered frames and `rendered_video.mp4` are written to numbered subdirectories under `--output_path`; depth maps are saved when `-depth` is supplied.

### VLM point localization on bird's-eye renders

When `--render_birds_eye` is enabled, inference writes both `source_camera_rendered_video.mp4` and `birds_eye_rendered_video.mp4`. Add `--vlm_prompt` together with `--assets_manifest` to place manifest assets from the last-frame correspondences. Qwen first selects currently empty, visible ground-contact locations suitable for inserting the requested new assets in the last front-view frame. It is explicitly instructed not to select points on existing vehicles, people, or obstacles; existing objects are context only. Each front-view placement point defines an epipolar line in the corresponding rendered bird's-eye frame. The bird's-eye VLM request receives both the front view with that point marked in red and the bird's-eye view with the red epipolar line, together with the front point's description, confidence, and pixel coordinate. Qwen then selects the same vacant physical location only on that line. The two camera rays are triangulated into a DGGT world coordinate, which replaces the corresponding manifest asset's translation in every rendered frame. Assets and localized points are paired in manifest order.

The results for each scene are written to `vlm_point_detections.json`. It records the selected last-frame index, both 2D selections, the bird's-eye epipolar line, the triangulated world coordinate, and the closest-ray error. Last-frame visualizations are saved under `vlm_point_visualizations/`.

By default, each localized asset is also snapped vertically to the last frame's local static ground surface. Low-opacity scene and asset Gaussians are ignored, a robust local ground plane is fitted, and the asset's opaque lower contour is moved into contact with that plane. Horizontal placement is unchanged. The manifest accepts these optional per-asset settings:

```json
{
  "ground_snap": true,
  "contact_opacity_threshold": 0.05,
  "contact_quantile": 0.99,
  "ground_opacity_threshold": 0.05,
  "ground_quantile": 0.75,
  "ground_search_radius": 1.5,
  "ground_vertical_band": 1.5,
  "ground_min_points": 32,
  "ground_clearance_m": 0.0
}
```

Distances in `ground_search_radius`, `ground_vertical_band`, and `ground_clearance_m` are meters. Set `ground_snap` to `false` to keep the raw triangulated height. Ground fitting status, point counts, residual, original position, vertical adjustment, and final position are recorded under each correspondence's `ground_snap` field in `vlm_point_detections.json`.

After ground snapping, VLM localization also refines each inserted asset's horizontal orientation and uniform physical scale by default. Candidate frames are selected without another model call: opaque asset Gaussians are projected into every source frame, scored by projected size, in-frame ratio, and crop completeness, then up to three geometrically diverse views are retained. The selected front views and one bird's-eye view are sent in a single VLM request per asset. Asset-local axes are not used by frame selection or by the VLM protocol. Qwen returns a bounded yaw correction about the reconstructed scene's vertical axis and a bounded scale factor relative to the manifest `scale`. The rotated and resized asset is snapped to the ground again before the final sequence is rendered.

Optional per-asset manifest settings are:

```json
{
  "vlm_orientation_refine": true,
  "orientation_prompt": "Place the vehicle parallel to its lane and facing the traffic direction.",
  "orientation_top_k": 3,
  "orientation_projection_max_points": 3000,
  "max_yaw_delta_deg": 90,
  "min_scale_factor": 0.3,
  "max_scale_factor": 4.0,
  "vlm_refinement_max_rounds": 5
}
```

Set `vlm_orientation_refine` to `false` to retain both the manifest rotation and scale. Frame selection uses at most `orientation_projection_max_points` asset Gaussians and does not call VLM. Otherwise, each round renders the current asset in the same selected views and asks Qwen whether both orientation and physical scale are satisfactory. A satisfactory response stops immediately; otherwise the returned incremental yaw and scale are applied, the asset is snapped to the ground again, and another round begins. `vlm_refinement_max_rounds` defaults to 5 and must be in `[1, 5]`. Cumulative scale remains within the manifest's global `min_scale_factor` and `max_scale_factor` bounds. Debug artifacts are written to `vlm_orientation_refinement/<asset_id>/round_XX/`, while the asset-level `result.json` contains all rounds and the final cumulative transformation. The same data is embedded under `orientation_refinement` in `vlm_point_detections.json`. Final corrected frames remain under `source_camera/` and `birds_eye/`.

Install the optional client dependency and set an API key first:

```bash
pip install -r requirements_annotation.txt
export DASHSCOPE_API_KEY='...'
```

```bash
python inference.py \
    --image_dir /path/to/images \
    --scene_names 3 \
    --input_camera 0 \
    --sequence_length 4 \
    --mode 2 \
    --ckpt_path /path/to/checkpoint.pth \
    --output_path /path/to/output \
    --render_birds_eye \
    --assets_manifest assets/scene_assets.json \
    --vlm_prompt 'Find the center point of every orange traffic cone.'
```

By default, localization uses `qwen3.8-max`, the supplied Qwen OpenAI-compatible endpoint, and `enable_thinking=true`. Use `--vlm_prompt_file` for a longer UTF-8 prompt, `--vlm_model` or `--vlm_base_url` to select the endpoint/model, and `--no-vlm-enable-thinking` to disable reasoning mode.

### Train
<!-- You need to further process the training data according to the .md file in the data processing section.  -->
And download vggt pretrained model [here](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) to `pretrained/model.pt`.

You can train the model on the Waymo Open dataset.
```bash
torchrun --nproc_per_node=1 --master_port=0000 train.py \
  --image_dir /path/to/processed_waymo \
  --log_dir logs/xxx \
  --ckpt_path /path/to/pretrained_checkpoint.pth \
  --sky_model cubemap \
  --scene_names_file /path/to/processed_scene_names.txt
```

    --image_dir <path>: Path to the Waymo dataset directory containing processed training data (required).
    --log_dir <path>: Directory where training logs, checkpoints, and visualizations will be saved (required).
    --ckpt_path <path>: Path to the pretrained model checkpoint to initialize weights (required).
    --sequence_length <length>: Frames per sample in processed-data mode, defaulting to 4.
    --raw_sequence_lengths <lengths...>: Candidate contiguous interval lengths in raw Parquet mode, defaulting to `4 8 12`. Each epoch samples one interval per scene.
    --sky_model cubemap: Train the new world-aligned cubemap sky head; this training entry point rejects the legacy Gaussian mode.
    --scene_names / --scene_names_file: Select processed Waymo scene directories; the file contains one directory name per line.

The cubemap sky head follows Instant NuRec's resolution split: ray-conditioned queries attend to scene tokens on a patch grid, then a four-scale DPT reassemble/fusion decoder produces full-resolution cubemap faces. Decoder activation checkpointing is enabled during training to limit memory use. The training entry point freezes the complete checkpoint and trains only `sky_head.*`. A checkpoint with a missing or architecturally incompatible `sky_head.*` restarts only the sky head; all frozen DGGT weights must still match. Standard Waymo inference must also pass `--sky_model cubemap`; use `--sky_model gaussian` with legacy checkpoints.

To avoid materializing a processed Waymo copy, train directly from the v2 modular
Parquet split. With neither `--scene_names` nor `--scene_names_file`, every
segment in the split is used. Those options are only for debugging or subset
training; numeric names are zero-based indices into the sorted segment filenames,
and full segment names are also accepted. Each epoch samples one randomly
positioned contiguous interval from every scene using `camera_image`,
`camera_calibration`, and `vehicle_pose` without writing decoded data to disk.
Waymo supplies camera segmentation for only about 20 of roughly 200 frames per
segment. In raw mode, every decoded image is instead sent through the configured
SegFormer-B5 process to produce a Cityscapes sky mask. Newly generated masks are
kept in the small in-memory segment cache and also atomically written as
uncompressed binary `uint8` NumPy arrays under `--sky_mask_cache_dir` (default:
`/scratch/junyizh3/waymo_sky_masks`). Later epochs and restarted jobs on the
same compute node load those files and skip SegFormer. On this cluster,
`/scratch` is node-local NVMe storage, so the same path on a different node does
not contain the same files. At original Waymo resolution, caching every frame
can require roughly 365 GiB.

```bash
torchrun --nproc_per_node=1 --master_port=29500 train.py \
  --raw_waymo_dir /data/datasets/waymo/waymo-open-dataset-v2.0.1/training \
  --sky_mask_cache_dir /scratch/junyizh3/waymo_sky_masks \
  --image_dir '' \
  --ckpt_path pretrained/model_latest_waymo.pt \
  --log_dir logs/sky_only \
  --sky_model cubemap \
  --raw_sequence_lengths 4 8 12 --batch_size 1 --max_epoch 1000 \
  --save_image 50 \
  --save_ckpt 50
```

`--save_ckpt` and `--save_image` are measured in optimizer steps, not epochs.
Checkpoints are atomically written to the two rolling files
`LOG_DIR/ckpt/model_latest.pt` and `LOG_DIR/ckpt/model_best.pt`, while image
filenames include the iteration and are never overwritten. The final state is
always written to `model_latest.pt` when training exits normally.

TensorBoard scalars are written every optimizer step under
`LOG_DIR/tensorboard`. Start a viewer with
`tensorboard --logdir LOG_DIR/tensorboard --port 6006`.

To reuse the node-local mask cache when a pending Slurm job is assigned to a
different node, run the cache listener separately from the training job:

```bash
nohup ./sync_sky_masks_when_allocated.sh JOB_ID babel-s5-32 \
  > sync_sky_masks_JOB_ID.log 2>&1 &
```

The listener waits until Slurm reports the target node, then uses `rsync` to
copy only masks that are not already present. If the job is assigned to the
source node, no copy is performed. The synchronization runs concurrently with
training; any mask that has not arrived yet is generated locally on demand.


### Zero-shot and trained experiment​s

Quantitative Comparison under Trained and Zero-Shot Settings on nuScenes and Argoverse2 datasets. 

You can evaluate the model in two complementary settings to demonstrate both generalization and adaptability:

#### Zero-shot (Generalization)
You can use the model trained on Waymo to perform inference directly on the Argoverse2 or nuScenes datasets — without any retraining or pose calibration.

This setting highlights the model’s strong cross-dataset generalization and robustness to unseen driving domains.

Argoverse2/Nuscenes
```bash
python inference.py \
    --image_dir /path/to/argoverse_or_nuscenes_images \
    --scene_names 3 5 7 \
    --input_views 1 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/waymo_checkpoint.pth \
    --output_path /path/to/output \
    -images \
    -depth \
    -metrics \
```

#### Trained (Adaptability / Upper-bound Performance)
You can also train the model on the target dataset (e.g., Argoverse2) and evaluate it on the same domain.

This setting measures the model’s in-domain adaptability, showing its capacity to achieve state-of-the-art reconstruction quality when optimized for the target environment.

Argoverse2/Nuscenes
```bash
python inference.py \
    --image_dir /path/to/argoverse_or_nuscenes_images \
    --scene_names 3 5 7 \
    --input_views 1 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path /path/to/argoverse_or_nuscenes_checkpoint.pth \
    --output_path /path/to/output \
```

Together, these two experiments verify that our model not only generalizes well across unseen scenes, but also scales effectively to achieve top performance when fine-tuned on new domains.
## Citation
If you find this project useful, please consider citing:

```
@article{chenfeedforward,
  title={Feedforward 4D Reconstruction for Dynamic Driving Scenes using Unposed Images},
  author={Chen, Xiaoxue and Xiong, Ziyi and Chen, Yuantao and Li, Gen and Wang, Nan and Luo, Hongcheng and Chen, Long and Sun, Haiyang and WANG, BING and Chen, Guang and others}
}
```

## License
This project is licensed under the Apache License 2.0.
Some files in this repository are derived from VGGT (facebookresearch/vggt) and are licensed under the VGGT upstream license. See NOTICE for details.
