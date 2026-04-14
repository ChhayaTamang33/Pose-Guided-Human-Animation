import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ANIMATE_PATH = BASE_DIR / "external/animateanyone"

sys.path.insert(0, str(ANIMATE_PATH))

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames

import json
from datetime import datetime
import torch
import torchvision
from tqdm import tqdm


def load_pipeline(cfg):

    dtype = torch.float16 if cfg.inference.dtype == "float16" else torch.float32
    device = cfg.runtime.device

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(cfg.model.vae_path).to(device="cpu", dtype=dtype)

    print("Loading reference UNet...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.model.pretrained_base, subfolder="unet"
    ).to(device, dtype=dtype)

    print("Loading denoising UNet...")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.model.pretrained_base,
        cfg.model.motion_module,
        subfolder="unet",
        unet_additional_kwargs=cfg.unet_additional_kwargs,
    ).to(device, dtype=dtype)

    print("Loading pose guider...")
    pose_guider = PoseGuider(
        320, block_out_channels=(16, 32, 96, 256)
    ).to(device, dtype=dtype)

    print("Loading image encoder...")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.model.image_encoder
    ).to(device="cpu", dtype=dtype)

    scheduler = DDIMScheduler(**cfg.scheduler)

    denoising_unet.load_state_dict(torch.load(cfg.model.denoising_unet, map_location = "cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(cfg.model.reference_unet, map_location = "cpu"))
    pose_guider.load_state_dict(torch.load(cfg.model.pose_guider, map_location = "cpu"))

    pipe = Pose2VideoPipeline(
        vae,
        image_enc,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler,
    ).to(device, dtype=dtype)

    pipe.enable_vae_slicing()

    return pipe


def run_pose2vid(cfg):

    video_dir = Path(cfg.paths.video_dir)
    pose_dir  = Path(cfg.paths.pose_dir)
    output_dir = Path(cfg.paths.output_dir)
    log_path  = Path(cfg.paths.log_path)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video dir does not exist: {video_dir}")
    if not pose_dir.exists():
        raise FileNotFoundError(f"Pose dir does not exist: {pose_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pose2Vid - AnimateAnyone INFERENCE")
    print("=" * 70)
    print(f"Video Dir:  {video_dir}")
    print(f"Pose Dir:   {pose_dir}")
    print(f"Output Dir: {output_dir}")
    print("=" * 70)

    # load pipeline
    print("\nLoading pipeline...")
    try:
        pipe = load_pipeline(cfg)
        print("Pipeline loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline: {e}")

    # find all pose videos
    pose_videos = sorted(pose_dir.rglob("*.mp4"))
    print(f"\nFound {len(pose_videos)} pose videos to process.")

    results = {
        "success": [],
        "skipped": [],
        "failed": {}
    }

    start_time = datetime.now()
    pbar = tqdm(pose_videos, desc="Running inference", unit="video")

    for pose_path in pbar:
        pbar.set_description(f"Processing {pose_path.name[:40]}...")

        rel = pose_path.relative_to(pose_dir)
        video_path = video_dir / rel
        out_path = (output_dir / rel).with_suffix(".mp4")
        ref_image_path = output_dir / "ref_images" / rel.with_suffix(".png")
        
        # skip if output already exists
        # if out_path.exists():
        #     results["skipped"].append(str(rel))
        #     _write_log(log_path, results, len(pose_videos), start_time,
        #                video_dir, pose_dir, output_dir)
        #     continue

        if not video_path.exists():
            msg = f"Matching source video not found: {video_path}"
            print(f"\nWarning: {msg}")
            results["failed"][str(rel)] = msg
            _write_log(log_path, results, len(pose_videos), start_time,
                       video_dir, pose_dir, output_dir)
            continue

        try:
            pose_frames    = read_frames(str(pose_path))
            original_frames = read_frames(str(video_path))

            if len(pose_frames) == 0:
                raise ValueError("No frames in pose video")
            if len(original_frames) == 0:
                raise ValueError("No frames in source video")

            ref_image = original_frames[0].convert("RGB")
            fps = get_fps(str(video_path))

            chunks = []
            
            with torch.no_grad():
                for i in range(0, len(pose_frames), cfg.inference.chunk_size):
                    chunk = pose_frames[i:i + cfg.inference.chunk_size]

                    out = pipe(
                        ref_image,
                        chunk,
                        cfg.inference.width,
                        cfg.inference.height,
                        len(chunk),
                        cfg.inference.steps,
                        cfg.inference.cfg,
                        generator=torch.manual_seed(cfg.inference.seed),
                    ).videos

                    chunks.append(out[0])

            full = torch.cat(chunks, dim=1)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchvision.io.write_video(
                str(out_path),
                (full.permute(1, 2, 3, 0).cpu() * 255).byte(),
                fps=fps,
            )

            results["success"].append(str(rel))
            print(f"\nSaved: {out_path}")
            
            ref_image.save(ref_image_path)

        except Exception as e:
            results["failed"][str(rel)] = str(e)
            print(f"\nFailed: {rel} — {e}")

        # write log after every video
        _write_log(log_path, results, len(pose_videos), start_time,
                   video_dir, pose_dir, output_dir)

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE!")
    print("=" * 70)
    print(f"Total pose videos:      {len(pose_videos)}")
    print(f"Successfully generated: {len(results['success'])}")
    print(f"Skipped (exists):       {len(results['skipped'])}")
    print(f"Failed:                 {len(results['failed'])}")
    print(f"Total time:             {elapsed}")
    print(f"Log saved to:           {log_path}")
    print(f"Outputs saved in:       {output_dir}")


def _write_log(log_path, results, total, start_time, video_dir, pose_dir, output_dir):
    """Write progress log to disk after every video."""
    with open(log_path, "w") as f:
        json.dump({
            "success":    results["success"],
            "skipped":    results["skipped"],
            "failed":     results["failed"],
            "total_found": total,
            "timestamp":  str(datetime.now()),
            "elapsed":    str(datetime.now() - start_time),
            "video_dir":  str(video_dir),
            "pose_dir":   str(pose_dir),
            "output_dir": str(output_dir),
        }, f, indent=2)