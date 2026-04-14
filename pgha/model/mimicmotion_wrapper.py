import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_PATH = BASE_DIR / "external/mimicmotion"

sys.path.insert(0, str(MIMIC_PATH))

from datetime import datetime
import logging
import numpy as np
import torch
import imageio
from PIL import Image

from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image

from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose


logger = logging.getLogger(__name__)

def preprocess(video_path, image_path, resolution=576, sample_stride=2):

    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)

    image_pixels = resize(image_pixels, [resolution, resolution], antialias=None)
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()

    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)

    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

    return (
        torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
        torch.from_numpy(image_pixels) / 127.5 - 1
    )

@torch.no_grad()
def run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config):

    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]

    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)

    frames = pipeline(
        image_pixels,
        image_pose=pose_pixels,
        num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames,
        tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=7,
        noise_aug_strength=task_config.noise_aug_strength,
        num_inference_steps=task_config.num_inference_steps,
        generator=generator,
        min_guidance_scale=task_config.guidance_scale,
        max_guidance_scale=task_config.guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device
    ).frames.cpu()

    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


def find_all_videos(root_dir):
    return sorted(Path(root_dir).rglob("*.mp4"))


def extract_first_frame(video_path, ref_dir):

    reader = imageio.get_reader(str(video_path))
    frame = reader.get_data(0)
    reader.close()

    img = Image.fromarray(frame)

    ref_path = ref_dir / (video_path.stem + "_ref.png")
    img.save(ref_path)

    return ref_path


def run_mimicmotion(cfg):

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    if cfg.system.use_float16:
        torch.set_default_dtype(torch.float16)

    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = create_pipeline(cfg, device)

    video_paths = find_all_videos(cfg.paths.video_root)

    for video_path in video_paths:
        try:
            ref_img = extract_first_frame(video_path, Path(cfg.paths.output_dir))

            pose_pixels, image_pixels = preprocess(
                str(video_path),
                str(ref_img),
                resolution=cfg.inference.resolution,
                sample_stride=cfg.inference.sample_stride,
            )

            frames = run_pipeline(
                pipeline,
                image_pixels,
                pose_pixels,
                device,
                cfg.inference,
            )

            out_path = Path(cfg.paths.output_dir) / video_path.relative_to(cfg.paths.video_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_to_mp4(frames, str(out_path), fps=cfg.inference.fps)

        except Exception as e:
            logger.error(f"Failed {video_path}: {e}")


def set_logger(log_file=None, log_level=logging.INFO):

    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)

    logger.addHandler(log_handler)