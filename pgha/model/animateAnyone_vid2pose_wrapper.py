import sys
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
ANIMATE_PATH = BASE_DIR / "external/animateanyone"

sys.path.insert(0, str(ANIMATE_PATH))

# external imports 
from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil

import json
from datetime import datetime
from tqdm import tqdm
import torch


def process_video(video_path, output_path, detector, force_rerun=False):

    if output_path.exists() and not force_rerun:
        return True, "already_exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fps = get_fps(str(video_path))
        frames = read_frames(str(video_path))

        if len(frames) == 0:
            return False, f"No frames could be read from {video_path}"

        kps_results = []
        failed_frames = 0

        for idx, frame_pil in enumerate(frames):
            try:
                result, score = detector(frame_pil)
                kps_results.append(result)
            except Exception as e:
                failed_frames += 1
                print(f"Warning: Frame {idx} failed in {video_path.name}: {e}")
                if idx > 0:
                    kps_results.append(kps_results[-1])
                else:
                    return False, f"First frame failed: {e}"

        if len(kps_results) == 0:
            return False, "No poses could be extracted from any frame"

        if failed_frames > 0:
            print(f"{failed_frames}/{len(frames)} frames had issues in {video_path.name}")

        save_videos_from_pil(kps_results, str(output_path), fps=fps)
        return True, f"success ({len(frames)} frames, {failed_frames} issues)"

    except Exception as e:
        return False, str(e)


def run_vid2pose(cfg):

    video_root = Path(cfg.paths.video_root)
    pose_root = Path(cfg.paths.pose_root)
    log_path = Path(cfg.paths.log_file)

    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    pose_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DWpose Extraction - PROCESSING ALL VIDEOS")
    print("=" * 70)
    print(f"Video Root: {video_root}")
    print(f"Pose Root: {pose_root}")
    print(f"Force re-run: {cfg.processing.force_rerun}")
    print(f"File pattern: {cfg.processing.file_pattern}")
    print("=" * 70)

    processed_videos = set()
    failed_videos = {}

    if cfg.processing.resume and log_path.exists():
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                processed_videos = set(log_data.get('processed', []))
                failed_videos = log_data.get('failed', {})
            print(f"Resuming from log: {len(processed_videos)} already processed")
        except:
            print("Could not load log file, starting fresh")

    print("\nLoading DWpose detector...")
    original_cwd = os.getcwd()
    try:
        os.chdir(str(ANIMATE_PATH))   # wholebody.py needs cwd = external/animateanyone
        detector = DWposeDetector()         
        detector = detector.to("cuda" if torch.cuda.is_available() else "cpu")
    finally:
        os.chdir(original_cwd)        # always restored before anything else runs

    try:
        if cfg.runtime.use_cuda_if_available and torch.cuda.is_available():
            print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA not available, using CPU (will be SLOW)")
    except Exception as e:
        raise RuntimeError(f"Failed to load detector: {e}")

    print("\nScanning for ALL videos...")

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv']
    all_videos = []

    for ext in video_extensions:
        all_videos.extend(video_root.rglob(f"*{ext}"))

    if cfg.processing.file_pattern != "*.mp4":
        all_videos.extend(video_root.rglob(cfg.processing.file_pattern))

    all_videos = sorted(set(all_videos))
    print(f"Found {len(all_videos)} TOTAL video files")

    if cfg.processing.resume and not cfg.processing.force_rerun:
        videos_to_process = [v for v in all_videos if str(v) not in processed_videos]
        print(f"After resume filtering: {len(videos_to_process)} videos to process")
    else:
        videos_to_process = all_videos

    results = {
        'success': [],
        'failed': {},
        'already_existed': []
    }

    start_time = datetime.now()
    pbar = tqdm(videos_to_process, desc="Processing videos", unit="video")

    for video_path in pbar:
        pbar.set_description(f"Processing {video_path.name[:30]}...")

        relative_path = video_path.relative_to(video_root)
        output_path = pose_root / relative_path

        success, message = process_video(
            video_path,
            output_path,
            detector,
            force_rerun=cfg.processing.force_rerun
        )

        if success:
            if message == "already_exists":
                results['already_existed'].append(str(video_path))
            else:
                results['success'].append(str(video_path))
        else:
            results['failed'][str(video_path)] = message

        with open(log_path, 'w') as f:
            json.dump({
                'processed': results['success'] + results['already_existed'],
                'failed': results['failed'],
                'total_found': len(all_videos),
                'timestamp': str(datetime.now())
            }, f, indent=2)

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Total videos found: {len(all_videos)}")
    print(f"Successfully processed: {len(results['success'])}")
    print(f"Already existed: {len(results['already_existed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Total time: {elapsed}")

    with open(log_path, 'w') as f:
        json.dump({
            'processed': results['success'],
            'already_existed': results['already_existed'],
            'failed': results['failed'],
            'total_found': len(all_videos),
            'timestamp': str(datetime.now()),
            'video_root': str(video_root),
            'pose_root': str(pose_root)
        }, f, indent=2)

    print(f"\nDetailed log saved to: {log_path}")
    print(f"\nPose videos saved in: {pose_root}")