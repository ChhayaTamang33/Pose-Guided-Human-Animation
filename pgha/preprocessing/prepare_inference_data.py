# stage 2 of the preprocessing: prepare_inference_data.py
import os
import csv
from pathlib import Path
import subprocess
import cv2
from imageio import get_reader, get_writer
import mediapipe as mp


def run_stage2(cfg):

    if not os.path.exists(cfg.paths.csv_path):
        raise FileNotFoundError(f"FINAL CSV not found at {cfg.paths.csv_path}. Run Stage 1 first.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=cfg.mediapipe.model_complexity,
        min_detection_confidence=cfg.mediapipe.min_detection_confidence,
        min_tracking_confidence=cfg.mediapipe.min_tracking_confidence
    )

    with open(cfg.paths.csv_path) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        video_name = row["video_name"]
        start_f = int(row["start_frame"])
        end_f   = int(row["end_frame"])

        video_path = os.path.join(cfg.paths.input_dir, video_name)

        if not os.path.exists(video_path):
            print(f"[SKIP] Missing video: {video_name}")
            continue

        stem = Path(video_name).stem
        video_out_dir = os.path.join(cfg.paths.output_dir, stem)
        tmp_dir = os.path.join(video_out_dir, "tmp")

        os.makedirs(video_out_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)

        reader = get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 30)
        reader.close()

        start_time = start_f / fps
        duration = (end_f - start_f) / fps

        tmp_clip = os.path.join(tmp_dir, f"seg_{start_f}_{end_f}.mp4")

        # ffmpeg
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            tmp_clip
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        out_path = os.path.join(video_out_dir, f"{stem}_{start_f}_{end_f}.mp4")

        reader_clip = get_reader(tmp_clip)
        fps2 = reader_clip.get_meta_data().get("fps", 30)
        writer = get_writer(out_path, fps=fps2)

        prev_x, prev_y = None, None

        for frame in reader_clip:
            rgb = frame[:, :, :3]
            H, W, _ = rgb.shape
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                lx = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                rx = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                ly = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                ry = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                x_now = int(((lx + rx) / 2) * W)
                y_now = int(((ly + ry) / 2) * H)
            else:
                x_now, y_now = (W // 2, H // 2) if prev_x is None else (prev_x, prev_y)

            # EMA smoothing
            if prev_x is None:
                x_s, y_s = x_now, y_now
            else:
                x_s = int(cfg.processing.smoothing_alpha * prev_x + (1 - cfg.processing.smoothing_alpha) * x_now)
                y_s = int(cfg.processing.smoothing_alpha * prev_y + (1 - cfg.processing.smoothing_alpha) * y_now)

            prev_x, prev_y = x_s, y_s

            # Crop params
            crop_size = int(cfg.processing.crop_body_ratio * H)
            half = crop_size // 2
            pad = int(cfg.processing.padding_ratio * crop_size)
            headroom = int(cfg.processing.headroom_ratio * crop_size)

            x1 = max(0, x_s - half - pad)
            x2 = min(W, x_s + half + pad)
            y1 = max(0, y_s - half - pad - headroom)
            y2 = min(H, y_s + half + pad)

            # Ensure minimum size (RESTORED)
            crop_w_now, crop_h_now = x2 - x1, y2 - y1
            min_size = int(cfg.processing.min_semantic_scale_ratio * min(H, W))

            if crop_w_now < min_size:
                delta = (min_size - crop_w_now) // 2
                x1, x2 = max(0, x1 - delta), min(W, x2 + delta)

            if crop_h_now < min_size:
                delta = (min_size - crop_h_now) // 2
                y1, y2 = max(0, y1 - delta), min(H, y2 + delta)

            # Semantic check (RESTORED EXACTLY)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def inside(px, py):
                    return x1 <= px <= x2 and y1 <= py <= y2

                nx = int(lm[mp_pose.PoseLandmark.NOSE].x * W)
                ny = int(lm[mp_pose.PoseLandmark.NOSE].y * H)

                lwx = int(lm[mp_pose.PoseLandmark.LEFT_WRIST].x * W)
                lwy = int(lm[mp_pose.PoseLandmark.LEFT_WRIST].y * H)

                rwx = int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * W)
                rwy = int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * H)

                if not (inside(nx, ny) and inside(lwx, lwy) and inside(rwx, rwy)):
                    x1 = max(0, x_now - half - pad)
                    x2 = min(W, x_now + half + pad)
                    y1 = max(0, y_now - half - pad - headroom)
                    y2 = min(H, y_now + half + pad)

            # Crop + resize
            crop = rgb[y1:y2, x1:x2]
            crop_resized = cv2.resize(
                crop,
                (cfg.processing.output_size, cfg.processing.output_size),
                interpolation=cv2.INTER_LINEAR
            )

            writer.append_data(crop_resized)

        reader_clip.close()
        writer.close()
        os.remove(tmp_clip)

        print(f"[DONE] {out_path}")

    pose.close()
    print("Stage 2 complete.")