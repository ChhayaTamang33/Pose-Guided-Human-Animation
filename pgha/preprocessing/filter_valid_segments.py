import os
import csv
import glob
from imageio import get_reader
import mediapipe as mp


def process_video_stage1(video_path, csv_path, cfg, pose):
    segments = []
    current_start = 0
    prev_kept = None

    reader = get_reader(video_path)
    video_name = os.path.basename(video_path)

    kept_frames = 0

    try:
        for idx, frame in enumerate(reader):
            rgb = frame[:, :, :3]
            results = pose.process(rgb)
            kept = 0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                face_ok = (
                    lm[mp.solutions.pose.PoseLandmark.NOSE].visibility > 0.5 and
                    lm[mp.solutions.pose.PoseLandmark.LEFT_EYE].visibility > 0.5 and
                    lm[mp.solutions.pose.PoseLandmark.RIGHT_EYE].visibility > 0.5
                )

                hands_ok = (
                    lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].visibility > 0.5 and
                    lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5
                )

                if face_ok and hands_ok:
                    shoulder_y = (lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y +
                                  lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                    hip_y = (lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y +
                             lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) / 2

                    body_ratio = abs(hip_y - shoulder_y)

                    if cfg.filters.min_body_height_ratio < body_ratio < cfg.filters.max_body_height_ratio:
                        kept = 1

            # Segment logic
            if prev_kept is None:
                prev_kept = kept
                current_start = idx
            elif kept != prev_kept:
                segments.append({
                    "video_name": video_name,
                    "start_frame": current_start,
                    "end_frame": idx - 1,
                    "kept": prev_kept
                })
                current_start = idx
                prev_kept = kept

            if kept:
                kept_frames += 1

        segments.append({
            "video_name": video_name,
            "start_frame": current_start,
            "end_frame": idx,
            "kept": prev_kept
        })

    finally:
        reader.close()

    filtered = [
        s for s in segments
        if s["kept"] == 1 and
        (s["end_frame"] - s["start_frame"] + 1) >= cfg.filters.min_frame_segment
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "start_frame", "end_frame", "kept"])
        writer.writeheader()
        writer.writerows(filtered)

    return len(filtered)


def merge_csvs(csv_dir, output_csv, min_frames):
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    final_rows = []

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = int(row["start_frame"])
                end = int(row["end_frame"])
                num_frames = end - start + 1

                if int(row["kept"]) != 1 or num_frames < min_frames:
                    continue

                final_rows.append({
                    "video_name": row["video_name"],
                    "start_frame": start,
                    "end_frame": end,
                    "num_frames": num_frames
                })

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "start_frame", "end_frame", "num_frames"])
        for r in final_rows:
            writer.writerow(r.values())

    return final_rows