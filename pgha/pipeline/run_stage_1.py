import os
import glob
import hydra
from omegaconf import DictConfig
import mediapipe as mp
from hydra.utils import to_absolute_path

from pgha.preprocessing.filter_valid_segments import process_video_stage1, merge_csvs

@hydra.main(config_path="../../configs", config_name="preprocessing_stage_1")
def main(cfg: DictConfig):

    cfg.paths.input_videos_dir = to_absolute_path(cfg.paths.input_videos_dir)
    cfg.paths.csv_output_dir = to_absolute_path(cfg.paths.csv_output_dir)
    cfg.paths.final_csv_path = to_absolute_path(cfg.paths.final_csv_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=cfg.mediapipe.model_complexity,
        min_detection_confidence=cfg.mediapipe.min_detection_confidence,
        min_tracking_confidence=cfg.mediapipe.min_tracking_confidence
    )

    videos = glob.glob(os.path.join(cfg.paths.input_videos_dir, "*.mp4"))

    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        csv_path = os.path.join(cfg.paths.csv_output_dir, f"{name}.csv")
        process_video_stage1(v, csv_path, cfg, pose)

    pose.close()

    merge_csvs(
        cfg.paths.csv_output_dir,
        cfg.paths.final_csv_path,
        cfg.filters.min_frame
    )


if __name__ == "__main__":
    main()