import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pgha.model.animateAnyone_vid2pose_wrapper import run_vid2pose


@hydra.main(config_path="../../configs", config_name="vid2pose_animateAnyone")
def main(cfg: DictConfig):

    cfg.paths.video_root = to_absolute_path(cfg.paths.video_root)
    cfg.paths.pose_root = to_absolute_path(cfg.paths.pose_root)
    cfg.paths.log_file = to_absolute_path(cfg.paths.log_file)

    run_vid2pose(cfg)


if __name__ == "__main__":
    main()