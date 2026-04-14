import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from datetime import datetime

from pgha.model.mimicmotion_wrapper import run_mimicmotion, set_logger

@hydra.main(config_path="../../configs", config_name="inference_mimicMotion")
def main(cfg: DictConfig):

    cfg.paths.video_root = to_absolute_path(cfg.paths.video_root)
    cfg.paths.output_dir = to_absolute_path(cfg.paths.output_dir)
    cfg.paths.log_file = to_absolute_path(cfg.paths.log_file)

    # Logger setup
    log_file = cfg.paths.log_file or f"{cfg.paths.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    set_logger(log_file)

    run_mimicmotion(cfg)


if __name__ == "__main__":
    main()