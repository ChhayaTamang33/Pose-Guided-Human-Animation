import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pgha.pipeline.run_stage_1 import main as stage1_main
from pgha.pipeline.run_stage_2 import main as stage2_main


@hydra.main(config_path="../../configs", config_name="preprocessing")
def main(cfg: DictConfig):
    cfg.paths.input_videos_dir = to_absolute_path(cfg.paths.input_videos_dir)
    cfg.paths.csv_output_dir = to_absolute_path(cfg.paths.csv_output_dir)
    cfg.paths.final_csv_path = to_absolute_path(cfg.paths.final_csv_path)

    cfg.paths.input_dir = to_absolute_path(cfg.paths.input_dir)
    cfg.paths.csv_path = to_absolute_path(cfg.paths.csv_path)
    cfg.paths.output_dir = to_absolute_path(cfg.paths.output_dir)
    stage1_main(cfg)
    stage2_main(cfg)


if __name__ == "__main__":
    main()