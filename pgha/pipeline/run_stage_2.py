import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pgha.preprocessing.prepare_inference_data import run_stage2


@hydra.main(config_path="../../configs", config_name="preprocessing_stage_2")
def main(cfg: DictConfig):
    cfg.paths.input_dir = to_absolute_path(cfg.paths.input_dir)
    cfg.paths.csv_path = to_absolute_path(cfg.paths.csv_path)
    cfg.paths.output_dir = to_absolute_path(cfg.paths.output_dir)
    run_stage2(cfg)


if __name__ == "__main__":
    main()