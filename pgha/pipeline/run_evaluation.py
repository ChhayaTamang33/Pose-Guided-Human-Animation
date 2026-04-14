import hydra
from omegaconf import DictConfig

from pgha.evaluation.temporal_jitter import main as tj_main
from pgha.evaluation.video_metrics import main as vm_main


@hydra.main(config_path="../../configs", config_name="evaluation")
def main(cfg: DictConfig):

    if cfg.temporal_jitter.enabled:
        tj_main(cfg)

    if cfg.video_metrics.enabled:
        vm_main(cfg)


if __name__ == "__main__":
    main()