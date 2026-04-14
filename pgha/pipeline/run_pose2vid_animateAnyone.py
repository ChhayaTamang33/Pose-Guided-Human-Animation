import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pgha.model.animateanyone_pose2vid_wrapper import run_pose2vid


@hydra.main(config_path="../../configs", config_name="pose2vid_animateAnyone")
def main(cfg: DictConfig):

    cfg.paths.video_dir = to_absolute_path(cfg.paths.video_dir)
    cfg.paths.pose_dir = to_absolute_path(cfg.paths.pose_dir)
    cfg.paths.output_dir = to_absolute_path(cfg.paths.output_dir)
    cfg.paths.log_path = to_absolute_path(cfg.paths.log_path)

    cfg.model.vae_path = to_absolute_path(cfg.model.vae_path)
    cfg.model.pretrained_base = to_absolute_path(cfg.model.pretrained_base)
    cfg.model.motion_module = to_absolute_path(cfg.model.motion_module)
    cfg.model.image_encoder = to_absolute_path(cfg.model.image_encoder)

    cfg.model.denoising_unet = to_absolute_path(cfg.model.denoising_unet)
    cfg.model.reference_unet = to_absolute_path(cfg.model.reference_unet)
    cfg.model.pose_guider = to_absolute_path(cfg.model.pose_guider)

    run_pose2vid(cfg)


if __name__ == "__main__":
    main()