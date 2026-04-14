import numpy as np
import imageio.v3 as iio
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def read_all_frames(video_path):
    return list(iio.imiter(video_path))


def align_all(real_frames, gen1_frames, gen2_frames):
    m = min(len(real_frames), len(gen1_frames), len(gen2_frames))
    return real_frames[:m], gen1_frames[:m], gen2_frames[:m]


def compute_self_jitter(frames, delta):
    return np.array([
        np.mean(np.abs(frames[t+delta].astype(np.float32) - frames[t].astype(np.float32)))
        for t in range(len(frames)-delta)
    ])


def compute_relative_jitter(real, gen, delta):
    errors = []
    for t in range(len(real)-delta):

        r0, r1 = real[t], real[t+delta]
        g0, g1 = gen[t], gen[t+delta]

        if r0.shape != g0.shape:
            g0 = cv2.resize(g0, (r0.shape[1], r0.shape[0]))
        if r1.shape != g1.shape:
            g1 = cv2.resize(g1, (r1.shape[1], r1.shape[0]))

        diff_real = np.abs(r1 - r0)
        diff_gen  = np.abs(g1 - g0)

        errors.append(np.mean(np.abs(diff_real - diff_gen)))

    return np.array(errors)


def run_pair(cfg, gt_path, anim_path, mimic_path, out_dir):

    real = read_all_frames(gt_path)
    a = read_all_frames(anim_path)
    m = read_all_frames(mimic_path)

    real, a, m = align_all(real, a, m)

    d = cfg.temporal_jitter.delta
    w = cfg.temporal_jitter.window_size

    er = compute_self_jitter(real, d)
    ea = compute_relative_jitter(real, a, d)
    em = compute_relative_jitter(real, m, d)

    kernel = np.ones(w)/w
    er = np.convolve(er, kernel, mode='same')
    ea = np.convolve(ea, kernel, mode='same')
    em = np.convolve(em, kernel, mode='same')

    name = Path(gt_path).stem
    pdf_path = out_dir / f"{name}_tje.pdf"

    plt.figure()
    plt.plot(er, label="Real")
    plt.plot(ea, label="AnimateAnyone")
    plt.plot(em, label="MimicMotion")
    plt.legend()
    plt.savefig(pdf_path)
    plt.close()


@hydra.main(config_path="../../configs", config_name="evaluation")
def main(cfg: DictConfig):

    # 🔥 critical for Hydra
    cfg.paths.gt_root = to_absolute_path(cfg.paths.gt_root)
    cfg.paths.animate_root = to_absolute_path(cfg.paths.animate_root)
    cfg.paths.mimic_root = to_absolute_path(cfg.paths.mimic_root)
    cfg.paths.output_dir = to_absolute_path(cfg.paths.output_dir)

    gt_root = Path(cfg.paths.gt_root)
    anim_root = Path(cfg.paths.animate_root)
    mimic_root = Path(cfg.paths.mimic_root)

    out_dir = Path(cfg.paths.output_dir) / "temporal_jitter"
    out_dir.mkdir(parents=True, exist_ok=True)

    for gt in gt_root.rglob("*.mp4"):

        name = gt.stem
        anim = list(anim_root.rglob(f"{name}*.mp4"))
        mimic = list(mimic_root.rglob(f"{name}*.mp4"))

        if not anim or not mimic:
            continue

        run_pair(cfg, gt, anim[0], mimic[0], out_dir)


if __name__ == "__main__":
    main()