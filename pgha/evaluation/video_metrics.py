import numpy as np
import pandas as pd
import torch
import lpips
import imageio.v3 as iio
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import math
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def compute_psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return 100 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(x, y):
    return ssim(x, y, channel_axis=2, data_range=255)


def compute_lpips(x, y, model, device):
    x = torch.tensor(x).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
    y = torch.tensor(y).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
    return model(x.to(device), y.to(device)).item()


def resize_frame(frame, shape):
    return resize(frame, shape, preserve_range=True).astype(np.uint8)


def compare(gt_path, gen_path, model, device):

    gt_iter = iio.imiter(gt_path)
    gen_iter = iio.imiter(gen_path)

    gt_first = next(gt_iter)
    shape = gt_first.shape

    gt_iter = iio.imiter(gt_path)
    gen_iter = iio.imiter(gen_path)

    psnr, ssim_v, lp = [], [], []

    for g, p in zip(gt_iter, gen_iter):
        p = resize_frame(p, shape)
        psnr.append(compute_psnr(g, p))
        ssim_v.append(compute_ssim(g, p))
        lp.append(compute_lpips(g, p, model, device))

    return np.mean(psnr), np.mean(ssim_v), np.mean(lp)


@hydra.main(config_path="../../configs", config_name="evaluation")
def main(cfg: DictConfig):

    cfg.paths.gt_root = to_absolute_path(cfg.paths.gt_root)
    cfg.paths.animate_root = to_absolute_path(cfg.paths.animate_root)
    cfg.paths.mimic_root = to_absolute_path(cfg.paths.mimic_root)
    cfg.paths.output_csv = to_absolute_path(cfg.paths.output_csv)

    device = cfg.system.device if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)

    gt = {p.stem: p for p in Path(cfg.paths.gt_root).rglob("*.mp4")}
    anim = {p.stem.replace("_generated",""): p for p in Path(cfg.paths.animate_root).rglob("*.mp4")}
    mimic = {p.stem: p for p in Path(cfg.paths.mimic_root).rglob("*.mp4")}

    results = []

    for name, gt_path in gt.items():

        if name not in anim or name not in mimic:
            continue

        psnr_a, ssim_a, lpips_a = compare(gt_path, anim[name], model, device)
        psnr_m, ssim_m, lpips_m = compare(gt_path, mimic[name], model, device)

        results.append({
            "video": name,
            "animate_psnr": psnr_a,
            "animate_ssim": ssim_a,
            "animate_lpips": lpips_a,
            "mimic_psnr": psnr_m,
            "mimic_ssim": ssim_m,
            "mimic_lpips": lpips_m,
        })

    pd.DataFrame(results).to_csv(cfg.paths.output_csv, index=False)


if __name__ == "__main__":
    main()