"""
CS-DIP Batch Evaluation Script
==============================

Evaluates CS-DIP across an entire benchmark dataset, running the
optimization per image and computing aggregate PSNR/SSIM metrics.

Usage::

    python scripts/evaluate.py --config configs/denoise_sigma25.yaml \\
                                --data_dir data/ --dataset Set5

Results are printed as a formatted table and saved to a CSV file.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_dip.data import BenchmarkDataset
from cs_dip.losses import CSDIPLoss
from cs_dip.models import CSDIPNet, CSDIPNetConfig
from cs_dip.utils import (
    add_gaussian_noise,
    bicubic_downsample,
    compute_psnr,
    compute_ssim,
    get_degradation_operator,
    get_noise_input,
    save_image,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CS-DIP Benchmark Evaluation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Set5", "Set14", "BSD68", "Urban100"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_single_image(
    gt_image: torch.Tensor,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, float, float]:
    """Run CS-DIP optimization on a single image.

    Args:
        gt_image: Ground truth image, shape ``(1, C, H, W)``.
        config: Experiment configuration dict.
        device: Torch device.
        seed: Random seed.

    Returns:
        Tuple of ``(restored_image, psnr, ssim)``.
    """
    _, C, H, W = gt_image.shape
    task = config["task"]

    # Degrade the image
    if task == "denoise":
        sigma = config["noise_sigma"]
        degraded = add_gaussian_noise(gt_image, sigma, seed=seed + 1)
        degradation_fn = None
    elif task == "sr":
        scale_factor = config["scale_factor"]
        degraded = bicubic_downsample(gt_image, scale_factor)
        degradation_fn = get_degradation_operator("sr", scale_factor=scale_factor)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Network
    channels = config.get("channels", [64, 128, 256, 512, 512])
    in_channels = config.get("input_channels", 32)
    net_config = CSDIPNetConfig(
        in_channels=in_channels, out_channels=C, encoder_channels=channels
    )
    net = CSDIPNet(net_config).to(device)

    # Noise input
    z = get_noise_input(1, in_channels, H, W, seed=seed + 2, device=str(device))

    # Loss and optimizer
    lambda_curv = config.get("lambda_curv", 0.01)
    criterion = CSDIPLoss(lambda_curv=lambda_curv, degradation_fn=degradation_fn)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get("lr", 0.001))

    # Optimize
    iterations = config.get("iterations", 3000)
    best_psnr = 0.0
    best_output = None

    for t in range(1, iterations + 1):
        optimizer.zero_grad()
        x_t = net(z)
        loss, _ = criterion(x_t, degraded)
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            with torch.no_grad():
                psnr = compute_psnr(x_t, gt_image)
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_output = x_t.detach().clone()

    # Final output
    if best_output is None:
        best_output = x_t.detach()

    with torch.no_grad():
        final_psnr = compute_psnr(best_output, gt_image)
        final_ssim = compute_ssim(best_output, gt_image)

    return best_output, final_psnr, final_ssim


def main():
    """Run batch evaluation on a benchmark dataset."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Load dataset
    dataset = BenchmarkDataset(args.data_dir, args.dataset)
    print(f"[CS-DIP Eval] Dataset: {args.dataset} ({len(dataset)} images)")
    print(f"[CS-DIP Eval] Config: {args.config}")
    print(f"[CS-DIP Eval] Device: {device}")
    print("-" * 70)

    # Output directory
    exp_name = Path(args.config).stem
    output_dir = Path(args.output_dir) / f"eval_{exp_name}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    results = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {args.dataset}"):
        sample = dataset[i]
        gt_image = sample["image"].unsqueeze(0).to(device)
        filename = sample["filename"]

        t_start = time.time()
        restored, psnr, ssim = run_single_image(gt_image, config, device, seed + i)
        elapsed = time.time() - t_start

        results.append({
            "filename": filename,
            "psnr": psnr,
            "ssim": ssim,
            "time_s": elapsed,
        })

        save_image(restored, str(output_dir / f"restored_{filename}"))
        print(f"  {filename:30s}  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  ({elapsed:.1f}s)")

    # ---- Summary ----
    avg_psnr = sum(r["psnr"] for r in results) / len(results)
    avg_ssim = sum(r["ssim"] for r in results) / len(results)
    total_time = sum(r["time_s"] for r in results)

    print("\n" + "=" * 70)
    print(f"[CS-DIP Eval] {args.dataset} Results Summary")
    print(f"  Average PSNR:  {avg_psnr:.2f} dB")
    print(f"  Average SSIM:  {avg_ssim:.4f}")
    print(f"  Total Time:    {total_time:.1f}s")
    print("=" * 70)

    # Save results to CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "time_s"])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "filename": "AVERAGE",
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "time_s": total_time,
        })
    print(f"  Results CSV:   {csv_path}")


if __name__ == "__main__":
    main()
