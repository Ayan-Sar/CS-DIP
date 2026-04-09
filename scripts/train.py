"""
CS-DIP Training Script
======================

Implements Algorithm 1 from the CS-DIP paper: single-image optimization
using the Curvature-Steered Deep Image Prior.

Usage::

    python scripts/train.py --config configs/denoise_sigma25.yaml --image path/to/image.png

The script optimizes the CS-DIP network weights to reconstruct a clean
image from a degraded observation, using only the architecture as an
implicit regularizer — no external training data required.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_dip.losses import CSDIPLoss
from cs_dip.models import CSDIPNet, CSDIPNetConfig
from cs_dip.utils import (
    add_gaussian_noise,
    bicubic_downsample,
    compute_psnr,
    compute_ssim,
    get_degradation_operator,
    get_noise_input,
    load_image,
    save_image,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CS-DIP: Curvature-Steered Deep Image Prior Optimization"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input clean image (ground truth)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results. Default: results/"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'. Default: auto"
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save intermediate results every N iterations."
    )
    parser.add_argument(
        "--log_every", type=int, default=50,
        help="Log metrics every N iterations."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Run CS-DIP optimization (Algorithm 1)."""
    args = parse_args()
    config = load_config(args.config)

    # ---- Device setup ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[CS-DIP] Device: {device}")

    # ---- Reproducibility ----
    seed = config.get("seed", 42)
    set_seed(seed)

    # ---- Load clean image (ground truth) ----
    gt_image = load_image(args.image).to(device)  # (1, C, H, W)
    _, C, H, W = gt_image.shape
    print(f"[CS-DIP] Clean image: {args.image} ({C}ch, {H}×{W})")

    # ---- Create degraded observation ----
    task = config["task"]
    if task == "denoise":
        sigma = config["noise_sigma"]
        degraded = add_gaussian_noise(gt_image, sigma, seed=seed + 1)
        degradation_fn = None
        print(f"[CS-DIP] Task: Denoising (σ={sigma})")
    elif task == "sr":
        scale_factor = config["scale_factor"]
        degraded = bicubic_downsample(gt_image, scale_factor)
        degradation_fn = get_degradation_operator("sr", scale_factor=scale_factor)
        print(f"[CS-DIP] Task: Super-Resolution (×{scale_factor})")
    else:
        raise ValueError(f"Unknown task: {task}")

    # ---- Network setup ----
    channels = config.get("channels", [64, 128, 256, 512, 512])
    in_channels = config.get("input_channels", 32)
    net_config = CSDIPNetConfig(
        in_channels=in_channels,
        out_channels=C,
        encoder_channels=channels,
    )
    net = CSDIPNet(net_config).to(device)
    num_params = net.get_num_params()
    print(f"[CS-DIP] Network parameters: {num_params:,}")

    # ---- Fixed noise input z ----
    z = get_noise_input(1, in_channels, H, W, seed=seed + 2, device=str(device))

    # ---- Loss and optimizer ----
    lambda_curv = config.get("lambda_curv", 0.01)
    criterion = CSDIPLoss(lambda_curv=lambda_curv, degradation_fn=degradation_fn)
    lr = config.get("lr", 0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # ---- Output setup ----
    exp_name = Path(args.config).stem
    img_name = Path(args.image).stem
    output_dir = Path(args.output_dir) / exp_name / img_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save degraded image
    save_image(degraded, str(output_dir / "degraded.png"))

    # TensorBoard logging
    tb_dir = output_dir / "logs"
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ---- Optimization loop (Algorithm 1) ----
    iterations = config.get("iterations", 3000)
    best_psnr = 0.0
    best_output = None

    print(f"[CS-DIP] Starting optimization: {iterations} iterations, lr={lr}")
    print(f"[CS-DIP] λ_curv={lambda_curv}")
    print("-" * 60)

    pbar = tqdm(range(1, iterations + 1), desc="CS-DIP", ncols=100)
    start_time = time.time()

    for t in pbar:
        optimizer.zero_grad()

        # Forward pass: x_t = f_θ(z)
        x_t = net(z)

        # Compute total loss
        loss, loss_dict = criterion(x_t, degraded)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # ---- Logging ----
        if t % args.log_every == 0 or t == 1:
            with torch.no_grad():
                psnr = compute_psnr(x_t, gt_image)
                ssim = compute_ssim(x_t, gt_image)

            writer.add_scalar("Loss/total", loss_dict["total"], t)
            writer.add_scalar("Loss/data_fidelity", loss_dict["data_fidelity"], t)
            writer.add_scalar("Loss/curvature", loss_dict["curvature_consistency"], t)
            writer.add_scalar("Metrics/PSNR", psnr, t)
            writer.add_scalar("Metrics/SSIM", ssim, t)

            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                psnr=f"{psnr:.2f}",
                ssim=f"{ssim:.4f}",
            )

            # Track best result
            if psnr > best_psnr:
                best_psnr = psnr
                best_output = x_t.detach().clone()

        # ---- Save intermediate results ----
        if t % args.save_every == 0:
            save_image(x_t, str(output_dir / f"iter_{t:05d}.png"))

    elapsed = time.time() - start_time

    # ---- Save final results ----
    final_output = x_t.detach()
    save_image(final_output, str(output_dir / "restored_final.png"))
    if best_output is not None:
        save_image(best_output, str(output_dir / "restored_best.png"))

    # Final metrics
    with torch.no_grad():
        final_psnr = compute_psnr(final_output, gt_image)
        final_ssim = compute_ssim(final_output, gt_image)

    writer.close()

    print("\n" + "=" * 60)
    print("[CS-DIP] Optimization Complete")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Final PSNR:   {final_psnr:.2f} dB")
    print(f"  Final SSIM:   {final_ssim:.4f}")
    print(f"  Best PSNR:    {best_psnr:.2f} dB")
    print(f"  Results:      {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
