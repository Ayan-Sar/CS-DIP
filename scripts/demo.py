"""
CS-DIP Demo Script
==================

Run CS-DIP on a single image with side-by-side visualization showing:
- Degraded input
- Restored output
- Ground truth
- Curvature map visualization
- PSNR convergence curve

Usage::

    python scripts/demo.py --config configs/denoise_sigma25.yaml \\
                           --image path/to/image.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs_dip.losses import CSDIPLoss
from cs_dip.models import CSDIPNet, CSDIPNetConfig, CurvatureMap
from cs_dip.utils import (
    add_gaussian_noise,
    bicubic_downsample,
    bicubic_upsample,
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
    parser = argparse.ArgumentParser(description="CS-DIP Single Image Demo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/demo")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    return parser.parse_args()


def tensor_to_numpy(tensor: torch.Tensor):
    """Convert a (1, C, H, W) or (C, H, W) tensor to numpy for display."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def main():
    """Run CS-DIP demo with visualization."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Load image
    gt_image = load_image(args.image).to(device)
    _, C, H, W = gt_image.shape

    # Degrade
    task = config["task"]
    if task == "denoise":
        sigma = config["noise_sigma"]
        degraded = add_gaussian_noise(gt_image, sigma, seed=seed + 1)
        degradation_fn = None
        task_str = f"Denoising (σ={sigma})"
    elif task == "sr":
        scale_factor = config["scale_factor"]
        degraded = bicubic_downsample(gt_image, scale_factor)
        degradation_fn = get_degradation_operator("sr", scale_factor=scale_factor)
        task_str = f"Super-Resolution (×{scale_factor})"
    else:
        raise ValueError(f"Unknown task: {task}")

    # Network
    channels = config.get("channels", [64, 128, 256, 512, 512])
    in_channels = config.get("input_channels", 32)
    net_config = CSDIPNetConfig(
        in_channels=in_channels, out_channels=C, encoder_channels=channels
    )
    net = CSDIPNet(net_config).to(device)
    z = get_noise_input(1, in_channels, H, W, seed=seed + 2, device=str(device))

    # Loss and optimizer
    lambda_curv = config.get("lambda_curv", 0.01)
    criterion = CSDIPLoss(lambda_curv=lambda_curv, degradation_fn=degradation_fn)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get("lr", 0.001))

    # Optimize and track PSNR
    iterations = config.get("iterations", 3000)
    psnr_history = []
    loss_history = []

    print(f"[CS-DIP Demo] {task_str}")
    print(f"[CS-DIP Demo] Image: {args.image} ({C}ch, {H}×{W})")
    print(f"[CS-DIP Demo] Running {iterations} iterations on {device}...")

    pbar = tqdm(range(1, iterations + 1), desc="CS-DIP Demo", ncols=100)
    best_psnr = 0.0
    best_output = None

    for t in pbar:
        optimizer.zero_grad()
        x_t = net(z)
        loss, loss_dict = criterion(x_t, degraded)
        loss.backward()
        optimizer.step()

        if t % 20 == 0 or t == 1:
            with torch.no_grad():
                psnr = compute_psnr(x_t, gt_image)
                psnr_history.append((t, psnr))
                loss_history.append((t, loss_dict["total"]))
                pbar.set_postfix(psnr=f"{psnr:.2f}", loss=f"{loss_dict['total']:.4f}")
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_output = x_t.detach().clone()

    if best_output is None:
        best_output = x_t.detach()

    final_psnr = compute_psnr(best_output, gt_image)
    final_ssim = compute_ssim(best_output, gt_image)

    # Compute curvature map for visualization
    with torch.no_grad():
        curv_module = CurvatureMap().to(device)
        K, H_curv, kappa = curv_module(best_output)

    # ---- Visualization ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"CS-DIP: {task_str} | PSNR={final_psnr:.2f} dB | SSIM={final_ssim:.4f}",
        fontsize=16, fontweight="bold",
    )

    # Row 1: Images
    degraded_display = degraded
    if task == "sr":
        degraded_display = bicubic_upsample(degraded, scale_factor)

    axes[0, 0].imshow(tensor_to_numpy(degraded_display), cmap="gray" if C == 1 else None)
    axes[0, 0].set_title("Degraded Input", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tensor_to_numpy(best_output), cmap="gray" if C == 1 else None)
    axes[0, 1].set_title(f"CS-DIP Restored (PSNR={final_psnr:.2f})", fontsize=13)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(tensor_to_numpy(gt_image), cmap="gray" if C == 1 else None)
    axes[0, 2].set_title("Ground Truth", fontsize=13)
    axes[0, 2].axis("off")

    # Row 2: Curvature + Convergence
    kappa_np = kappa[0, 0].detach().cpu().numpy()
    axes[1, 0].imshow(kappa_np, cmap="inferno")
    axes[1, 0].set_title("Curvature Map κ", fontsize=13)
    axes[1, 0].axis("off")

    # PSNR convergence
    iters, psnrs = zip(*psnr_history)
    axes[1, 1].plot(iters, psnrs, color="#2196F3", linewidth=2)
    axes[1, 1].set_xlabel("Iteration", fontsize=11)
    axes[1, 1].set_ylabel("PSNR (dB)", fontsize=11)
    axes[1, 1].set_title("PSNR Convergence", fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)

    # Loss convergence
    iters_l, losses = zip(*loss_history)
    axes[1, 2].semilogy(iters_l, losses, color="#FF5722", linewidth=2)
    axes[1, 2].set_xlabel("Iteration", fontsize=11)
    axes[1, 2].set_ylabel("Total Loss", fontsize=11)
    axes[1, 2].set_title("Loss Convergence", fontsize=13)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "demo_result.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n[CS-DIP Demo] Figure saved: {fig_path}")

    if args.show:
        plt.show()
    plt.close()

    # Save individual images
    save_image(best_output, str(output_dir / "restored.png"))
    save_image(degraded, str(output_dir / "degraded.png"))

    print(f"[CS-DIP Demo] Final PSNR: {final_psnr:.2f} dB, SSIM: {final_ssim:.4f}")


if __name__ == "__main__":
    main()
