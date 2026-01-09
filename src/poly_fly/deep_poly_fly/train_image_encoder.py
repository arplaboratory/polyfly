# create a training environment to train the auto network defined in network.py to correctly predict the depth images 
# 1. Load dataset using build_dataset in zarr_io.py
# 2. Create nn module defined in network.py
# 3. Train the network using MSE loss between input depth and reconstructed depth
# 4. Use pytorch. Adam optimizer, lr=1e-3, batch size = 64, num epochs = 100
#.5. During training, print debug info every 10 iterations
# 6. After training, plot the training loss curve and save the trained model
# 7. Pick an example from the dataset and visualize the input depth and reconstructed depth

from __future__ import annotations
import os
from pathlib import Path
import math
import importlib
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)  # non-interactive backend to avoid Tk issues in workers
import matplotlib.pyplot as plt
import zarr  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from poly_fly.data_io.zarr_io import ZARR_DIR as ZARR_DIR
from poly_fly.data_io.zarr_io import load_zarr_folder
from poly_fly.deep_poly_fly.model.vae import VanillaVAE

class DepthFramesDataset(Dataset):
    """
    Dataset over all depth frames aggregated from load_zarr_folder.
    Each item is a float32 tensor of shape (1, H, W), optionally normalized by a fixed scale.
    """
    def __init__(self, datasets, normalize_scale: float | None = None):
        super().__init__()
        self.datasets = datasets
        self.index = []
        self.min_depth_range = 0.02
        self.max_depth_range = 1.0
        for di, ds in enumerate(self.datasets):
            depth = ds.get("depth")
            if depth is None:
                continue
            T = int(depth.shape[0])
            if T <= 0:
                continue
            # map to (dataset_idx, local_frame_idx)
            self.index.extend([(di, i) for i in range(T)])
        if not self.index:
            raise RuntimeError("No depth frames found in provided Zarr datasets")

        # Compute global mean and std over all pixels in all frames
        print("Computing dataset-wide depth mean and std over {} frames...".format(len(self.index)))
        n_mean_computation = 200
        if len(self.index) > n_mean_computation:
            np.random.seed(0)
            sampled_indices = np.random.choice(len(self.index), size=n_mean_computation, replace=False)
            self.index_mean_computation = [self.index[i] for i in sampled_indices]

        acc_sum = 0.0
        acc_sq = 0.0
        n_pix = 0
        for di, ti in self.index_mean_computation:
            f = np.asarray(self.datasets[di]["depth"][ti], dtype=np.float32)
            acc_sum += float(f.sum())
            acc_sq += float((f * f).sum())
            n_pix += int(f.size)
        mean = acc_sum / max(n_pix, 1)
        var = max(acc_sq / max(n_pix, 1) - mean * mean, 0.0)
        std = float(np.sqrt(var)) if var > 0 else 1.0

        self.mean = float(mean)
        self.std = float(std if std > 0 else 1.0)

        print(f"approx mean={self.mean:.6f}  std={self.std:.6f}  n_frames={len(self.index_mean_computation)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        di, ti = self.index[idx]
        depth = self.datasets[di]["depth"]
        frame = np.asarray(depth[ti])  # (H, W) or (H, W, C)
        frame = frame.astype(np.float32, copy=False)

        # Normalize to zero mean and unit std using dataset-wide stats
        frame = (frame - self.mean) / (self.std + 1e-6)
        # frame = (frame - self.min_depth_range) / (self.max_depth_range - self.min_depth_range + 1e-6)
        return torch.from_numpy(frame).unsqueeze(0)  # (1, H, W)


def build_dataset(zarr_path: Path) -> Dataset:
    """
    Use zarr_io.load_zarr_folder to gather trajectories, then expose frames as a Dataset.
    """
    args = argparse.Namespace(combined=False, lazy_depth_only=True)  # enable lazy depth-only loading
    datasets = load_zarr_folder(zarr_path, args, limit=None)
    return DepthFramesDataset(datasets)


def resolve_zarr_path(arg: str | None) -> Path:
    """
    If arg endswith .zarr -> treat as full path (under ZARR_DIR unless absolute).
    Else treat as dataset subdirectory name and resolve as ZARR_DIR/<arg>.zarr.
    If arg is None, default to 'forests.zarr'.
    """
    p = Path(arg)
    return Path(ZARR_DIR) / f"{arg}.zarr"

def create_model(img_shape, latent_dim: int = 64):
    """
    Build VanillaVAE with in_channels=1 and provided latent_dim.
    img_shape is a tuple like (1, H, W).
    """
    model = VanillaVAE(in_channels=1, latent_dim=latent_dim)
    model.apply(init_weights_kaiming)
    return model 

# Kaiming (He) initialization for common layers
def init_weights_kaiming(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if getattr(m, "weight", None) is not None:
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if getattr(m, "weight", None) is not None and m.weight is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias)

# NEW: LR scheduler factory
def build_lr_scheduler(optimizer, args):
    """
    Returns a torch.optim.lr_scheduler.* instance or None.
    """
    kind = str(getattr(args, "lr_scheduler", "none")).lower()
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=getattr(args, "lr_step_size", 30), gamma=getattr(args, "lr_decay_rate", 0.5)
        )
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(args, "epochs", 100), eta_min=getattr(args, "lr_min", 1e-6)
        )
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=getattr(args, "lr_decay_rate", 0.1),
            patience=getattr(args, "lr_patience", 10),
            min_lr=getattr(args, "lr_min", 1e-6),
            verbose=False,
        )
    return None


def train_one_epoch(model, loader, optimizer, device, log_interval=10):
    model.train()
    l2_reg = 0 #1e-2
    running = 0.0
    count = 0
    losses = []
    for batch_idx, x in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        recons, input_x, mu, log_var = model(x)
        loss = model.loss_function(recons, input_x, mu, log_var)['loss']

        # add l2 regularization on mu to keep it small
        # Find l2 norm of mu and add to loss
        loss += torch.square(mu).mean() * l2_reg
        loss.backward()
        optimizer.step()

        running += loss.item()
        count += 1
        running, count = 0.0, 0
        losses.append(loss.item())
    print(f"[iter {batch_idx:5d}] loss={loss.item():0.3f}, mu mean={torch.mean(torch.abs(mu)):.3f}")
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    losses = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        recons, input_x, mu, log_var = model(x)
        loss = model.loss_function(recons, input_x, mu, log_var, M_N=mn)['loss']
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else math.nan


def save_curves(train_hist, val_hist, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(train_hist, label="train")
    if val_hist:
        plt.plot(val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png")
    plt.close()


@torch.no_grad()
def save_example_reconstruction(model, loader, device, out_dir: Path, epoch: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    ds = loader.dataset
    if len(ds) == 0:
        return

    # Pick up to 10 random indices and build a batch
    n_samples = min(10, len(ds))
    idxs = torch.randperm(len(ds))[:n_samples].tolist()
    samples = [ds[i] for i in idxs]
    xb = torch.stack(samples, dim=0).to(device)  # (N,1,H,W)

    # Forward pass once for the batch
    yb, input_x, mu, log_var = model(xb)

    x_np = xb[:, 0].detach().cpu().numpy()
    y_np = yb[:, 0].detach().cpu().numpy()

    # Plot N rows (input, reconstruction)
    n_rows = n_samples
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(n_rows):
        axes[r, 0].imshow(x_np[r], cmap="plasma")
        axes[r, 0].set_title(f"Input #{idxs[r]}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(y_np[r], cmap="plasma")
        axes[r, 1].set_title(f"Reconstruction #{idxs[r]}")
        axes[r, 1].axis("off")

    plt.tight_layout()
    suffix = f"_ep{epoch:04d}" if epoch is not None else ""
    out_path = out_dir / f"reconstruction_examples{suffix}_{n_samples}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="forests", help="Zarr dataset name or .zarr path")
    ap.add_argument("--latent-dim", type=int, default=64, help="Latent dimension for VanillaVAE")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=100*1e-5)
    # NEW: evaluation-only flag
    ap.add_argument("--eval", action="store_true", help="Run evaluation only: load checkpoint and save reconstructions, then exit")
    # NEW: LR decay options
    ap.add_argument("--lr-scheduler", type=str, default="step", choices=["none", "step", "cosine", "plateau"], help="LR decay schedule")
    ap.add_argument("--lr-decay-rate", type=float, default=0.9, help="Gamma for StepLR/ReduceLROnPlateau")
    ap.add_argument("--lr-step-size", type=int, default=500, help="Epoch interval for StepLR")
    # KL annealing (M_N scheduling)
    ap.add_argument("--kl-start", type=float, default=0.0, help="Starting M_N (KL weight)")
    ap.add_argument("--kl-end", type=float, default=0.000, help="Final M_N (KL weight)")
    ap.add_argument("--kl-anneal-epochs", type=int, default=300, help="Epochs to ramp M_N from kl-start to kl-end")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for model and plots")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zarr_path = resolve_zarr_path(args.dataset)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(ZARR_DIR) / "models" / (zarr_path.stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "autoencoder.pth"

    ds = build_dataset(zarr_path)
    sample_shape = tuple(ds[0].shape)
    model = create_model(sample_shape, latent_dim=args.latent_dim).to(device)

    # Evaluation-only mode: load checkpoint if present and save reconstructions, then exit
    if args.eval:
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                print(f"Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint: {e}. Proceeding with current model.")
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}. Proceeding with current model.")
        eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        save_example_reconstruction(model, eval_loader, device, out_dir)
        print(f"Saved evaluation reconstructions to {out_dir}")
        return

    # Train/val split
    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True) if val_len > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)

    train_hist, val_hist = [], []
    for epoch in range(1, args.epochs + 1):
        # Linear KL annealing: clamp after warmup
        if args.kl_anneal_epochs <= 0:
            current_M_N = float(args.kl_end)
        else:
            alpha = min(max((epoch - 1) / float(args.kl_anneal_epochs), 0.0), 1.0)
            current_M_N = float(args.kl_start + alpha * (args.kl_end - args.kl_start))
        
        print(f"[epoch {epoch}] M_N={current_M_N:.4f}")
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, log_interval=args.log_interval)
        train_hist.append(tr_loss)
        if val_loader is not None:
            va_loss = eval_epoch(model, val_loader, device)
            val_hist.append(va_loss)
            print(f"[epoch {epoch}] train={tr_loss:.6f}  val={va_loss:.6f}")
        else:
            va_loss = None
            print(f"[epoch {epoch}] train={tr_loss:.6f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss if va_loss is not None else tr_loss)
            else:
                scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"[epoch {epoch}] lr={curr_lr:.6g}")

        # Every 5 epochs: save checkpoint and reconstructions with epoch in filename
        if epoch % 5 == 0:
            ckpt_epoch_path = out_dir / f"autoencoder_ep{epoch:04d}.pth"
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_epoch_path)
            print(f"[epoch {epoch}] saved checkpoint: {ckpt_epoch_path.name}")
            save_example_reconstruction(model, train_loader, device, out_dir, epoch=epoch)
            print(f"[epoch {epoch}] saved reconstructions")

    # Save model and curves
    torch.save({"model_state": model.state_dict(), "epochs": args.epochs}, ckpt_path)
    print(f"Saved model to {ckpt_path}")

    save_curves(train_hist, val_hist, out_dir)
    save_example_reconstruction(model, train_loader, device, out_dir)
    print(f"Saved training curve and example reconstruction to {out_dir}")

if __name__ == "__main__":
    main()

