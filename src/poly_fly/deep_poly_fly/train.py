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
from poly_fly.data_io.utils import load_normalization_stats
from poly_fly.data_io.enums import DatasetKeys as DK

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from poly_fly.data_io.zarr_io import ZARR_DIR as ZARR_DIR
from poly_fly.data_io.zarr_io import load_zarr_folder
from poly_fly.deep_poly_fly.model.policy import Policy, build_policy_from_config, load_model_from_checkpoint
from poly_fly.deep_poly_fly.dataloader.loader import DepthAndStateDataset
import yaml


def build_dataset(zarr_path: Path, use_rotation_mat) -> Dataset:
    """
    Use zarr_io.load_zarr_folder to gather trajectories, then expose frames as a Dataset.
    """
    # Must not limit to depth-only; we need state and future arrays too.
    args = argparse.Namespace(combined=False, lazy_depth_only=True)
    datasets = load_zarr_folder(zarr_path, args, limit=1, random_sample=False)
    return DepthAndStateDataset(datasets, use_rotation_mat=use_rotation_mat)


def resolve_zarr_path(arg: str | None) -> Path:
    """
    If arg endswith .zarr -> treat as full path (under ZARR_DIR unless absolute).
    Else treat as dataset subdirectory name and resolve as ZARR_DIR/<arg>.zarr.
    If arg is None, default to 'forests.zarr'.
    """
    p = Path(arg)
    return Path(ZARR_DIR) / f"{arg}.zarr"

def create_model(img_shape, *, policy_config_path: str) -> nn.Module:
    """
    Build a Policy model from policy.py using a provided YAML config path.
    img_shape is a tuple like (1, H, W) and is not used here; kept for signature compatibility.
    """
    if not policy_config_path or not Path(policy_config_path).exists():
        raise FileNotFoundError("A valid --policy-config YAML path is required to build the Policy model.")
    model = build_policy_from_config(policy_config_path)
    if not hasattr(model, "loss_function"):
        raise AttributeError("Policy model must define a 'loss_function' method to compute training loss.")
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
    losses = []
    ae_losses = []
    prediction_losses = []
    # running window for debug prints
    run_loss_sum, run_ae_sum, run_pred_sum, run_count = 0, 0.0, 0.0, 0
    for batch_idx, batch in enumerate(loader, 1):
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
            raise RuntimeError("Expected each batch to be a 4-tuple: (depth, states, future_traj)")

        depth, states, gt_trajectory = batch
        depth = depth.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        gt_trajectory = gt_trajectory.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(states=states, depth_images=depth)
        predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var = outputs

        if predicted_traj.shape[-2] != gt_trajectory.shape[-2] or predicted_traj.shape[-1] != gt_trajectory.shape[-1]: 
            raise RuntimeError(f"predicted_traj shape {predicted_traj.shape} does not match gt_trajectory shape {gt_trajectory.shape}")

        all_loss = model.loss_function(gt_trajectory, predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var)
        loss = all_loss["loss"]
        ae_loss = all_loss["scaled_ae_loss"]
        prediction_loss = all_loss["prediction_loss"]
        loss.backward()
        optimizer.step()

        # aggregate epoch metrics
        lval = float(loss.item())
        aval = float(ae_loss)
        losses.append(lval)
        ae_losses.append(aval)
        prediction_losses.append(float(prediction_loss.item()))

        # accumulate and print every log_interval batches
        run_loss_sum += lval
        run_ae_sum += aval
        run_pred_sum += float(prediction_loss.item())
        run_count += 1
        if batch_idx % max(int(log_interval), 1) == 0:
            print(f"[iter {batch_idx:5d}] loss={(run_loss_sum/run_count):0.6f}, ae_loss={(run_ae_sum/run_count):0.3f}, pred_loss={(run_pred_sum/run_count):0.3f}")
            run_loss_sum, run_ae_sum, run_count = 0.0, 0.0, 0
            run_pred_sum = 0.0

    avg_loss = float(np.mean(losses)) if losses else math.nan
    avg_ae = float(np.mean(ae_losses)) if ae_losses else math.nan
    avg_pred = float(np.mean(prediction_losses)) if prediction_losses else math.nan
    return avg_loss, avg_ae, avg_pred


@torch.no_grad()
def eval_epoch(model, loader, device, ds):
    model.eval()
    losses = []
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
            raise RuntimeError("Expected each batch to be a 3-tuple: (depth, states, future_traj)")
        depth, states, gt_trajectory = batch

        depth = depth.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        gt_trajectory = gt_trajectory.to(device, non_blocking=True)

        outputs = model(states=states, depth_images=depth)
        if not (isinstance(outputs, (list, tuple)) and len(outputs) == 6):
            raise RuntimeError("Policy model forward must return (predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var)")
        predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var = outputs

        all_loss = model.loss_function(gt_trajectory, predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var)
        loss = all_loss["loss"]
        recon_loss = all_loss["recon_loss"]
        kld_loss = all_loss["kld_loss"]
        ae_loss = all_loss["scaled_ae_loss"]

        losses.append(float(loss.item()))

    print_unnormalized_errors(ds, predicted_traj, gt_trajectory)
        
    return float(np.mean(losses)) if losses else math.nan

def print_unnormalized_errors(base_ds, predicted_traj, gt_trajectory):
    pred_mode0 = predicted_traj[:, 0, ...] if predicted_traj.ndim == 4 else predicted_traj  # (N, H, D)

    # unnormalize both pred and gt
    pred_un = base_ds.unnormalize_data(pred_mode0)          # (N, H, D) in original units
    gt_un   = base_ds.unnormalize_data(gt_trajectory)       # (N, H, D) in original units

    # determine split sizes using stats and current representation
    use_rot = bool(getattr(base_ds, "use_rotation_mat", True))
    ordered_keys = [
        DK.FUTURE_ROBOT_POS,
        DK.FUTURE_PAYLOAD_POS,
        DK.FUTURE_ROT_MAT if use_rot else DK.FUTURE_QUATERNION,
        DK.FUTURE_ROBOT_VEL,
        DK.FUTURE_PAYLOAD_VEL,
    ]
    split_sizes = [int(base_ds.future_stats[k]["mean"].shape[-1]) for k in ordered_keys]

    # split into components
    p_parts = torch.split(pred_un, split_sizes, dim=-1)
    g_parts = torch.split(gt_un,   split_sizes, dim=-1)

        
    # mse helper
    def mse(a, b) -> float:
        err_sq = torch.sum((a - b) ** 2, dim=-1)  # (N, H)
        err = torch.sqrt(err_sq)               # (N, H)
        err_mean = torch.mean(err, dim=(0, 1)) 
        return float(err_mean.item())

    def std(a, b) -> float:
        err_sq = torch.sum((a - b) ** 2, dim=-1)  # (N, H)
        err = torch.sqrt(err_sq)               # (N, H)
        err_mean = torch.mean(err, dim=(1)) 
        err_std = torch.std(err_mean, dim=0)
        return float(err_std.item())

    def max_val(a, b) -> float:
        err_sq = torch.sum((a - b) ** 2, dim=-1)  # (N, H)
        err = torch.sqrt(err_sq)               # (N, H)
        err_mean = torch.mean(err, dim=(1))
        err_max = torch.max(err_mean, dim=0)[0]
        return float(err_max.item())

    comp_names = [
        "robot_pos",
        "payload_pos",
        "rot_mat" if use_rot else "quat",
        "robot_vel",
        "payload_vel",
    ]
    comp_errors = [mse(p, g) for p, g in zip(p_parts, g_parts)]
    comp_std = [std(p, g) for p, g in zip(p_parts, g_parts)]
    comp_max_errors = [max_val(p, g) for p, g in zip(p_parts, g_parts)]

    print("[eval unnorm mse] "
            + ", ".join(f"{n}={e:.3f}" for n, e in zip(comp_names, comp_errors)))
    print("[eval unnorm std] "
            + ", ".join(f"{n}={e:.3f}" for n, e in zip(comp_names, comp_std)))
    print("[eval unnorm max] "  
            + ", ".join(f"{n}={e:.3f}" for n, e in zip(comp_names, comp_max_errors)))

    # --- REPLACE: per-horizon mean error printing ---
    def per_horizon_mean(a, b) -> torch.Tensor:
        # a,b: (N, H, Dcomp) -> return (H,)
        err_sq = torch.sum((a - b) ** 2, dim=-1)   # (N, H)
        err = torch.sqrt(err_sq)                   # (N, H)
        return torch.mean(err, dim=0)              # (H,)

    per_h_means = [per_horizon_mean(p, g) for p, g in zip(p_parts, g_parts)]  # list[(H,)]
    # Convert to lists once for printing
    per_h_means_list = [ph.detach().cpu().tolist() for ph in per_h_means]
    H = len(per_h_means_list[0]) if per_h_means_list else 0

    print("eval unnorm mean")
    for i in range(H):
        vals = ", ".join(f"{name}={per_h_means_list[j][i]:.3f}" for j, name in enumerate(comp_names))
        print(f"step {i+1}: {vals}")
    # --- END REPLACE ---

@torch.no_grad()
def visualize_positions_only(model,
                             val_loader,
                             device,
                             out_dir,
                             epoch: int,
                             n_examples: int = 4):
    """
    Visualize GT robot positions vs predicted robot positions.
    Assumptions:
      - val_loader yields (depth_images, states, future_trajectory)
      - model(depth_images=..., states=...) -> tuple where first element is predicted_traj
        with shape (N, M, H, 13). We plot pred[:, 0, :, :3] vs GT Y[..., :3].
      - Unscaling uses per-horizon stats from load_normalization_stats() for DK.FUTURE_ROBOT_POS.
    """
    import torch
    import numpy as np
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from poly_fly.data_io.utils import load_normalization_stats
    from poly_fly.data_io.enums import DatasetKeys as DK

    model.eval()
    viz_dir = Path(out_dir) / "viz_pos" / f"epoch_{epoch:04d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fetch one validation batch ----
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print("[viz_pos] val_loader empty; skipping.")
        return

    # Unpack (depth, state, Y)
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        depth, state, Y = batch[0], batch[1], batch[2]
    elif isinstance(batch, dict):
        depth = batch.get("depth") or batch.get("depth_images")
        state = batch.get("state") or batch.get("states") or batch.get("state_concat")
        Y     = batch.get("y") or batch.get("traj") or batch.get("future")
    else:
        print("[viz_pos] Unrecognized batch format; expected (depth, state, Y) or dict.")
        return
    if depth is None or state is None or Y is None:
        print("[viz_pos] Missing depth/state/targets in batch; skipping.")
        return

    # Move to device
    depth_b = depth.to(device) if torch.is_tensor(depth) else torch.as_tensor(depth, device=device)
    state_b = state.to(device) if torch.is_tensor(state) else torch.as_tensor(state, device=device)
    Y_b     = Y.to(device)     if torch.is_tensor(Y)     else torch.as_tensor(Y,     device=device)

    # ---- Predict ----
    try:
        outputs = model(depth_images=depth_b, states=state_b)
    except TypeError:
        try:
            outputs = model(depth_b, state_b)
        except TypeError:
            outputs = model((depth_b, state_b))

    # Model returns a tuple; first is predicted_traj
    predicted_traj = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if not torch.is_tensor(predicted_traj):
        predicted_traj = torch.as_tensor(predicted_traj, device=device)

    # Expect (N, M, H, 13)
    assert predicted_traj.ndim == 4 and predicted_traj.shape[-1] == 16, \
        f"predicted_traj expected (N,M,H,13), got {tuple(predicted_traj.shape)}"

    # Select mode 0 and robot xyz
    P = predicted_traj[:, 0, :, :3]  # (N, H, 3)

    # GT robot positions are leading 3 dims by dataset construction
    assert Y_b.ndim >= 3 and Y_b.shape[-1] >= 3, f"targets must have last dim >=3, got {tuple(Y_b.shape)}"
    Y_pos = Y_b[..., :3]  # (N, H, 3) or compatible

    # ---- Unscale using per-horizon stats ----
    dict_mean, dict_std = load_normalization_stats()
    fut_pos_mean = np.asarray(dict_mean[DK.FUTURE_ROBOT_POS], dtype=np.float32)  # (H_stats, 3)
    fut_pos_std  = np.asarray(dict_std[DK.FUTURE_ROBOT_POS],  dtype=np.float32)  # (H_stats, 3)

    H_pred  = P.shape[1]
    H_gt    = Y_pos.shape[1]
    H_stats = fut_pos_mean.shape[0]
    H       = min(H_pred, H_gt, H_stats)

    # Slice to common horizon
    P     = P[:, :H]          # (N, H, 3) normalized
    Y_pos = Y_pos[:, :H]      # (N, H, 3) normalized
    m = torch.as_tensor(fut_pos_mean[:H], device=device)  # (H, 3)
    s = torch.as_tensor(fut_pos_std[:H],  device=device)  # (H, 3)

    # x = x_norm * std + mean
    eps = 1e-6
    P_un = P * (s.unsqueeze(0) + eps) + m.unsqueeze(0)   # (N, H, 3)
    Y_un = Y_pos * (s.unsqueeze(0) + eps) + m.unsqueeze(0)

    # ---- Plot a few examples ----
    N = Y_un.shape[0]
    n_plot = N
    t_np = np.arange(H)
    P_np = P_un.detach().cpu().numpy()
    Y_np = Y_un.detach().cpu().numpy()

    for i in range(n_plot):
        fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
        labels = ["x", "y", "z"]
        for d in range(3):
            ax = axes[d]
            ax.plot(t_np, Y_np[i, :, d], label="gt", linewidth=1.6)
            ax.plot(t_np, P_np[i, :, d], label="pred", linestyle="--", linewidth=1.3)
            ax.set_ylabel(labels[d])
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("step")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Robot position — example {i} — epoch {epoch}")
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_path = viz_dir / f"pos_ex{i}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    print(f"[viz_pos] Saved {n_plot} position figure(s) to: {viz_dir}")

@torch.no_grad()
def evaluate_checkpoint_and_visualize(
    ckpt_path,
    config_path,
    device,
    val_loader,
    out_dir,
    epoch,
    ds, 
    n_examples: int = 4,
):
    # Load via your provided helper
    re_model = load_model_from_checkpoint(
        path_chkpt=str(ckpt_path),
        path_config=str(config_path),
        device=device,
    )

    # One validation epoch
    val_score = eval_epoch(re_model, val_loader, device, ds)
    print(f"[epoch {epoch}] reload-check (via load_model_from_checkpoint): val_loss={val_score:.6f}")

    # Reuse for position-only visualization
    # visualize_positions_only(
    #     model=re_model,
    #     val_loader=val_loader,
    #     device=device,
    #     out_dir=out_dir,
    #     epoch=epoch,
    #     n_examples=n_examples,
    # )
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="forests", help="Zarr dataset name or .zarr path")
    ap.add_argument("--policy-config", type=str, required=True, help="Path to Policy YAML config (required)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=100*1e-5)
    ap.add_argument("--eval", action="store_true", help="Run evaluation only on the validation split")
    ap.add_argument("--lr-scheduler", type=str, default="step", choices=["none", "step", "cosine", "plateau"], help="LR decay schedule")
    ap.add_argument("--lr-decay-rate", type=float, default=0.9, help="Gamma for StepLR/ReduceLROnPlateau")
    ap.add_argument("--lr-step-size", type=int, default=500, help="Epoch interval for StepLR")
    # KL annealing args retained but unused by Policy; kept to avoid breaking CLI
    ap.add_argument("--kl-start", type=float, default=0.0, help="Starting M_N (KL weight)")
    ap.add_argument("--kl-end", type=float, default=0.000, help="Final M_N (KL weight)")
    ap.add_argument("--kl-anneal-epochs", type=int, default=300, help="Epochs to ramp M_N from kl-start to kl-end")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for model and plots")
    args = ap.parse_args()

    # load training params from YAML and override args
    with open(args.policy_config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    training_cfg = (cfg or {}).get("training", {}) or {}

    # override CLI defaults with config values
    if "epochs" in training_cfg:
        args.epochs = int(training_cfg["epochs"])
    if "batch_size" in training_cfg:
        args.batch_size = int(training_cfg["batch_size"])
    if "learning_rate" in training_cfg:
        args.lr = float(training_cfg["learning_rate"])
    if "lr_scheduler" in training_cfg:
        args.lr_scheduler = str(training_cfg["lr_scheduler"])
    if "lr_decay_rate" in training_cfg:
        args.lr_decay_rate = float(training_cfg["lr_decay_rate"])
    if "lr_step_size" in training_cfg:
        args.lr_step_size = int(training_cfg["lr_step_size"])

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zarr_path = resolve_zarr_path(args.dataset)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(ZARR_DIR) / "models" / (zarr_path.stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "policy.pth"

    ds = build_dataset(zarr_path, cfg["model"]["policy"]["use_rotation_mat"])
    sample = ds[0]
    if not (isinstance(sample, (list, tuple)) and len(sample) == 3):
        raise RuntimeError("Dataset must yield (depth, states, future_trajectory)")
    sample_depth, sample_states, sample_future_traj = sample
    sample_shape = tuple(sample_depth.shape)

    model = create_model(sample_shape, policy_config_path=args.policy_config).to(device)
    model.print_info()
    
    # Train/val split
    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    if val_len <= 0:
        raise ValueError("val_split must produce at least 1 validation sample for evaluation.")
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)

    if args.eval:
        va_loss = eval_epoch(model, val_loader, device)
        print(f"[eval] val_loss={va_loss:.6f}")
        return

    train_hist, val_hist = [], []
    for epoch in range(1, args.epochs + 1):
        # Keep M_N knob to enforce no hidden assumptions; set to zero by default
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_ae, tr_pred = train_one_epoch(model, train_loader, optimizer, device, log_interval=args.log_interval)
        # print debug loss once per epoch (mimicking the original format)
        print(f"[iter {len(train_loader):5d}] loss={tr_loss:0.6f}, ae_loss={float(tr_ae):0.3f}, pred_loss={float(tr_pred):0.3f}")
        train_hist.append(tr_loss)
        va_loss = eval_epoch(model, val_loader, device, ds)
        val_hist.append(va_loss)
        print(f"[epoch {epoch}] train={tr_loss:.6f}  val={va_loss:.6f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"[epoch {epoch}] lr={curr_lr:.6g}")

        # Save periodic checkpoints every 50 epochs
        if epoch % 100 == 0:
            ckpt_epoch_path = out_dir / f"policy_ep{epoch:04d}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss
            }, ckpt_epoch_path)
            print(f"[epoch {epoch}] saved checkpoint: {ckpt_epoch_path.name}")

            # NEW: use your loader to validate and visualize
            evaluate_checkpoint_and_visualize(
                ckpt_epoch_path,
                args.policy_config,   # or whatever config path you pass on startup
                device,
                val_loader,
                out_dir,
                epoch,
                ds,
                n_examples=4,
            )
                
    # Save final model and curves (if desired, user can load hist arrays)
    # NEW: Plot and save training/validation loss curves
    try:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(range(1, len(train_hist) + 1), train_hist, label="train")
        if val_hist:
            ax.plot(range(1, len(val_hist) + 1), val_hist, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_path = out_dir / "loss_curve.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"Saved loss plot to {plot_path}")
    except Exception as e:
        print(f"Warning: failed to save loss plot: {e}")

    torch.save({"model_state": model.state_dict(), "epochs": args.epochs}, ckpt_path)
    print(f"Saved model to {ckpt_path}")

if __name__ == "__main__":
    main()

