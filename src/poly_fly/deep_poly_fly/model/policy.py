from typing import Sequence, Union, Mapping, Any, Dict
import copy
import torch
from torch import nn, Tensor
import yaml
import torch.nn.functional as F

from poly_fly.deep_poly_fly.model.state_encoder import StateEncoder, get_activation
from poly_fly.deep_poly_fly.model.vae import VanillaVAE

class Policy(nn.Module):
    """
    Policy network that fuses state and depth-image latents to predict a sequence of actions.

    All architecture hyperparameters should be provided via config (see from_config).
    """
    def __init__(
        self,
        *,
        # State encoder config
        state_dim: int,
        state_latent_dim: int,
        state_hidden_dims: Sequence[int],
        state_activation: Union[str, nn.Module, Dict[str, Any]],
        # Image VAE encoder config
        image_in_channels: int,
        image_latent_dim: int,
        image_hidden_dims: Sequence[int],
        image_latent_mode: str,
        image_input_height: int,
        image_input_width: int,
        # Post-fusion MLP config
        combined_hidden_dims: Sequence[int],
        combined_activation: Union[str, nn.Module, Dict[str, Any]],
        # Output config
        action_dim: int,
        horizon: int,
        n_prediction_modes: int,
        prediction_head_num_linear_layers: int,
        eps: float,
        ae_loss_factor: float,  # NEW
        use_soft_assignment: bool,  # NEW
        use_rotation_mat: bool
    ):
        super().__init__()

        # Validate state encoder config
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive int.")
        if not isinstance(state_latent_dim, int) or state_latent_dim <= 0:
            raise ValueError("state_latent_dim must be a positive int.")
        if not isinstance(state_hidden_dims, Sequence) or len(state_hidden_dims) < 1:
            raise ValueError("state_hidden_dims must be a non-empty sequence.")

        # Validate image VAE config
        if not isinstance(image_in_channels, int) or image_in_channels <= 0:
            raise ValueError("image_in_channels must be a positive int.")
        if not isinstance(image_latent_dim, int) or image_latent_dim <= 0:
            raise ValueError("image_latent_dim must be a positive int.")
        if not isinstance(image_hidden_dims, Sequence) or len(image_hidden_dims) < 1:
            raise ValueError("image_hidden_dims must be a non-empty sequence.")
        if image_latent_mode not in ("mu", "sample"):
            raise ValueError("image_latent_mode must be either 'mu' or 'sample'.")
        if not isinstance(image_input_height, int) or not isinstance(image_input_width, int):
            raise ValueError("image_input_height/width must be ints.")

        # Validate post-fusion head config
        if not isinstance(combined_hidden_dims, Sequence) or len(combined_hidden_dims) < 1:
            raise ValueError("combined_hidden_dims must be a non-empty sequence.")
        self.combined_hidden_dims = combined_hidden_dims

        # Validate output config
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("action_dim must be a positive int.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive int.")

        # Validate prediction head config
        if not isinstance(prediction_head_num_linear_layers, int) or prediction_head_num_linear_layers < 1:
            raise ValueError("prediction_head_num_linear_layers must be a positive int (>= 1).")

        # Validate eps weighting for mode losses
        if not isinstance(eps, (float, int)) or not (0.0 <= float(eps) < 0.5):
            raise ValueError("eps must be a float in [0.0, 0.5).")
        self.eps = float(eps)

        # Validate AE loss factor
        if not isinstance(ae_loss_factor, (float, int)) or float(ae_loss_factor) < 0.0:
            raise ValueError("ae_loss_factor must be a non-negative float.")
        self.ae_loss_factor = float(ae_loss_factor)

        # Validate use_soft_assignment
        if not isinstance(use_soft_assignment, bool):
            raise ValueError("use_soft_assignment must be a bool.")
        self.use_soft_assignment = use_soft_assignment

        if not isinstance(use_rotation_mat, bool):
            raise ValueError("use_rotation_mat must be a bool.")
        self.use_rotation_mat = use_rotation_mat
        
        # Instantiate encoders from params
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            latent_dim=state_latent_dim,
            hidden_dims=state_hidden_dims,
            activation=state_activation,  # resolved inside StateEncoder
        )
        self.image_vae = VanillaVAE(
            in_channels=image_in_channels,
            latent_dim=image_latent_dim,
            hidden_dims=list(image_hidden_dims),
            input_height=image_input_height,
            input_width=image_input_width,
        )

        self.action_dim = action_dim
        self.n_prediction_modes = n_prediction_modes
        self.horizon = horizon
        self.image_latent_mode = image_latent_mode

        self.combined_latent_dim = self.state_encoder.latent_dim + self.image_vae.latent_dim

        # Build post-fusion MLP head
        act = get_activation(combined_activation)
        mlp_layers = []
        in_dim = self.combined_latent_dim
        for h in combined_hidden_dims:
            if not isinstance(h, int) or h <= 0:
                raise ValueError("All combined_hidden_dims entries must be positive ints.")
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(copy.deepcopy(act))
            in_dim = h
        
        self.head = nn.Sequential(*mlp_layers)

        self.prediction_heads = nn.ModuleList()
        for _ in range(self.n_prediction_modes):
            layers = []
            if prediction_head_num_linear_layers == 1:
                layers.append(nn.Linear(in_dim, horizon * action_dim))
            else:
                # (num_layers - 1) hidden linear layers with activation, then final linear to output
                for _ in range(prediction_head_num_linear_layers - 1):
                    layers.append(nn.Linear(in_dim, in_dim))
                    layers.append(copy.deepcopy(act))
                layers.append(nn.Linear(in_dim, horizon * action_dim))
            self.prediction_heads.append(nn.Sequential(*layers))
        
        # TODO Add temperature parameter?
        self.logit_heads = nn.ModuleList()
        for _ in range(self.n_prediction_modes):
            self.logit_heads.append(nn.Linear(in_dim + horizon * action_dim, 1))

        self.softmin = nn.Softmin(dim=1)

    def print_info(self) -> None:
        self.image_vae.print_info()
        self.state_encoder.print_info()
        print(
            f"Policy: state_dim={self.state_encoder.state_dim}, "
            f"state_latent_dim={self.state_encoder.latent_dim}, "
            f"image_latent_dim={self.image_vae.latent_dim}, "
            f"action_dim={self.action_dim}, "
            f"horizon={self.horizon}, "
            f"n_prediction_modes={self.n_prediction_modes}, "
            f"combined_latent_dim={self.combined_latent_dim}, "
            f"combined_hidden_dims={self.combined_hidden_dims}"
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Policy":
        """
        Build a Policy from a config dict (see config.yaml for schema).
        """
        pol = cfg["model"]["policy"]
        se = pol["state_encoder"]
        iv = pol["image_vae"]
        head = pol["head"]
        ph = pol.get("prediction_head", {})
        return cls(
            state_dim=se["state_dim"],
            state_latent_dim=se["latent_dim"],
            state_hidden_dims=se["hidden_dims"],
            state_activation=se["activation"],
            image_in_channels=iv["in_channels"],
            image_latent_dim=iv["latent_dim"],
            image_hidden_dims=iv["hidden_dims"],
            image_latent_mode=iv.get("latent_mode", "mu"),
            image_input_height=iv["input_height"],
            image_input_width=iv["input_width"],
            combined_hidden_dims=head["hidden_dims"],
            combined_activation=head["activation"],
            action_dim=pol["action_dim"],
            horizon=pol["horizon"],
            n_prediction_modes=pol["n_prediction_modes"],
            prediction_head_num_linear_layers=int(ph.get("num_linear_layers", 2)),
            eps=float(pol["eps"]),  # default 0.05
            ae_loss_factor=float(pol["ae_loss_factor"]),  # NEW
            use_soft_assignment=bool(pol["use_soft_assignment"]),  # NEW
            use_rotation_mat=bool(pol["use_rotation_mat"])
        )

    def image_encoder(self, depth_images: Tensor) -> Tensor:
        recons, input_x, mu, log_var = self.image_vae(depth_images)
        return recons, input_x, mu, log_var

    def forward(self, states: Tensor, depth_images: Tensor) -> Tensor:
        """
        - states: (N, state_dim)
        - depth_images: (N, C, H, W)
        - returns: (N, horizon * action_dim * n_prediction_modes)
        """
        # Encode modalities
        state_latent = self.state_encoder(states)
        depth_recons, depth_input, depth_latent, depth_log_var = self.image_encoder(depth_images)

        # Fuse
        combined_latent = torch.cat([state_latent, depth_latent], dim=1)

        # Shared head
        combined_latent = self.head(combined_latent)

        # Mode-specific predictions
        mode_outputs = []
        for head in self.prediction_heads:
            flat = head(combined_latent)  # (N, horizon * action_dim)
            mode_outputs.append(flat.view(-1, self.horizon, self.action_dim))  # (N, H, A)

        # Stack into (N, n_prediction_modes, H, A)
        predicted_traj = torch.stack(mode_outputs, dim=1)

        # Get logits for each mode
        logits = []
        for i, head in enumerate(self.logit_heads):
            head_input = torch.cat([combined_latent, predicted_traj[:, i, :, :].view(states.shape[0], -1)], dim=1)
            logits.append(head(head_input))  # (N, 1)
        
        logits = torch.cat(logits, dim=1)  # (N, n_prediction_modes)
        mode_probs = F.softmax(logits, dim=-1)     

        return predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var
        
    def loss_function(self, gt_traj: Tensor, predicted_traj: Tensor, mode_probs: Tensor,
        depth_recons, depth_input, image_latent, depth_log_var) -> Tensor:
        """
        compute loss function for each mode of the input prediction. 
        # predicted_traj: (N, modes, H, A)
        # gt_traj: (N, H, A)
        # mode_probs: (N, modes)
        Returns a scalar final loss.
        """
        mode_losses = [
            self.mode_loss_function(predicted_traj[:, j, :, :], gt_traj)  # (N,)
            for j in range(predicted_traj.size(1))
        ]
        losses = torch.stack(mode_losses, dim=1)  # (N, modes)
        assert losses.shape == mode_probs.shape 

        distances = [
            self.traj_distance_metric(predicted_traj[:, j, :, :], gt_traj)  # (N,)
            for j in range(predicted_traj.size(1))
        ]
        distances = torch.stack(distances, dim=1)  # (N, modes)
        assert distances.shape == mode_probs.shape 

        closest_indices = torch.argmin(distances, dim=1)  # (N,)
        distance_logits = self.softmin(distances)  # (N, modes)

        if not self.use_soft_assignment:
            prediction_loss = torch.tensor(0.0, device=predicted_traj.device, dtype=losses.dtype)
            for j in range(predicted_traj.size(1)):
                mask = (closest_indices == j).float()
                if self.n_prediction_modes == 1:
                    prediction_loss += (losses[:, j] * mask).mean()
                else:
                    prediction_loss += (losses[:, j] * mask * (1 - self.eps) + losses[:, j] * (1 - mask) * self.eps/(self.n_prediction_modes - 1)).mean()
        else:
            prediction_loss = torch.tensor(0.0, device=predicted_traj.device, dtype=losses.dtype)
            for j in range(predicted_traj.size(1)):
                if self.n_prediction_modes == 1:
                    prediction_loss += (losses[:, j] * distance_logits[:, j]).mean()
                else:
                    prediction_loss += (losses[:, j] * distance_logits[:, j] * (1 - self.eps) + 
                        losses[:, j] * (1 - distance_logits[:, j]) * self.eps/(self.n_prediction_modes - 1)).mean()
                
        prediction_loss = prediction_loss / predicted_traj.size(1)        
        prediction_loss *= 500.0

        # Negative log-likelihood of the closest trajectory
        prob_loss = 0.0
        for i in range(predicted_traj.size(0)):
            prob_loss += -torch.log(mode_probs[i, closest_indices[i]] + 1e-6)
        prob_loss = prob_loss / predicted_traj.size(0)

        final_loss = prediction_loss + prob_loss

        # Get autoencoder loss for depth images
        ae_loss = self.image_vae.loss_function(depth_recons, depth_input, image_latent, depth_log_var)
        scaled_ae_loss = self.ae_loss_factor * ae_loss['loss']
        final_loss += scaled_ae_loss
        
        return {"loss": final_loss, "recon_loss": ae_loss['recon_loss'], "kld_loss": ae_loss['kld_loss'], "scaled_ae_loss": scaled_ae_loss, "prediction_loss": prediction_loss, "prob_loss": prob_loss}

    def traj_distance_metric(self, traj1: Tensor, traj2: Tensor) -> Tensor:
        """
        Compute a later-weighted distance between two trajectories.
        traj1: (N, H=10, A)
        traj2: (N, H=10, A)
        Returns: (N,)
        """
        assert traj1.shape == traj2.shape, "traj1 and traj2 must have the same shape"
        N, H, A = traj1.shape
        # assert H == 10, f"Expected H=10, got H={H}"

        # Increasing weights for later horizons (1..10), normalized to sum to 1
        h_weight = torch.ones((H)).to(traj1.device)
        # h_weight[:7] = 0
        # h_weight[7:] = 0.0
        # h_weight = (h_weight / h_weight.sum()).to(traj1.device)  # shape: (10,)

        diff = traj1 - traj2                      # (N, H, A)
        se_t = diff.pow(2).mean(dim=2)            # (N, H)  MSE over actions at each timestep
        return (se_t * h_weight).mean(dim=1)     # (N,)    weighted mean over H

    def mode_loss_function(self, pos: Tensor, future_pos: Tensor) -> Tensor:
        """
        Compute MSE loss between a single predicted trajectory and ground truth future positions.
        pos: (N, H, A)
        future_pos: (N, H, A)
        Returns: (N,)
        """
        N, H, A = pos.shape

        h_weight = torch.zeros((H)).to(pos.device)
        for i in range(H):
            h_weight[i] = (i+1)**2
            h_weight[i] = 0.1
        # h_weight[:5] = 0.25
        # h_weight[5:] = 0.75
        
        diff = pos - future_pos  # (N, H, A)
        se_t = diff.pow(2).mean(dim=2)  # (N, H)  MSE over actions at each timestep
        return (se_t * h_weight).mean(dim=1)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_policy_from_config(config_or_path: Union[str, Mapping[str, Any]]) -> Policy:
    """
    Convenience function to build a Policy from a path or dict.
    """
    cfg = load_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    return Policy.from_config(cfg)


def make_optimizer_from_config(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from training config.
    """
    train_cfg = cfg.get("training", {})
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        nesterov = bool(train_cfg.get("nesterov", True))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unsupported optimizer: {opt_name}")

def load_model_from_checkpoint(path_chkpt, path_config, device):
    """
    Load a Policy from a checkpoint saved as:
      torch.save({"model_state": model.state_dict(), "epochs": ...}, path_chkpt)

    The policy architecture is built from the provided YAML config at path_config.
    """
    ckpt = torch.load(path_chkpt, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("Checkpoint must be a dict with key 'model_state'.")

    cfg = load_config(path_config)
    model = Policy.from_config(cfg)

    state_dict = ckpt["model_state"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

def test_load_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(
        "/home/mrunal/Documents/poly_fly/data/zarr/models/forests/policy.pth",
        "/home/mrunal/Documents/poly_fly/src/poly_fly/deep_poly_fly/model/config.yaml",
        device,
    )
    model.print_info()

if __name__ == "__main__":
    test_load_checkpoint()

    raise Exception("Exit after test_load_checkpoint")
    # Simple test
    import numpy as np
    cfg = load_config("config.yaml")
    policy = Policy.from_config(cfg)
    policy.print_info()

    N = 4
    C = cfg["model"]["policy"]["image_vae"]["in_channels"]
    H = cfg["model"]["policy"]["image_vae"]["input_height"]
    W = cfg["model"]["policy"]["image_vae"]["input_width"]
    state_dim = cfg["model"]["policy"]["state_encoder"]["state_dim"]
    action_dim = cfg["model"]["policy"]["action_dim"]
    horizon = cfg["model"]["policy"]["horizon"]
    modes = cfg["model"]["policy"]["n_prediction_modes"]

    states = torch.from_numpy(np.random.randn(N, state_dim).astype(np.float32))
    depth_images = torch.from_numpy(np.random.randn(N, C, H, W).astype(np.float32))
    future_traj = torch.from_numpy(np.random.randn(N, horizon, action_dim).astype(np.float32))

    predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var = policy(states, depth_images)
    print(f"predicted_traj: {predicted_traj.shape}, mode_probs: {mode_probs.shape}")  # (N, modes, H, A), (N, modes)

    all_losses = policy.loss_function(future_traj, predicted_traj, mode_probs, depth_recons, depth_input, depth_latent, depth_log_var)
    loss = all_losses['loss']
    print(f"loss: {loss.item()}")