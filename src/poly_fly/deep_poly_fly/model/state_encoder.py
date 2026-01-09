import copy
from typing import List, Sequence, Union, Dict, Any

import torch
import torch.nn as nn


def get_activation(spec: Union[str, nn.Module, Dict[str, Any]]) -> nn.Module:
    """
    Resolve activation spec into an nn.Module.
    - str: name of activation
    - dict: {name: <str>, params: {...}} to pass kwargs to constructor
    - nn.Module: returned as-is (deepcopied by caller when used multiple times)
    """
    if isinstance(spec, nn.Module):
        return spec

    def make(name: str, params: Dict[str, Any]) -> nn.Module:
        name = name.lower()
        if name in ("relu",):
            return nn.ReLU(**params)
        if name in ("leaky_relu", "lrelu"):
            return nn.LeakyReLU(**({"negative_slope": 0.2, "inplace": True} | params))
        if name in ("tanh",):
            return nn.Tanh()
        if name in ("sigmoid",):
            return nn.Sigmoid()
        if name in ("gelu",):
            return nn.GELU(**params)
        if name in ("elu",):
            return nn.ELU(**params)
        if name in ("selu",):
            return nn.SELU(**params)
        if name in ("prelu",):
            return nn.PReLU(**params)
        if name in ("silu", "swish"):
            return nn.SiLU(**params)
        if name in ("mish",):
            return nn.Mish(**params)
        if name in ("identity", "none", "linear"):
            return nn.Identity()
        raise ValueError(f"Unsupported activation: {name}")

    if isinstance(spec, str):
        return make(spec, {})
    if isinstance(spec, dict):
        name = spec.get("name")
        if not isinstance(name, str):
            raise ValueError("Activation dict must contain a 'name' string.")
        params = spec.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError("Activation 'params' must be a dict.")
        return make(name, params)

    raise TypeError("activation must be str, dict, or nn.Module.")


class StateEncoder(nn.Module):
    """
    A simple MLP encoder for robot states.

    - Input shape: (N, state_dim)
    - Output shape: (N, latent_dim)
    - Architecture: [Linear -> Activation]* for each hidden dim, then final Linear to latent_dim.

    Parameters:
    - state_dim: input feature dimension.
    - latent_dim: output latent dimension.
    - hidden_dims: sequence of hidden layer sizes. Must be length >= 1 to ensure at least 2 Linear layers total.
    - activation: str | dict | nn.Module (resolved via get_activation).
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: Union[str, nn.Module, Dict[str, Any]],
    ):
        super().__init__()

        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive int.")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive int.")
        if not isinstance(hidden_dims, Sequence) or len(hidden_dims) < 1:
            raise ValueError("hidden_dims must be a non-empty sequence to ensure at least two Linear layers.")

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        act = get_activation(activation)  # resolve once

        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            if not isinstance(h, int) or h <= 0:
                raise ValueError("All hidden_dims entries must be positive ints.")
            layers.append(nn.Linear(in_dim, h))
            layers.append(copy.deepcopy(act))
            in_dim = h
        # Final projection to latent space (no activation by default)
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def print_info(self) -> None:
        print(f"StateEncoder: state_dim={self.state_dim}, latent_dim={self.latent_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - x: tensor of shape (N, state_dim)
        - returns: tensor of shape (N, latent_dim)
        """
        if x.dim() != 2 or x.size(1) != self.state_dim:
            raise ValueError(f"Expected input of shape (N, {self.state_dim}), got {tuple(x.shape)}")
        return self.net(x)


if __name__ == "__main__":
    # Simple test to verify the encoder works as expected.
    torch.manual_seed(0)

    # Explicit parameters (no assumptions about user's real setup; this is a minimal runnable test).
    state_dim = 10
    latent_dim = 4
    hidden_dims = [16]  # At least one hidden layer -> total of two Linear layers including output
    activation = "relu"  # string works now

    encoder = StateEncoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )

    N = 5
    x = torch.randn(N, state_dim)
    z = encoder(x)

    assert z.shape == (N, latent_dim), f"Unexpected output shape: {z.shape}"
    print("StateEncoder test passed. Output shape:", z.shape)