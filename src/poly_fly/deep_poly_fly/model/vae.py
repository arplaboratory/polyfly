from torch import nn
from torch import Tensor
from typing import List, Any, Mapping
import torch
import torch.nn.functional as F
from abc import abstractmethod

class VanillaVAE(nn.Module):
    def __init__(self,
                    in_channels: int,
                    latent_dim: int,
                    hidden_dims: List = None,
                    input_height: int = 64,
                    input_width: int = 64,
                    m_n: float = 0.0,
                    **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.m_n = m_n

        modules = []
        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [32, 32, 32, 32, 32]

        # Track spatial dims through the encoder
        def down(hw: int) -> int:
            # conv2d: kernel=3, stride=2, padding=1 => H_out = floor((H + 2 - 3)/2) + 1
            return (hw + 2 - 3) // 2 + 1

        h, w = int(input_height), int(input_width)
        in_ch = in_channels

        # Build Encoder
        for out_ch in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels=out_ch,
                        kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()))
            in_ch = out_ch
            h, w = down(h), down(w)

        self.encoder = nn.Sequential(*modules)

        # Store encoder output shape for decoder
        self.enc_out_channels = hidden_dims[-1]
        self.enc_out_h = h
        self.enc_out_w = w
        flat_dim = self.enc_out_channels * self.enc_out_h * self.enc_out_w
        print(f"VAE: final enc shape: {self.enc_out_channels} x {self.enc_out_h} x {self.enc_out_w} (flat {flat_dim})")

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flat_dim)

        hidden_dims = list(hidden_dims)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1),
                    nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                            in_channels,  # match input channels
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1))

    def print_info(self) -> None:
        in_h = self.enc_out_h * (2 ** len(self.encoder))
        in_w = self.enc_out_w * (2 ** len(self.encoder))
        print(
            f"VanillaVAE: in_channels={self.encoder[0][0].in_channels}, "
            f"latent_dim={self.latent_dim}, "
            f"input_height={in_h}, input_width={in_w}"
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VanillaVAE":
        return cls(
            in_channels=cfg["in_channels"],
            latent_dim=cfg["latent_dim"],
            hidden_dims=list(cfg["hidden_dims"]),
            input_height=cfg["input_height"],
            input_width=cfg["input_width"],
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.enc_out_channels, self.enc_out_h, self.enc_out_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                        recons, 
                        input,
                        mu,
                        log_var,
                        **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = self.m_n # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'recon_loss':recons_loss.detach(), 'kld_loss':-kld_loss.detach()}

    def sample(self,
                num_samples:int,
                current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

# example 
if __name__ == '__main__':
    vae = VanillaVAE(in_channels=1, latent_dim=128, hidden_dims=[32,32,32,32,32], input_height=64, input_width=64)
    x = torch.randn(2, 1, 64, 64)
    recons, input, mu, log_var = vae(x)
    print(recons.shape, input.shape, mu.shape, log_var.shape)