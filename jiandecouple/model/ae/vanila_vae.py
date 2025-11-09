"""
modified from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Any
from torch import Tensor
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

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
        result = result.view(-1, 512, 2, 2)
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
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
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


if __name__ == '__main__':
    # --- 1. Define Parameters ---
    # Model parameters
    latent_dim = 128
    in_channels = 3  # For a sample RGB image

    # Data parameters
    batch_size = 4
    image_height = 64
    image_width = 64

    print("--- Test Parameters ---")
    print(f"Latent Dim: {latent_dim}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {image_height}x{image_width}")
    print("-----------------------\n")

    # --- 2. Instantiate the Model ---
    # We pass the image_size to make the linear layer calculation robust
    model = VanillaVAE(in_channels=in_channels, latent_dim=latent_dim, image_size=image_height)
    print("✅ Model instantiated successfully.")
    # print(model) # Uncomment to see the full model architecture

    # --- 3. Create Dummy Input Data ---
    # Create a random tensor with the shape [Batch, Channels, Height, Width]
    # This simulates a batch of 4 RGB images of size 64x64
    input_tensor = torch.randn(batch_size, in_channels, image_height, image_width)
    print(f"Created a dummy input tensor with shape: {input_tensor.shape}")

    # --- 4. Perform a Forward Pass ---
    try:
        results = model.forward(input_tensor)
        print("✅ Forward pass completed successfully.")

        # --- 5. Check Output Shapes ---
        recons, original_input, mu, log_var = results

        print("\n--- Output Shape Verification ---")
        print(f"  - Original Input Shape:  {original_input.shape}")
        print(f"  - Reconstructed Shape:   {recons.shape}")
        print(f"  - Mu Shape:              {mu.shape}")
        print(f"  - Log Var Shape:         {log_var.shape}")

        # Assert that the reconstructed image has the same shape as the input
        assert recons.shape == original_input.shape
        # Assert that mu and log_var have the correct shape [Batch_size, Latent_dim]
        assert mu.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)
        print("✅ All output shapes are correct.")

        # --- 6. Calculate Loss ---
        # The loss function requires the outputs from the forward pass
        # and a KLD weight (M_N)
        loss_dict = model.loss_function(*results, M_N=1.0)  # Using 1.0 as a dummy KLD weight
        loss = loss_dict['loss']

        print("\n--- Loss Calculation ---")
        print(f"Calculated Loss: {loss.item():.4f}")
        print(f"  - Reconstruction Loss: {loss_dict['Reconstruction_Loss'].item():.4f}")
        print(f"  - KLD:                 {loss_dict['KLD'].item():.4f}")
        print("✅ Loss calculated successfully.")

    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")
