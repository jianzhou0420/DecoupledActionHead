'''
因为不要大规模实验调参，这里就用固定的参数
'''

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch


class PolicyVAE(BaseImagePolicy):
    """
    A VAE policy that can be used for training and inference.
    This policy is designed to work with the DiffusionUnetHybridImagePolicy.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vae.eval()  # Set to eval mode by default

    def predict_action(self, obs_dict):
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        # Assuming obs_dict contains the necessary input for the VAE
        with torch.no_grad():
            actions = self.forward(obs_dict['obs'])
        return {'actions': actions}

    def compute_loss(self, batch):
        """
        Compute the loss for the VAE policy.
        """
        return self.vae.compute_loss(batch)
