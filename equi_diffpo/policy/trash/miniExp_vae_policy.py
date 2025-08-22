'''
因为不要大规模实验调参，这里就用固定的参数
'''

import einops
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch

from typing import Dict, List
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.ae.optimized_vae import VAE1D


class VAEPolicy(BaseImagePolicy):
    """
    A VAE policy that can be used for training and inference.
    This policy is designed to work with the DiffusionUnetHybridImagePolicy.
    """

    def __init__(self):
        super().__init__()
        IN_DIMS = 8
        OUT_DIMS = 10
        SEQUENCE_LENGTH = 16
        LATENT_DIM = 16 * 8
        model = VAE1D(
            in_dim=IN_DIMS,
            out_dim=OUT_DIMS,
            latent_dim=LATENT_DIM,
            sequence_length=SEQUENCE_LENGTH,
            hidden_dims=[512, 1024, 2048]  # 可以根据需要调整
        )

        self.model = model  # Set to eval mode by default
        self.normalizer = LinearNormalizer()

    # def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """
    #     obs_dict: must include "obs" key
    #     result: must include "action" key
    #     """
    #     assert 'past_action' not in obs_dict  # not implemented yet
    #     # normalize input
    #     nobs = self.normalizer.normalize(obs_dict)
    #     value = next(iter(nobs.values()))
    #     B, To = value.shape[:2]
    #     T = self.horizon
    #     Da = self.action_dim
    #     Do = self.obs_feature_dim
    #     To = self.n_obs_steps

    #     # build input
    #     device = self.device
    #     dtype = self.dtype

    #     vae_out, _, mu, log_var = self.model(nobs['obs'].to(device=device, dtype=dtype))
        # TODO

    def compute_loss(self, batch, *args, **kwargs):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        # if self.rot_aug:
        #     nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        nobs = nobs['JPOpen']

        nobs = einops.rearrange(nobs, 'b h t -> b t h')
        nactions = einops.rearrange(nactions, 'b h t -> b t h')
        vae_out, _, mu, log_var = self.model(nobs)
        loss = self.model.loss_function(vae_out, nobs, mu, log_var, nactions)
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
