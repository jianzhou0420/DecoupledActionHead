'''
因为不要大规模实验调参，这里就用固定的参数
'''

import einops
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch

from typing import Dict, List
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.ae.optimized_ae import AE1D_Stage1


class AEPolicy(BaseImagePolicy):
    """
    A VAE policy that can be used for training and inference.
    This policy is designed to work with the DiffusionUnetHybridImagePolicy.
    """

    def __init__(self,
                 in_dim: int = 8,
                 out_dim: int = 10,
                 horizon: int = 16,
                 latent_dim: int = 16 * 8,
                 hidden_dims: List[int] = None,
                 stage: str = 'stage1',):
        super().__init__()

        if stage not in ['stage1', 'stage2']:
            raise ValueError(f"Invalid stage: {stage}. Must be 'stage1' or 'stage2'.")

        if stage == 'stage1':

            model = AE1D_Stage1(
                input_dim=in_dim,
                out_channels=out_dim,
                latent_dim=latent_dim,
                sequence_length=horizon,
                hidden_dims=hidden_dims if hidden_dims is not None else [64, 128, 256],
            )
        elif stage == 'stage2':

            pass
        self.model = model  # Set to eval mode by default
        self.normalizer = LinearNormalizer()

    def compute_loss(self, batch):
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
        ae_out, x, z = self.model(nobs)
        loss = self.model.loss_function(ae_out, nactions)
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
