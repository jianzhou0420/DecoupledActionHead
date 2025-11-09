import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F

from jiandecouple.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from jiandecouple.model.common.normalizer import LinearNormalizer
from jiandecouple.policy.base_image_policy import BaseImagePolicy
from jiandecouple.model.diffusion.conditional_unet1d import ConditionalUnet1D
from jiandecouple.model.diffusion.mask_generator import LowdimMaskGenerator
from jiandecouple.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import jiandecouple.model.vision.crop_randomizer as dmvc
from jiandecouple.common.pytorch_util import dict_apply, replace_submodules
from jiandecouple.model.vision.rot_randomizer import RotRandomizer

# ---------------------------------------
# region: 0. Blocks
# TODO: double check


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        if out_channels < n_groups:
            n_groups = 1 if out_channels == 1 else out_channels // 2

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Block1D(nn.Module):
    """
    Diffusion Policy 的基础block，去掉FILM的，global_conditioning
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

    def forward(self, x, cond=None):
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        return out
# endregion


# ---------------------------------------
# region: 1. AE1D

class AEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_groups=8, sequence_length=16, latent_dim=32):
        super().__init__()
        modules = []
        modules.append(
            Block1D(input_dim, hidden_dims[0], kernel_size=3, n_groups=n_groups)
        )
        input_dim = hidden_dims[0]

        for h_dim in hidden_dims[1:]:
            modules.append(
                Block1D(input_dim, h_dim, kernel_size=3, n_groups=n_groups)
            )
            modules.append(
                Downsample1d(h_dim)
            )
            input_dim = h_dim
        self.encoder = nn.Sequential(*modules)

        # 每一次 Downsample1d 都会使序列长度减半
        self.final_seq_len = sequence_length // (2 ** (len(hidden_dims) - 1))
        self.flattened_dim = hidden_dims[-1] * self.final_seq_len

        self.fc_latent = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_latent(result)
        return z


class ObsEncoder(nn.Module):
    """
    accept obs and generates z
    """

    def __init__(self,
                 shape_meta: dict,
                 crop_shape=(76, 76),
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 n_obs_steps=2,
                 latent_dim=32  # dim of z
                 ):
        super().__init__()
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        self.obs_encoder = obs_encoder
        self.n_obs_steps = n_obs_steps

        # extra layer: project global condition to a fixed Z dim, TODO: refine
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps
        # global_cond : (B, 237, 128)

        self.global_cond_proj = nn.Linear(
            in_features=self.obs_encoder.output_shape['agentview_image'][0],
            out_features=latent_dim
        )

    def forward(self, nobs):
        batch_size = nobs.items()[0].shape[0]
        nobs_features = self.encode_obs(nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

    def encode_obs(self, nobs):
        """
        obs: dict, keys are obs_config['low_dim'], obs_config['rgb'], obs_config['depth'], obs_config['scan']
        """
        # encode obs
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # encoded_obs: (B, C, T)
        return nobs_features


class AEDecoder(nn.Module):
    def __init__(self,
                 out_dim: int,
                 horizon: int,
                 hidden_dims: list,
                 z_dim: int,
                 n_groups=8,
                 ):
        super().__init__()

        self.final_seq_len = horizon // (2 ** (len(hidden_dims) - 1))
        flattened_dim = hidden_dims[-1] * self.final_seq_len

        self.flattened_dim = flattened_dim if flattened_dim is not None else hidden_dims[-1] * self.final_seq_len
        modules = []

        self.decoder_input = nn.Linear(z_dim, self.flattened_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                Upsample1d(hidden_dims[i])
            )
            modules.append(
                Block1D(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, n_groups=n_groups)
            )

        self.decoder = nn.Sequential(*modules)

        self.final_conv = nn.Conv1d(hidden_dims[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(result.size(0), -1, self.final_seq_len)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result


if __name__ == "__main__":
    # Test AEEncoder
    input_dim = 3
    hidden_dims = [64, 128, 256]
    n_groups = 8
    sequence_length = 16
    latent_dim = 32

    encoder = AEEncoder(input_dim, hidden_dims, n_groups, sequence_length, latent_dim)
    x = torch.randn(2, input_dim, sequence_length)  # batch size of 2
    z = encoder(x)
    print("Encoded z shape:", z.shape)

    # Test ObsEncoder
    shape_meta = {
        'action': {'shape': [4]},
        'obs': {
            'agentview_image': {'shape': [3, 64, 64], 'type': 'rgb'},
            'low_dim_obs': {'shape': [10], 'type': 'low_dim'}
        }
    }
    obs_encoder = ObsEncoder(shape_meta)
    nobs = {
        'agentview_image': torch.randn(2, 2, 3, 64, 64),  # batch size of 2, n_obs_steps of 2
        'low_dim_obs': torch.randn(2, 2, 10)
    }
    obs_features = obs_encoder.encode_obs(nobs)
    print("Encoded obs features shape:", obs_features.shape)
