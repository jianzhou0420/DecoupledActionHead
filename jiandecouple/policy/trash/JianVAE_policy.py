'''
因为不要大规模实验调参，这里就用固定的参数
'''

import einops
from jiandecouple.policy.base_image_policy import BaseImagePolicy
import torch

from typing import Dict, List
from jiandecouple.model.common.normalizer import LinearNormalizer
from jiandecouple.model.ae.designed_ae import AEDecoder
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

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


class JianAEPolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 obs_as_global_cond=True,
                 crop_shape=(76, 76),
                 down_dims=(256, 512, 1024),
                 n_groups=8,
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 z_dim=64,  # z_dim
                 # parameters passed to step
                 **kwargs):
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

        # 1. encoder: convert obs to features vector
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

        # 2. proj_obs_feature_2_latent
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        obs_features_dim = obs_feature_dim * n_obs_steps

        self.project_obs_feature_2_latent = nn.Sequential(
            nn.Linear(obs_features_dim, z_dim),
        )

        # TODO:
        # 3. decoder: convert latent to action

        decoder = AEDecoder(
            out_dim=action_dim,
            horizon=horizon,
            hidden_dims=down_dims,
            z_dim=z_dim,
            n_groups=8,
        )

        self.deocder = decoder

        # 4. Others
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

    def compute_loss(self, batch, *args, **kwargs):
        # normalize input
        nobs: Dict = self.normalizer.normalize(batch['obs'])
        batch_size = nobs[list(nobs.keys())[0]].shape[0]

        # 1. encode obs
        this_nobs = dict_apply(nobs,
                               lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        obs_features = obs_features.reshape(batch_size, -1)

        # 2. project obs features to latent space
        z = self.project_obs_feature_2_latent(obs_features)

        # 3. decode latent to action
        out = self.deocder(z)

        out = einops.rearrange(out, 'b t h -> b h t')

        nactions = self.normalizer['action'].normalize(batch['action'])

        # loss_function
        loss = F.mse_loss(out, nactions, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs: Dict = self.normalizer.normalize(obs_dict)
        batch_size = nobs[list(nobs.keys())[0]].shape[0]

        # 1. encode obs
        this_nobs = dict_apply(nobs,
                               lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        obs_features = obs_features.reshape(batch_size, -1)

        # 2. project obs features to latent space
        z = self.project_obs_feature_2_latent(obs_features)

        # 3. decode latent to action
        out = self.deocder(z)

        out = einops.rearrange(out, 'b t h -> b h t')
        out = self.normalizer['action'].unnormalize(out)
        return {'action': out[:, :self.n_action_steps, :],
                'action_pred': out}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
