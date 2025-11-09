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
    def __init__(self, input_dim, hidden_dims, n_groups=8):
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

    def forward(self, nobs):
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
    def __init__(self, hidden_dim, latent_dim, out_dim, final_seq_len, n_groups=8, flattened_dim=None,):
        super().__init__()
        self.flattened_dim = flattened_dim if flattened_dim is not None else hidden_dim[-1] * final_seq_len
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)

        hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            modules.append(
                Upsample1d(hidden_dim[i])
            )
            modules.append(
                Block1D(hidden_dim[i], hidden_dim[i + 1], kernel_size=3, n_groups=n_groups)
            )

        self.decoder = nn.Sequential(*modules)

        self.final_conv = nn.Conv1d(hidden_dim[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, z, final_seq_len):
        result = self.decoder_input(z)
        result = result.view(result.size(0), -1, final_seq_len)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result


class AE1D_Stage1(nn.Module):
    def __init__(self,
                 input_dim: int,
                 out_channels: int,
                 latent_dim: int,
                 sequence_length: int,
                 hidden_dims: Optional[List[int]] = None,
                 n_groups: int = 8,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.out_channels = out_channels

        hidden_dims = hidden_dims if hidden_dims is not None else [32, 64, 128]

        self.encoder = AEEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_groups=n_groups
        )

        # 每一次 Downsample1d 都会使序列长度减半
        self.final_seq_len = sequence_length // (2 ** (len(hidden_dims) - 1))
        self.flattened_dim = hidden_dims[-1] * self.final_seq_len

        self.fc_latent = nn.Linear(self.flattened_dim, latent_dim)

        # ----------- Decoder -----------
        self.decoder = AEDecoder(
            hidden_dim=hidden_dims,
            latent_dim=latent_dim,
            final_seq_len=self.final_seq_len,
            n_groups=n_groups,
            flattened_dim=self.flattened_dim
        )

        print(f"Hidden Dims:  {hidden_dims[::-1]}")
        print(f"Latent dim:   {latent_dim}")
        print(f"Flattened dim: {self.flattened_dim}")

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z, self.final_seq_len)
        return result

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return [reconstruction, x, z]

    def loss_function(self, ae_out, target) -> dict:
        mse_loss = F.mse_loss(ae_out, target)
        return mse_loss


class AE1D_Stage2(nn.Module):
    def __init__(self, input_dim: int, out_channels: int, latent_dim: int, sequence_length: int,
                 hidden_dims: Optional[List[int]] = None, n_groups: int = 8, **kwargs) -> None:
        super().__init__()
        # ----------- Encoder -----------

        # ----------- Decoder -----------
        self.decoder = AEDecoder(
            hidden_dim=hidden_dims,
            latent_dim=latent_dim,
            final_seq_len=self.final_seq_len,
            n_groups=n_groups,
            flattened_dim=self.flattened_dim
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z, self.final_seq_len)
        return result

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return [reconstruction, x, z]

    def loss_function(self, ae_out, target) -> dict:
        mse_loss = F.mse_loss(ae_out, target)
        return mse_loss


# endregion


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义模型超参数
    BATCH_SIZE = 4
    INPUT_CHANNELS = 8
    OUTPUT_CHANNELS = 10
    SEQUENCE_LENGTH = 16
    LATENT_DIM = 32

    # 创建一个虚拟的输入张量
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH).to(device)

    # 实例化 VAE 模型 (使用默认的 hidden_dims: [32, 64, 128])
    model = AE1D_Stage1(
        input_dim=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        latent_dim=LATENT_DIM,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dims=[512, 1024, 2048]  # 可以根据需要调整
    ).to(device)

    # 通过模型进行前向传播
    # forward 返回 [reconstruction, original_input, mu, log_var]
    results = model(dummy_input)
    reconstructed_output = results[0]

    # 打印输入和输出的形状以验证
    print("\n--- Verification ---")
    print(f"Input tensor shape:          {dummy_input.shape}")
    print(f"Reconstructed output shape:  {reconstructed_output.shape}")
    print(f"Mu shape:                    {results[2].shape}")
    print(f"Log Var shape:               {results[3].shape}")

    # 验证形状是否符合预期
    assert dummy_input.shape == (BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH)
    assert reconstructed_output.shape == (BATCH_SIZE, OUTPUT_CHANNELS, SEQUENCE_LENGTH)
    assert results[2].shape == (BATCH_SIZE, LATENT_DIM)
    assert results[3].shape == (BATCH_SIZE, LATENT_DIM)
    print("\nAll shapes are correct!")

    # 使用模型内置的 loss_function 计算损失
    loss_dict = model.loss_function(*results, kld_weight=0.005)
    print("\n--- Loss Calculation ---")
    print(f"Total Loss: {loss_dict['loss']:.4f}")
    print(f"Reconstruction Loss: {loss_dict['Reconstruction_Loss']:.4f}")
    print(f"KLD Loss: {loss_dict['KLD']:.4f}")
