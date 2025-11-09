from torchinfo import summary
from jiandecouple.model.diffusion.MLP_FiLM import MLPForDiffusion
import torch.nn as nn
import torch
from jiandecouple.model.diffusion.conditional_unet1d import ConditionalUnet1D


def torchinfo_DP_C():
    model = ConditionalUnet1D(
        input_dim=10,
        local_cond_dim=None,
        global_cond_dim=274,
        diffusion_step_embed_dim=128,
        down_dims=[512, 1024, 2048],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,)
    batch_size = 64
    horizon = 16
    input_dim = 10
    cond_dim = 137  # Assuming cond_dim is 137 based on your model's
    n_obs_steps = 2  # For example, if you have 5 observation steps
    # Create dummy input tensors
    sample_input = torch.randn(batch_size, horizon, input_dim)
    timestep_input = torch.randint(0, 1000, (batch_size,))
    cond_input = torch.randn(batch_size, 2 * cond_dim)
    # Use torchinfo to generate the summary
    print('shape of cond_input:', cond_input.shape)
    print('shape of sample_input:', sample_input.shape)
    print('shape of timestep_input:', timestep_input.shape)
    summary(model, input_data={'sample': sample_input,
                               'timestep': timestep_input,
                               'global_cond': cond_input}, depth=5)


def torchinfo_DP_MLP():
    model = MLPForDiffusion(
        input_dim=10,
        output_dim=10,
        horizon=10,
        n_obs_steps=2,
        cond_dim=137,
        n_layer=8,
        n_emb=256,
        p_drop_emb=0,
        p_drop_attn=0.3,
        time_as_cond=True,
        obs_as_cond=True,
        parallel_input_emb=True  # This is crucial based on your code
    )

    # Define a batch size, horizon, input_dim, and cond_dim that match your model's initialization
    batch_size = 64
    horizon = 10
    input_dim = 10
    cond_dim = 137
    n_obs_steps = 2  # For example, if you have 5 observation steps

    # Create dummy input tensors
    sample_input = torch.randn(batch_size, horizon, input_dim)
    timestep_input = torch.randint(0, 1000, (batch_size,))
    cond_input = torch.randn(batch_size, n_obs_steps, cond_dim)

    # Use torchinfo to generate the summary
    summary(model, input_data=[sample_input, timestep_input, cond_input])


torchinfo_DP_C()
torchinfo_DP_MLP()
