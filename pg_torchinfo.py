from torchinfo import summary
from equi_diffpo.model.diffusion.MLP_FiLM import MLPForDiffusion
import torch.nn as nn
import torch

model = MLPForDiffusion(
    input_dim=10,
    output_dim=10,
    horizon=10,
    n_obs_steps=2,
    cond_dim=137,
    n_layer=8,
    n_emb=64,
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
