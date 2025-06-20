import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Downsample1d(nn.Module):
    """
    一维下采样模块，使用步长为2的卷积将序列长度减半。
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """
    一维上采样模块，使用步长为2的转置卷积将序列长度加倍。
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
    一个基础的一维卷积块，结构为: Conv1d -> GroupNorm -> Mish 激活函数。
    '''

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

# Block1D is no longer used in the refactored VAE, but kept for reference
# Block1D 在重构后的 VAE 中不再使用，但保留以供参考


class Block1D(nn.Module):
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


# ======================================================================
# Refactored VAE Implementation
# 重构后的 VAE 实现
# ======================================================================

class VAE1D(nn.Module):
    """
    一个可配置的一维卷积变分自编码器。

    通过 `hidden_dims` 参数，可以灵活地定义编码器和解码器的层数与维度。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 sequence_length: int,
                 hidden_dims: Optional[List[int]] = None,
                 n_groups: int = 8,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----------- 编码器 (Encoder) -----------
        modules = []
        if hidden_dims is None:
            # 默认的隐藏层维度
            hidden_dims = [32, 64, 128]

        # 初始卷积层
        modules.append(
            Block1D(in_channels, hidden_dims[0], kernel_size=3, n_groups=n_groups)
        )
        in_channels = hidden_dims[0]

        # 动态构建编码器层
        for h_dim in hidden_dims[1:]:
            modules.append(
                Block1D(in_channels, h_dim, kernel_size=3, n_groups=n_groups)
            )
            modules.append(
                Downsample1d(h_dim)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 动态计算展平后的维度
        # 每一次 Downsample1d 都会使序列长度减半
        self.final_seq_len = sequence_length // (2 ** (len(hidden_dims) - 1))
        self.flattened_dim = hidden_dims[-1] * self.final_seq_len

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_dim, latent_dim)

        # ----------- 解码器 (Decoder) -----------
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)

        hidden_dims.reverse()

        # 动态构建解码器层
        for i in range(len(hidden_dims) - 1):
            modules.append(
                Upsample1d(hidden_dims[i])
            )
            modules.append(
                Block1D(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, n_groups=n_groups)
            )

        self.decoder = nn.Sequential(*modules)

        # 最终输出层
        self.final_conv = nn.Conv1d(hidden_dims[-1], self.out_channels, kernel_size=3, padding=1)

        print("Configurable VAE1D Model Initialized.")
        print(f"Hidden Dims:  {hidden_dims[::-1]}")
        print(f"Latent dim:   {latent_dim}")
        print(f"Flattened dim: {self.flattened_dim}")

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """编码器前向传播，返回均值和对数方差"""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码器前向传播，从潜在向量重构输出"""
        result = self.decoder_input(z)
        result = result.view(result.size(0), -1, self.final_seq_len)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """完整的 VAE 前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return [reconstruction, x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """计算 VAE 损失"""
        vae_out = args[0]
        original_input = args[1]
        mu = args[2]
        log_var = args[3]
        target = args[4]

        # kld_weight 用于平衡重构损失和KL散度
        kld_weight = kwargs.get('kld_weight', 1.0)

        # 重构损失 (MSE)W
        # 注意：目标的形状需要与重构输出匹配
        # 这里我们假设目标是多通道的，如果不是，需要调整
        mse_loss = F.mse_loss(vae_out, target)

        # KL 散度损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        loss = mse_loss + kld_weight * kld_loss
        return {'loss': loss, 'mseloss': mse_loss.detach(), 'kldloss': kld_loss.detach()}


# ======================================================================
# Example Usage
# 使用示例
# ======================================================================
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义模型超参数
    BATCH_SIZE = 4
    INPUT_CHANNELS = 7
    OUTPUT_CHANNELS = 10
    SEQUENCE_LENGTH = 16
    LATENT_DIM = 32

    # 创建一个虚拟的输入张量
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH).to(device)

    # 实例化 VAE 模型 (使用默认的 hidden_dims: [32, 64, 128])
    model = VAE1D(
        in_channels=INPUT_CHANNELS,
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
