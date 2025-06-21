import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# 假设 Block1D, Downsample1d, 和 Upsample1d 在别处已经定义。
# 为了使代码可运行，我们在这里添加简单的占位符实现。


class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Downsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class AE1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 sequence_length: int,
                 hidden_dims: Optional[List[int]] = None,
                 n_groups: int = 8,
                 **kwargs) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----------- Encoder -----------
        modules = []
        # 创建一个新的列表副本，以避免在反转时修改原始列表
        encoder_hidden_dims = list(hidden_dims)
        current_in_channels = self.in_channels

        modules.append(
            Block1D(current_in_channels, encoder_hidden_dims[0], kernel_size=3, n_groups=n_groups)
        )
        current_in_channels = encoder_hidden_dims[0]

        for h_dim in encoder_hidden_dims[1:]:
            modules.append(
                Block1D(current_in_channels, h_dim, kernel_size=3, n_groups=n_groups)
            )
            modules.append(
                Downsample1d(h_dim)
            )
            current_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.final_seq_len = sequence_length // (2 ** (len(encoder_hidden_dims) - 1))
        self.flattened_dim = encoder_hidden_dims[-1] * self.final_seq_len

        # --- 最小改动 1: 移除 log_var 层, 只保留 mu 层作为潜在向量输出 ---
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        # self.fc_log_var = nn.Linear(self.flattened_dim, latent_dim) # 移除

        # ----------- Decoder -----------
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)

        # 反转 hidden_dims 用于解码器路径
        hidden_dims.reverse()
        decoder_hidden_dims = hidden_dims

        current_in_channels = decoder_hidden_dims[0]
        for i in range(len(decoder_hidden_dims) - 1):
            modules.append(
                Upsample1d(current_in_channels)
            )
            modules.append(
                Block1D(current_in_channels, decoder_hidden_dims[i + 1], kernel_size=3, n_groups=n_groups)
            )
            current_in_channels = decoder_hidden_dims[i + 1]

        self.decoder = nn.Sequential(*modules)

        self.final_conv = nn.Conv1d(decoder_hidden_dims[-1], self.out_channels, kernel_size=3, padding=1)

        print(f"--- AE Initialized (Minimal Changes) ---")
        print(f"Hidden Dims:  {encoder_hidden_dims}")
        print(f"Latent dim:   {latent_dim}")
        print(f"Flattened dim: {self.flattened_dim}")

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        编码器现在只计算一个确定的潜在向量。
        为了保持 forward 函数的兼容性，返回一个占位符列表。
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        # --- 最小改动 2: 不再计算 log_var，返回一个零张量作为占位符 ---
        return [mu, torch.zeros_like(mu)]  # 返回 mu 和一个零张量

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧被禁用，直接返回 mu，使其成为确定性操作。
        """
        # --- 最小改动 3: 禁用随机采样 ---
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # return mu + eps * std
        return mu  # 直接返回 mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(result.size(0), -1, self.final_seq_len)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result

    def forward(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        前向传播函数结构保持不变。
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        # 返回值结构保持不变，尽管 mu 和 log_var 的含义已改变
        return [reconstruction, x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        损失函数只计算重构损失。
        """
        reconstruction = args[0]
        # original_input = args[1]
        # mu = args[2]
        # log_var = args[3]
        target = args[1]

        # --- 最小改动 4: 移除 KLD 损失 ---
        # kld_weight = kwargs.get('kld_weight', 1.0) # 移除
        mse_loss = F.mse_loss(reconstruction, target)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0) # 移除
        # loss = mse_loss + kld_weight * kld_loss # 移除
        loss = mse_loss

        # 返回的字典中也不再包含 kldloss
        return {'loss': loss, 'mseloss': mse_loss.detach()}


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
    model = AE1D(
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
    print(f"Reconstruction Loss: {loss_dict['loss']:.4f}")
