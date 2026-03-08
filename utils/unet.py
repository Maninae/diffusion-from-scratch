"""U-Net architecture for diffusion models.

Self-contained module with all building blocks:
- SinusoidalTimestepEmbedding
- ResBlock (with time conditioning)
- AttentionBlock (spatial self-attention)
- DownBlock / UpBlock
- UNet (complete model)

Exported from module_04_unet.ipynb.
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Maps integer timesteps to sinusoidal positional embeddings.

    Args:
        embedding_dim: dimension of the output embedding vector
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) integer tensor of timesteps
        Returns:
            embeddings: (B, embedding_dim) float tensor
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )  # (half_dim,)
        angles = timesteps[:, None].float() * frequencies[None, :]  # (B, half_dim)
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, embedding_dim)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time conditioning and dropout.

    Structure: GroupNorm -> SiLU -> Conv -> +time -> GroupNorm -> SiLU -> Dropout -> Conv + residual

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        time_embed_dim: dimension of the time embedding vector
        dropout: dropout rate applied before the second convolution
    """

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            time_emb: (B, time_embed_dim)
        Returns:
            (B, out_channels, H, W)
        """
        residual = self.residual_proj(x)        # (B, out_channels, H, W)
        x = self.act(self.norm1(x))             # (B, in_channels, H, W)
        x = self.conv1(x)                       # (B, out_channels, H, W)
        t = self.time_proj(time_emb)            # (B, out_channels)
        x = x + t[:, :, None, None]             # (B, out_channels, H, W)
        x = self.act(self.norm2(x))             # (B, out_channels, H, W)
        x = self.dropout(x)                     # (B, out_channels, H, W)
        x = self.conv2(x)                       # (B, out_channels, H, W)
        return x + residual                     # (B, out_channels, H, W)


class AttentionBlock(nn.Module):
    """Spatial self-attention block with residual connection.

    Args:
        channels: number of input (and output) channels
        num_heads: number of attention heads
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with self-attention applied + residual
        """
        B, C, H, W = x.shape
        residual = x                                                      # (B, C, H, W)
        x = self.norm(x)                                                  # (B, C, H, W)
        qkv = self.to_qkv(x)                                              # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)    # (B, 3, heads, hd, n)
        qkv = qkv.permute(1, 0, 2, 4, 3)                                 # (3, B, heads, n, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]                                 # each: (B, heads, n, hd)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale              # (B, heads, n, n)
        attn = F.softmax(attn, dim=-1)                                    # (B, heads, n, n)
        out = torch.matmul(attn, v)                                       # (B, heads, n, hd)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)                # (B, C, H, W)
        out = self.proj_out(out)                                          # (B, C, H, W)
        return out + residual                                             # (B, C, H, W)


class DownBlock(nn.Module):
    """Encoder block: N ResBlocks + optional attention + downsample.

    Args:
        in_channels: input channels
        out_channels: output channels
        time_embed_dim: time embedding dimension
        num_res_blocks: number of ResBlocks
        use_attention: whether to apply attention after each ResBlock
        dropout: dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch_in, out_channels, time_embed_dim, dropout))
            self.attn_blocks.append(
                AttentionBlock(out_channels) if use_attention else nn.Identity()
            )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_emb)
            x = attn_block(x)
            skips.append(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    """Decoder block: upsample + N ResBlocks with skip concat + optional attention.

    Args:
        in_channels: channels from deeper level
        out_channels: desired output channels
        skip_channels: channels from encoder skip
        time_embed_dim: time embedding dimension
        num_res_blocks: number of ResBlocks
        use_attention: whether to apply attention
        dropout: dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = (out_channels + skip_channels) if i == 0 else out_channels
            self.res_blocks.append(ResBlock(ch_in, out_channels, time_embed_dim, dropout))
            self.attn_blocks.append(
                AttentionBlock(out_channels) if use_attention else nn.Identity()
            )

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor], time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        for i, (res_block, attn_block) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            if i == 0:
                x = torch.cat([x, skip], dim=1)
            else:
                x = x + skip
            x = res_block(x, time_emb)
            x = attn_block(x)
        return x


class UNet(nn.Module):
    """Complete U-Net for diffusion models with time and optional class conditioning.

    Args:
        image_channels: input/output image channels (1 for grayscale, 3 for RGB)
        base_channels: base channel count
        channel_mults: per-level channel multipliers
        num_res_blocks: ResBlocks per encoder/decoder level
        attention_resolutions: spatial resolutions where attention is applied
        dropout: dropout rate
        num_classes: enables class-conditional generation if set
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (7,),
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes
        time_embed_dim = base_channels * 4

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Optional class embedding
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes + 1, time_embed_dim)
        else:
            self.class_embedding = None

        # Initial conv
        self.initial_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch_in = base_channels
        current_res = 28

        for level, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            use_attn = (current_res // 2) in attention_resolutions
            self.down_blocks.append(
                DownBlock(ch_in, ch_out, time_embed_dim, num_res_blocks, use_attn, dropout)
            )
            ch_in = ch_out
            current_res = current_res // 2
            channels.extend([ch_out] * num_res_blocks)

        # Middle
        self.middle_res1 = ResBlock(ch_in, ch_in, time_embed_dim, dropout)
        self.middle_attn = AttentionBlock(ch_in)
        self.middle_res2 = ResBlock(ch_in, ch_in, time_embed_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channel_mults))):
            mult = channel_mults[level]
            ch_out = base_channels * mult
            skip_ch = channels[-1]  # peek at last skip's channels for this level
            decoder_res = 28 // (2 ** level)
            use_attn = decoder_res in attention_resolutions
            self.up_blocks.append(
                UpBlock(ch_in, ch_out, skip_ch, time_embed_dim, num_res_blocks, use_attn, dropout)
            )
            ch_in = ch_out
            for _ in range(num_res_blocks):
                channels.pop()

        # Final output
        self.final_norm = nn.GroupNorm(min(32, ch_in), ch_in)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch_in, image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, image_channels, H, W) noisy input
            t: (B,) integer timesteps
            class_label: (B,) optional integer class labels
        Returns:
            (B, image_channels, H, W) predicted noise
        """
        time_emb = self.time_embedding(t)
        if self.class_embedding is not None and class_label is not None:
            time_emb = time_emb + self.class_embedding(class_label)

        x = self.initial_conv(x)

        all_skips = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, time_emb)
            all_skips.extend(skips)

        x = self.middle_res1(x, time_emb)
        x = self.middle_attn(x)
        x = self.middle_res2(x, time_emb)

        for up_block in self.up_blocks:
            x = up_block(x, all_skips, time_emb)

        x = self.final_act(self.final_norm(x))
        x = self.final_conv(x)
        return x
