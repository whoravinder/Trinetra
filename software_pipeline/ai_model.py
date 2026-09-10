import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class AttentionFusion(nn.Module):
    """Cross-modal attention over range, Doppler and DOA feature tokens."""

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        return self.norm2(tokens + self.ffn(tokens))


class UAVEstimator(nn.Module):
    """Multimodal UAS parameter estimator: matched-filter range + Doppler + array DOA."""

    def __init__(self):
        super().__init__()
        self.range_cnn = nn.Sequential(
            ResidualBlock1D(1, 16),
            ResidualBlock1D(16, 32),
            nn.AdaptiveAvgPool1d(16),
        )
        # Doppler is now a 1-D spectrum, so use a 1-D CNN rather than a 2-D image CNN.
        self.doppler_cnn = nn.Sequential(
            ResidualBlock1D(1, 16),
            ResidualBlock1D(16, 32),
            nn.AdaptiveAvgPool1d(16),
        )

        self.range_proj = nn.Linear(32 * 16, 128)
        self.doppler_proj = nn.Linear(32 * 16, 128)
        self.doa_proj = nn.Sequential(
            nn.Linear(181, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fusion = AttentionFusion(embed_dim=128, num_heads=4)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, range_feat, doppler_feat, doa_feat):
        r = self.range_cnn(range_feat.unsqueeze(1)).flatten(1)
        d = self.doppler_cnn(doppler_feat.unsqueeze(1)).flatten(1)
        r = self.range_proj(r)
        d = self.doppler_proj(d)
        doa = self.doa_proj(doa_feat)

        # Three tokens make attention meaningful: one token per sensing modality.
        tokens = torch.stack([r, d, doa], dim=1)
        fused = self.fusion(tokens).mean(dim=1)
        return self.head(fused)
