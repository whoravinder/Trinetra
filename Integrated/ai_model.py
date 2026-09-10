import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class UAVEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.range_cnn = nn.Sequential(
            ResidualBlock1D(1, 16),
            ResidualBlock1D(16, 32),
            nn.AdaptiveAvgPool1d(16)
        )
        self.doppler_cnn = nn.Sequential(
            ResidualBlock2D(1, 16),
            ResidualBlock2D(16, 32),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.doa_fc = nn.Sequential(
            nn.Linear(180, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion_proj = nn.Linear(32*16 + 32*8*8 + 128, 256)
        self.attn = AttentionFusion(embed_dim=256, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, range_feat, doppler_feat, doa_feat):
        r = self.range_cnn(range_feat.unsqueeze(1))
        r = r.view(r.size(0), -1)
        d = self.doppler_cnn(doppler_feat.unsqueeze(1))
        d = d.view(d.size(0), -1)
        doa = self.doa_fc(doa_feat)
        fused = torch.cat([r, d, doa], dim=1)
        fused = self.fusion_proj(fused).unsqueeze(1)
        fused = self.attn(fused).squeeze(1)
        return self.fc(fused)
