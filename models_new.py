# File 1: models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim=self.dim)
        return torch.tanh(out) * torch.sigmoid(gate)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
    def forward(self, x, style):
        h = self.fc(style)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma.unsqueeze(2)) * self.norm(x) + beta.unsqueeze(2)

class AdaINResidualBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2)
        self.adain1 = AdaIN(style_dim, channels * 2)
        self.glu1 = GatedLinearUnit(dim=1)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2)
        self.adain2 = AdaIN(style_dim, channels * 2)
        self.glu2 = GatedLinearUnit(dim=1)
    def forward(self, x, style):
        identity = x
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.glu1(out)
        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.glu2(out)
        return out + identity

class DCC_VAE_Encoder(nn.Module):
    def __init__(self, content_dim=256, speaker_dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
        )
        self.content_head_conv = nn.Conv1d(256 * 10, 512, kernel_size=5, padding=2)
        self.content_head_norm = nn.InstanceNorm1d(512)
        self.content_head_glu = GatedLinearUnit(dim=1)
        self.content_fc_mu = nn.Linear(256, content_dim)
        self.content_fc_logvar = nn.Linear(256, content_dim)
        self.speaker_head_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.speaker_head_norm = nn.InstanceNorm2d(512)
        self.speaker_fc_mu = nn.Linear(512, speaker_dim)
        self.speaker_fc_logvar = nn.Linear(512, speaker_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        features = self.body(x)
        content_in = features.view(features.size(0), -1, features.size(3))
        content_out = self.content_head_conv(content_in)
        content_out = self.content_head_norm(content_out)
        content_out = self.content_head_glu(content_out).transpose(1, 2)
        mu_c = self.content_fc_mu(content_out)
        logvar_c = torch.tanh(self.content_fc_logvar(content_out))
        speaker_out = F.leaky_relu(self.speaker_head_norm(self.speaker_head_conv(features)))
        speaker_pooled = F.adaptive_avg_pool2d(speaker_out, 1).view(x.size(0), -1)
        mu_s = self.speaker_fc_mu(speaker_pooled)
        logvar_s = torch.tanh(self.speaker_fc_logvar(speaker_pooled))
        return mu_c, logvar_c, mu_s, logvar_s
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class PitchPredictor(nn.Module):
    def __init__(self, content_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(content_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, z_content):
        lstm_out, _ = self.lstm(z_content)
        f0_pred = self.fc(lstm_out)
        return f0_pred.squeeze(-1)

# In models.py

# In models.py (or models_new.py)

# --- REPLACE THE DECODER CLASS WITH THIS FINAL, CORRECTED VERSION ---
class DCC_VAE_Decoder(nn.Module):
    def __init__(self, content_dim=256, speaker_dim=64, phonation_dim=2):
        super().__init__()
        # An embedding layer for the F0 input
        self.f0_embedding = nn.Linear(1, content_dim)
        
        # Input layer now takes concatenated content and the F0 embedding
        self.input_conv = nn.Conv1d(content_dim * 2 + phonation_dim, 512, kernel_size=5, padding=2)
        self.input_glu = GatedLinearUnit(dim=1) # Halves channels to 256
        
        self.blocks = nn.ModuleList([
            AdaINResidualBlock(256, speaker_dim), AdaINResidualBlock(256, speaker_dim),
            AdaINResidualBlock(256, speaker_dim), AdaINResidualBlock(256, speaker_dim),
        ])
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
        )
        self.output_conv = nn.Conv2d(32, 1, kernel_size=7, padding=3)

    def forward(self, z_content, z_speaker, z_phonation, f0_contour):
        # z_content shape: (batch, 16, content_dim)
        # f0_contour shape can be (batch, 128) OR (batch, 16)

        # --- THIS IS THE KEY FIX ---
        # Make the F0 contour ready for embedding.
        # If it's the long, ground-truth version, downsample it. Otherwise, use as is.
        if f0_contour.size(1) > z_content.size(1):
            f0_reshaped = f0_contour.unsqueeze(1)
            f0_downsampled = F.interpolate(f0_reshaped, size=z_content.size(1), mode='linear', align_corners=False)
            f0_processed = f0_downsampled.transpose(1, 2)
        else:
            f0_processed = f0_contour.unsqueeze(-1)
        # --- END OF FIX ---
        
        # Now f0_processed has shape (batch, 16, 1)
        f0_embedded = F.relu(self.f0_embedding(f0_processed))

        # Now f0_embedded has shape (batch, 16, content_dim), which matches z_content
        
        z_phonation_expanded = z_phonation.unsqueeze(1).expand(-1, z_content.size(1), -1)
        
        z_input_content = torch.cat([z_content, f0_embedded], dim=-1).transpose(1, 2)
        z_input_phonation = z_phonation_expanded.transpose(1,2)
        
        z_input_full = torch.cat([z_input_content, z_input_phonation], dim=1)

        h = self.input_conv(z_input_full)
        h = self.input_glu(h)

        for block in self.blocks:
            h = block(h, z_speaker)

        h = h.unsqueeze(2)
        h = F.interpolate(h, size=(10, h.size(3)), mode='bilinear', align_corners=False)
        
        output = self.upsample(h)
        output = self.output_conv(output)
        
        return torch.tanh(output.squeeze(1))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

class LatentMLPDiscriminator(nn.Module):
    def __init__(self, content_dim, num_speakers):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(content_dim, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_speakers)
        )
    def forward(self, z_content):
        if z_content.dim() == 3:
            z_content = z_content.mean(dim=1)
        return self.model(z_content)