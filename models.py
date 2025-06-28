# In dcc_vae_project/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules from the Paper's References ---

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit activation function.
    This provides a data-driven gating mechanism as mentioned in the paper.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Split the tensor into two halves along the channel dimension
        # One half is the "input" and the other is the "gate"
        out, gate = torch.chunk(x, 2, dim=self.dim)
        return torch.tanh(out) * torch.sigmoid(gate)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer.
    This is the core mechanism for injecting speaker style into the decoder.
    The implementation follows the formula: AdaIN(h, z_s) = γ(z_s) * (h - μ(h)) / σ(h) + β(z_s).

    Args:
        style_dim (int): The dimensionality of the speaker style vector (z_speaker).
        num_features (int): The number of feature channels in the input tensor 'h'.
    """
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        # A small MLP to produce the scale (gamma) and bias (beta) parameters from the style vector
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        # Generate scale and bias from the style vector
        h = self.fc(style)
        # Split the output into gamma and beta
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # Reshape to match the input tensor dimensions for broadcasting
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        # Apply the affine transformation
        return (1 + gamma) * self.norm(x) + beta

# In models.py

# --- Replace the old AdaINResidualBlock with this corrected version ---
class AdaINResidualBlock(nn.Module):
    """
    A residual block that incorporates AdaIN for style modulation.
    This is the main building block of the decoder.
    (Corrected to handle GLU channel reduction).
    """
    def __init__(self, channels, style_dim):
        super().__init__()
        # The convolutional layers will now output 2x the channels,
        # which the GLU will then halve back to the correct number.
        
        # First conv block
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2)
        self.adain1 = AdaIN(style_dim, channels * 2)
        self.glu1 = GatedLinearUnit(dim=1) # Operates on channel dimension

        # Second conv block
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2)
        self.adain2 = AdaIN(style_dim, channels * 2)
        self.glu2 = GatedLinearUnit(dim=1)

    def forward(self, x, style):
        identity = x
        # First block
        out = self.conv1(x)         # -> (batch, channels * 2, seq_len)
        out = self.adain1(out, style)
        out = self.glu1(out)        # -> (batch, channels, seq_len)
        
        # Second block
        out = self.conv2(out)       # -> (batch, channels * 2, seq_len)
        out = self.adain2(out, style)
        out = self.glu2(out)        # -> (batch, channels, seq_len)
        
        # The residual connection
        return out + identity
# In models.py

# --- Replace the entire DCC_VAE_Encoder class with this stabilized version ---
class DCC_VAE_Encoder(nn.Module):
    """
    The shared-body, multi-head encoder for the DCC-VAE system.
    (Stabilized version with bounded logvar output).
    """
    def __init__(self, content_dim=256, speaker_dim=64):
        super().__init__()

        # --- Encoder Body (Shared Feature Extractor) ---
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Content Head (1D Gated CNNs) ---
        self.content_head_conv = nn.Conv1d(256 * 10, 512, kernel_size=5, padding=2) # n_mels(80)/8 = 10
        self.content_head_norm = nn.InstanceNorm1d(512)
        self.content_head_glu = GatedLinearUnit(dim=1)
        
        # --- NEW: Separate FC layers for mu and logvar ---
        self.content_fc_mu = nn.Linear(256, content_dim)
        self.content_fc_logvar = nn.Linear(256, content_dim)

        # --- Speaker Head (2D Convs + Global Average Pooling) ---
        self.speaker_head_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.speaker_head_norm = nn.InstanceNorm2d(512)

        # --- NEW: Separate FC layers for mu and logvar ---
        self.speaker_fc_mu = nn.Linear(512, speaker_dim)
        self.speaker_fc_logvar = nn.Linear(512, speaker_dim)


    def forward(self, x):
        x = x.unsqueeze(1)
        features = self.body(x)

        # --- Content Path ---
        content_in = features.view(features.size(0), -1, features.size(3))
        content_out = self.content_head_conv(content_in)
        content_out = self.content_head_norm(content_out)
        content_out = self.content_head_glu(content_out).transpose(1, 2)
        
        # --- THIS IS THE KEY FIX ---
        # Calculate mu and logvar with a bounded activation on logvar
        mu_c = self.content_fc_mu(content_out)
        logvar_c = torch.tanh(self.content_fc_logvar(content_out))

        # --- Speaker Path ---
        speaker_out = F.leaky_relu(self.speaker_head_norm(self.speaker_head_conv(features)))
        speaker_pooled = F.adaptive_avg_pool2d(speaker_out, 1).view(x.size(0), -1)
        
        # --- APPLY THE SAME FIX HERE ---
        mu_s = self.speaker_fc_mu(speaker_pooled)
        logvar_s = torch.tanh(self.speaker_fc_logvar(speaker_pooled))
        
        return mu_c, logvar_c, mu_s, logvar_s

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class DCC_VAE_Decoder(nn.Module):
    """
    Conditional generative decoder with AdaIN for style injection.
    Reconstructs a spectrogram from content, speaker, and phonation codes.
    Follows Table 1 specifications.
    """
    def __init__(self, content_dim=256, speaker_dim=64, phonation_dim=2):
        super().__init__()
        # Input layer concatenates content and phonation code 
        self.input_conv = nn.Conv1d(content_dim + phonation_dim, 512, kernel_size=5, padding=2)
        self.input_glu = GatedLinearUnit(dim=1)

        # A series of AdaIN residual blocks for styled synthesis
        self.blocks = nn.ModuleList([
            AdaINResidualBlock(channels=256, style_dim=speaker_dim),
            AdaINResidualBlock(channels=256, style_dim=speaker_dim),
            AdaINResidualBlock(channels=256, style_dim=speaker_dim),
            AdaINResidualBlock(channels=256, style_dim=speaker_dim),
        ])

        # Upsampling layers using Transposed 2D Convolutions to restore spectrogram dimensions
        self.upsample = nn.Sequential(
            # Input: (batch, 256, n_frames/8) -> (batch, 256, n_mels/8, n_frames/8) after unsqueeze
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
        )
        # Final output layer to generate the single-channel spectrogram
        self.output_conv = nn.Conv2d(32, 1, kernel_size=7, padding=3)

    def forward(self, z_content, z_speaker, z_phonation):
        # z_content: (batch, seq_len, content_dim)
        # z_speaker: (batch, speaker_dim)
        # z_phonation: (batch, phonation_dim)

        # Prepare phonation code for concatenation
        z_phonation_expanded = z_phonation.unsqueeze(1).expand(-1, z_content.size(1), -1)
        # Concatenate content and phonation codes
        z_input = torch.cat([z_content, z_phonation_expanded], dim=-1) # -> (batch, seq_len, content+phon_dim)
        z_input = z_input.transpose(1, 2) # -> (batch, content+phon_dim, seq_len)

        # Pass through input layers
        h = self.input_conv(z_input)
        h = self.input_glu(h) # -> (batch, 256, seq_len)

        # Pass through AdaIN residual blocks, injecting speaker style at each block
        for block in self.blocks:
            h = block(h, z_speaker)

        # Unsqueeze to add height dimension for 2D upsampling
        h = h.unsqueeze(2) # -> (batch, 256, 1, seq_len)
        h = F.interpolate(h, scale_factor=(10, 1)) # Scale up height to n_mels/8=10

        # Upsample to full spectrogram dimensions
        output = self.upsample(h)
        output = self.output_conv(output)

        # Squeeze the channel dimension and apply Tanh activation
        return torch.tanh(output.squeeze(1))

class PatchGANDiscriminator(nn.Module):
    """
    The PatchGAN discriminator (D_spec).
    Operates on patches of the spectrogram to determine if they are real or fake.
    Architecture is based on CycleGAN-VC2.
    """
    def __init__(self, input_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            # Output Layer
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        # Add channel dimension if it's not there
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)

class LatentMLPDiscriminator(nn.Module):
    """
    MLP discriminator for the latent space (D_latent_spk).
    Takes the content embedding z_c and tries to predict the speaker ID,
    enforcing disentanglement via an adversarial loss.
    """
    def __init__(self, content_dim, num_speakers):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(content_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_speakers)
        )

    def forward(self, z_content):
        # We might get z_content with a sequence length, so average over it.
        if z_content.dim() == 3:
            z_content = z_content.mean(dim=1)
        return self.model(z_content)