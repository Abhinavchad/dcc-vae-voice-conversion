# In count_parameters.py

import torch
from models_new import (
    DCC_VAE_Encoder,
    DCC_VAE_Decoder,
    PitchPredictor,
    PatchGANDiscriminator,
    LatentMLPDiscriminator
)

def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Initializes all models and prints their parameter counts."""
    # --- Model Hyperparameters (must match your training script) ---
    content_dim = 256
    speaker_dim = 64
    phonation_dim = 2
    num_speakers = 4 # From our mini-dataset run

    # --- Initialize all model components ---
    encoder = DCC_VAE_Encoder(content_dim, speaker_dim)
    decoder = DCC_VAE_Decoder(content_dim, speaker_dim, phonation_dim)
    pitch_predictor = PitchPredictor(content_dim)
    d_spec = PatchGANDiscriminator()
    d_latent_spk = LatentMLPDiscriminator(content_dim, num_speakers)

    # --- Count parameters for each component ---
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    pitch_predictor_params = count_parameters(pitch_predictor)
    d_spec_params = count_parameters(d_spec)
    d_latent_spk_params = count_parameters(d_latent_spk)

    # --- Calculate totals ---
    # The Generator-side models are updated by optimizer_G
    total_generator_params = encoder_params + decoder_params + pitch_predictor_params
    
    # The Discriminator-side models are updated by optimizer_D
    # We have two spectrogram discriminators (d_spec_A and d_spec_B)
    total_discriminator_params = (d_spec_params * 2) + d_latent_spk_params
    
    # The grand total for the entire pipeline
    grand_total_params = total_generator_params + total_discriminator_params

    # --- Print the results in a readable format ---
    print("-" * 50)
    print("Model Parameter Counts")
    print("-" * 50)
    print(f"DCC_VAE_Encoder:              {encoder_params:,}")
    print(f"PitchPredictor:               {pitch_predictor_params:,}")
    print(f"DCC_VAE_Decoder:              {decoder_params:,}")
    print("-" * 50)
    print(f"Total GENERATOR Parameters:   {total_generator_params:,}")
    print("-" * 50)
    print(f"PatchGANDiscriminator (x1):   {d_spec_params:,}")
    print(f"LatentMLPDiscriminator:       {d_latent_spk_params:,}")
    print("-" * 50)
    print(f"Total DISCRIMINATOR Parameters: {total_discriminator_params:,}")
    print("=" * 50)
    print(f"GRAND TOTAL Parameters:         {grand_total_params:,}")
    print("=" * 50)


if __name__ == '__main__':
    main()