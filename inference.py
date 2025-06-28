# In dcc_vae_project/inference.py (FINAL, VERIFIED FIX 4)

import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
from models import DCC_VAE_Encoder, DCC_VAE_Decoder

def inference(args):
    """
    Main inference function to convert a whispered audio file to normal speech.
    """
    # --- 1. Setup and Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained DCC-VAE models
    print(f"Loading encoder checkpoint from: {args.encoder_checkpoint}")
    encoder = DCC_VAE_Encoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device, weights_only=True))
    print(f"Loading decoder checkpoint from: {args.decoder_checkpoint}")
    decoder = DCC_VAE_Decoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()

    # --- Load the official NVIDIA HiFi-GAN vocoder ---
    print("Loading official NVIDIA HiFi-GAN vocoder...")
    torch.hub.set_dir('.')
    
    hifigan_tuple = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan', pretrained=True)
    vocoder = hifigan_tuple[0]
    vocoder.to(device)

    vocoder.eval()
    print("Vocoder loaded.")


    # --- 2. Load and Preprocess the Input Audio ---
    print(f"Loading input audio file: {args.input_wav}")
    wav, sr = librosa.load(args.input_wav, sr=args.sampling_rate)
    wav = librosa.effects.preemphasis(wav)
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Access the FFT parameters from the 'args' object
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, 
        sr=args.sampling_rate, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        n_mels=args.n_mels
    )
    # --- END OF CORRECTION ---

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Load the global normalization stats
    print("Loading normalization stats from norm_stats.npz")
    norm_stats = np.load('norm_stats.npz')
    spec_mean = norm_stats['mean']
    spec_std = norm_stats['std']
    
    normalized_spec = (log_mel_spectrogram - spec_mean) / spec_std
    normalized_spec = torch.from_numpy(normalized_spec).unsqueeze(0).to(device)

    # --- 3. Run the DCC-VAE Model ---
    with torch.no_grad():
        mu_c, _, mu_s, _ = encoder(normalized_spec)
        z_phonation_normal = torch.tensor([[0.0, 1.0]], device=device)
        converted_spec = decoder(mu_c, mu_s, z_phonation_normal).squeeze(0)

        # --- 4. De-normalize and Convert to Waveform ---
        spec_mean_tensor = torch.tensor(spec_mean, device=device)
        spec_std_tensor = torch.tensor(spec_std, device=device)
        
        converted_spec = (converted_spec * spec_std_tensor) + spec_mean_tensor
        
        audio = vocoder(converted_spec.unsqueeze(0))[0]
        final_waveform = audio.squeeze().cpu().numpy()

    # --- 5. Save the Output Audio ---
    print(f"Saving converted audio to: {args.output_wav}")
    sf.write(args.output_wav, final_waveform, args.sampling_rate)
    print("Inference complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a whispered audio file to normal speech using a trained DCC-VAE model.")
    parser.add_argument('--input_wav', type=str, required=True, help='Path to the input whispered .wav file.')
    parser.add_argument('--output_wav', type=str, required=True, help='Path to save the output converted .wav file.')
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help='Path to the trained encoder checkpoint (.pth).')
    parser.add_argument('--decoder_checkpoint', type=str, required=True, help='Path to the trained decoder checkpoint (.pth).')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Audio sampling rate.')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size.')
    parser.add_argument('--hop_length', type=int, default=256, help='FFT hop length.')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel bins.')
    parser.add_argument('--content_dim', type=int, default=256, help='Dimensionality of the content latent space.')
    parser.add_argument('--speaker_dim', type=int, default=64, help='Dimensionality of the speaker latent space.')
    args = parser.parse_args()
    inference(args)