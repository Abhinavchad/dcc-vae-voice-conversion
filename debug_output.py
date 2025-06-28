# In dcc_vae_project/debug_output.py

import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
from models import DCC_VAE_Encoder, DCC_VAE_Decoder

def debug_inference(args):
    """
    Runs the model and saves an image of the input and output spectrograms for debugging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Debug Mode ---")
    print(f"Using device: {device}")

    # Load models
    encoder = DCC_VAE_Encoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device, weights_only=True))
    decoder = DCC_VAE_Decoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device, weights_only=True))
    encoder.eval()
    decoder.eval()

    # Load and process audio
    wav, sr = librosa.load(args.input_wav, sr=args.sampling_rate)
    wav = librosa.effects.preemphasis(wav)
    mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=args.sampling_rate, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Load normalization stats
    norm_stats = np.load('norm_stats.npz')
    spec_mean, spec_std = norm_stats['mean'], norm_stats['std']
    normalized_spec_np = (log_mel_spectrogram - spec_mean) / spec_std
    normalized_spec = torch.from_numpy(normalized_spec_np).unsqueeze(0).to(device)

    # Define the target phonation (normal speech)
    phonation_code = torch.tensor([[0.0, 1.0]], device=device) # [0,1] for normal

    # Run the model
    with torch.no_grad():
        mu_c, _, mu_s, _ = encoder(normalized_spec)
        converted_spec_normalized = decoder(mu_c, mu_s, phonation_code).squeeze(0)

    # De-normalize the output
    spec_mean_tensor = torch.tensor(spec_mean, device=device)
    spec_std_tensor = torch.tensor(spec_std, device=device)
    converted_spec = (converted_spec_normalized * spec_std_tensor) + spec_mean_tensor
    converted_spec_np = converted_spec.cpu().numpy()

    # --- Visualization ---
    print("Saving spectrogram comparison to 'debug_output.png'")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot original spectrogram
    img_orig = librosa.display.specshow(log_mel_spectrogram, sr=args.sampling_rate, x_axis='time', y_axis='mel', ax=axs[0])
    fig.colorbar(img_orig, ax=axs[0], format='%+2.0f dB')
    axs[0].set_title('Original Input Spectrogram')
    
    # Plot converted spectrogram
    img_conv = librosa.display.specshow(converted_spec_np, sr=args.sampling_rate, x_axis='time', y_axis='mel', ax=axs[1])
    fig.colorbar(img_conv, ax=axs[1], format='%+2.0f dB')
    axs[1].set_title('Model Output Spectrogram (Before Vocoder)')
    
    plt.tight_layout()
    plt.savefig('debug_output.png')
    print("Debug image saved.")

if __name__ == '__main__':
    # You'll need matplotlib: pip install matplotlib
    parser = argparse.ArgumentParser(description="Debug the DCC-VAE model by visualizing its output spectrogram.")
    # ... (argparse section is the same as inference.py)
    parser.add_argument('--input_wav', type=str, required=True, help='Path to the input .wav file.')
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help='Path to the trained encoder checkpoint (.pth).')
    parser.add_argument('--decoder_checkpoint', type=str, required=True, help='Path to the trained decoder checkpoint (.pth).')
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--content_dim', type=int, default=256)
    parser.add_argument('--speaker_dim', type=int, default=64)
    args = parser.parse_args()
    debug_inference(args)