# In final_diagnostics.py

import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import pyworld as pw
from models_new import DCC_VAE_Encoder, DCC_VAE_Decoder, PitchPredictor

def get_f0(filepath, sr, hop_length):
    """Utility to extract and process F0 from a single file."""
    wav, _ = librosa.load(filepath, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=hop_length / sr * 1000)
    f0 = f0.astype(np.float32)
    if np.any(f0 > 0):
        unvoiced_indices = np.where(f0 == 0)[0]
        voiced_indices = np.where(f0 > 0)[0]
        if len(unvoiced_indices) > 0 and len(voiced_indices) > 0:
            f0[unvoiced_indices] = np.interp(unvoiced_indices, voiced_indices, f0[voiced_indices])
    return f0

def run_diagnostics(args):
    """
    Runs the full pipeline and visualizes the predicted F0 contour.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Final Diagnostics Mode ---")

    # Load all three models
    encoder = DCC_VAE_Encoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device, weights_only=True))
    pitch_predictor = PitchPredictor(args.content_dim).to(device)
    pitch_predictor.load_state_dict(torch.load(args.pitch_predictor_checkpoint, map_location=device, weights_only=True))
    encoder.eval()
    pitch_predictor.eval()

    # Load normalization stats
    norm_stats = np.load(args.norm_stats)
    spec_mean, spec_std = norm_stats['mean'], norm_stats['std']
    f0_mean, f0_std = norm_stats['f0_mean'], norm_stats['f0_std']

    # --- Process Whisper Input ---
    whisper_wav, sr = librosa.load(args.whisper_input, sr=args.sampling_rate)
    whisper_mel = librosa.feature.melspectrogram(y=whisper_wav, sr=sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    whisper_log_mel = librosa.power_to_db(whisper_mel, ref=np.max)
    normalized_spec = torch.from_numpy((whisper_log_mel - spec_mean) / spec_std).unsqueeze(0).to(device)

    # --- Get Predicted F0 ---
    with torch.no_grad():
        mu_c, _, _, _ = encoder(normalized_spec)
        pred_f0_normalized = pitch_predictor(mu_c).squeeze(0).cpu().numpy()

    # De-normalize the predicted F0 to see its real values in Hz
    pred_f0_denormalized = np.exp((pred_f0_normalized * f0_std) + f0_mean) - 1e-8

    # --- Get Ground-Truth F0 for comparison ---
    truth_f0 = get_f0(args.normal_input, args.sampling_rate, args.hop_length)
    
    # --- Visualization ---
    print(f"Saving diagnostics plot to '{args.output_image}'")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: The input whisper spectrogram
    librosa.display.specshow(whisper_log_mel, sr=sr, x_axis='time', y_axis='mel', ax=axs[0], hop_length=args.hop_length)
    axs[0].set_title('Input Whisper Spectrogram')

    # Plot 2: The predicted F0 vs the ground-truth F0
    min_len = min(len(pred_f0_denormalized), len(truth_f0))
    axs[1].plot(truth_f0[:min_len], label='Ground-Truth F0 (from normal speech)', color='green', linewidth=2)
    axs[1].plot(pred_f0_denormalized[:min_len], label='Predicted F0 (from whisper)', color='red', linestyle='--')
    axs[1].set_title('Pitch Predictor Output vs. Ground-Truth')
    axs[1].set_xlabel('Time (frames)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(args.output_image)
    print("Diagnostics complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full diagnostics on the pitch-aware model.")
    parser.add_argument('--whisper_input', type=str, required=True, help='Path to an input whisper .wav file.')
    parser.add_argument('--normal_input', type=str, required=True, help='Path to the corresponding normal speech .wav file for ground-truth F0.')
    parser.add_argument('--output_image', type=str, default='final_diagnostics.png', help='Path to save the output plot.')
    parser.add_argument('--encoder_checkpoint', type=str, required=True)
    parser.add_argument('--pitch_predictor_checkpoint', type=str, required=True)
    parser.add_argument('--norm_stats', type=str, default='norm_stats_mini.npz')
    # Add other params...
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--content_dim', type=int, default=256)
    parser.add_argument('--speaker_dim', type=int, default=64)
    args = parser.parse_args()
    run_diagnostics(args)