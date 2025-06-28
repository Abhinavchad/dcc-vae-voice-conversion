# In dcc_vae_project/preprocess.py (or preprocess_mini.py)

import librosa
import numpy as np
import os
import glob
import pyworld as pw

def get_audio_paths(directory):
    """Recursively finds all .wav audio files using the robust os.walk method."""
    audio_files = []
    print(f"Searching for .wav files in absolute path: {os.path.abspath(directory)}")
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found at {os.path.abspath(directory)}")
        return []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    print(f"Found {len(audio_files)} .wav files.")
    return audio_files

def process_and_extract_features(filepath, sr, n_fft, hop_length, n_mels):
    """
    Loads a single audio file, converts it to a log-mel-spectrogram,
    and extracts its F0 contour.
    """
    wav, _ = librosa.load(filepath, sr=sr)
    wav = wav.astype(np.float64)

    f0, t = pw.dio(wav, sr, frame_period=hop_length / sr * 1000)
    f0 = f0.astype(np.float32)
    
    if np.any(f0 > 0):
        unvoiced_indices = np.where(f0 == 0)[0]
        voiced_indices = np.where(f0 > 0)[0]
        
        if len(unvoiced_indices) > 0 and len(voiced_indices) > 0:
            f0[unvoiced_indices] = np.interp(unvoiced_indices, voiced_indices, f0[voiced_indices])
        
        f0 = np.log(f0 + 1e-8)

    wav = librosa.effects.preemphasis(wav.astype(np.float32))
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    min_len = min(len(f0), log_mel_spectrogram.shape[1])
    f0 = f0[:min_len]
    log_mel_spectrogram = log_mel_spectrogram[:, :min_len]

    return log_mel_spectrogram, f0

def calculate_global_stats_robust(wav_paths, sr, n_fft, hop_length, n_mels):
    """
    Calculates global mean and std using a memory-efficient two-pass algorithm.
    """
    print("Calculating global normalization stats (this may take a while)...")
    
    if not wav_paths:
        raise FileNotFoundError(f"No audio files (.wav) found in the specified directories. Please double-check the paths.")

    all_specs, all_f0s = [], []
    for i, path in enumerate(wav_paths):
        print(f"  Stats Pass: Processing file {i+1}/{len(wav_paths)}", end='\r')
        spec, f0 = process_and_extract_features(path, sr, n_fft, hop_length, n_mels)
        all_specs.append(spec)
        if np.any(f0 > 0):
            all_f0s.append(f0)
    
    full_spec = np.concatenate(all_specs, axis=1)
    full_f0 = np.concatenate(all_f0s)
    
    spec_mean, spec_std = np.mean(full_spec), np.std(full_spec)
    f0_mean, f0_std = np.mean(full_f0), np.std(full_f0)

    print(f"\nSpec Mean: {spec_mean:.4f}, Spec Std: {spec_std:.4f}")
    print(f"F0 Mean: {f0_mean:.4f}, F0 Std: {f0_std:.4f}")
    return spec_mean, spec_std, f0_mean, f0_std

def normalize_and_save(wav_paths, input_dir, output_dir, stats, sr, n_fft, hop_length, n_mels):
    """
    Normalizes and saves all spectrograms using the pre-calculated global stats.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing and saving normalized features for: {input_dir}")
    for i, filepath in enumerate(wav_paths):
        print(f"  Saving file {i+1}/{len(wav_paths)}", end='\r')
        spec, f0 = process_and_extract_features(filepath, sr, n_fft, hop_length, n_mels)
        
        normalized_spec = (spec - stats['spec_mean']) / stats['spec_std']
        
        if np.any(f0 > 0):
             normalized_f0 = (f0 - stats['f0_mean']) / stats['f0_std']
        else:
            normalized_f0 = f0

        relative_path = os.path.relpath(filepath, input_dir)
        base_filename = relative_path.replace(os.sep, '_').replace('.wav', '')
        
        np.save(os.path.join(output_dir, base_filename + '_spec.npy'), normalized_spec)
        np.save(os.path.join(output_dir, base_filename + '_f0.npy'), normalized_f0)
        
    print(f"\nFinished saving to {output_dir}")


if __name__ == '__main__':
    sampling_rate, n_fft, hop_length, n_mels = 22050, 1024, 256, 80
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Add 'raw_audio' to the beginning of the path to match your folder structure
    raw_whisper_dir = 'raw_audio/raw_audio_mini/whisper_files'
    raw_normal_dir = 'raw_audio/raw_audio_mini/normal_files'
    
    output_trainA_dir = 'data_mini/trainA'
    output_trainB_dir = 'data_mini/trainB'
    stats_file = 'norm_stats_mini.npz'

    # --- Run the robust preprocessing ---
    normal_audio_paths = get_audio_paths(raw_normal_dir)
    whisper_audio_paths = get_audio_paths(raw_whisper_dir)
    
    spec_mean, spec_std, f0_mean, f0_std = calculate_global_stats_robust(normal_audio_paths, sampling_rate, n_fft, hop_length, n_mels)
    
    np.savez(stats_file, mean=spec_mean, std=spec_std, f0_mean=f0_mean, f0_std=f0_std)
    
    normal_stats = {'spec_mean': spec_mean, 'spec_std': spec_std, 'f0_mean': f0_mean, 'f0_std': f0_std}
    whisper_stats = {'spec_mean': spec_mean, 'spec_std': spec_std, 'f0_mean': f0_mean, 'f0_std': f0_std}
    
    normalize_and_save(whisper_audio_paths, raw_whisper_dir, output_trainA_dir, whisper_stats, sampling_rate, n_fft, hop_length, n_mels)
    normalize_and_save(normal_audio_paths, raw_normal_dir, output_trainB_dir, normal_stats, sampling_rate, n_fft, hop_length, n_mels)