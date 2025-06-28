# In dcc_vae_project/preprocess.py (ROBUST FINAL VERSION)

import librosa
import numpy as np
import os
import glob

def get_wav_paths(directory):
    """Recursively find all .wav files."""
    return glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)

def process_wav(filepath, sr, n_fft, hop_length, n_mels):
    """Loads a single wav file and converts it to a log-mel-spectrogram."""
    wav, _ = librosa.load(filepath, sr=sr)
    wav = librosa.effects.preemphasis(wav)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def calculate_global_stats_robust(wav_paths, sr, n_fft, hop_length, n_mels):
    """
    Calculates global mean and std using a memory-efficient two-pass algorithm.
    """
    print("Calculating global normalization stats (this may take a while)...")
    
    # --- First Pass: Calculate Mean ---
    total_sum = 0
    total_elements = 0
    for i, path in enumerate(wav_paths):
        print(f"  Pass 1/2: Processing file {i+1}/{len(wav_paths)}", end='\r')
        spec = process_wav(path, sr, n_fft, hop_length, n_mels)
        total_sum += np.sum(spec)
        total_elements += spec.size
    mean = total_sum / total_elements
    print("\nPass 1 complete. Global Mean:", mean)

    # --- Second Pass: Calculate Standard Deviation ---
    sum_of_squared_diff = 0
    for i, path in enumerate(wav_paths):
        print(f"  Pass 2/2: Processing file {i+1}/{len(wav_paths)}", end='\r')
        spec = process_wav(path, sr, n_fft, hop_length, n_mels)
        sum_of_squared_diff += np.sum((spec - mean) ** 2)
    std = np.sqrt(sum_of_squared_diff / total_elements)
    print("\nPass 2 complete. Global Std:", std)
    
    return mean, std

def normalize_and_save(wav_paths, input_dir, output_dir, mean, std, sr, n_fft, hop_length, n_mels):
    """
    Normalizes and saves all spectrograms using the pre-calculated global stats.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing and saving normalized spectrograms for: {input_dir}")
    for i, filepath in enumerate(wav_paths):
        print(f"  Saving file {i+1}/{len(wav_paths)}", end='\r')
        spec = process_wav(filepath, sr, n_fft, hop_length, n_mels)
        normalized_spec = (spec - mean) / std

        relative_path = os.path.relpath(filepath, input_dir)
        output_filename = relative_path.replace(os.sep, '_').replace('.wav', '.npy')
        np.save(os.path.join(output_dir, output_filename), normalized_spec)
    print(f"\nFinished saving to {output_dir}")

if __name__ == '__main__':
    # --- Define parameters ---
    sampling_rate = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    
    raw_whisper_dir = 'raw_audio/whisper_files'
    raw_normal_dir = 'raw_audio/normal_files'
    
    output_trainA_dir = 'data/trainA' # Whisper
    output_trainB_dir = 'data/trainB' # Normal

    # --- Run the robust preprocessing ---
    # 1. Get all file paths first
    normal_wav_paths = get_wav_paths(raw_normal_dir)
    whisper_wav_paths = get_wav_paths(raw_whisper_dir)
    
    # 2. Calculate stats ONLY on the normal training data (LibriSpeech)
    global_mean, global_std = calculate_global_stats_robust(normal_wav_paths, sampling_rate, n_fft, hop_length, n_mels)
    
    # 3. Save stats for later use in inference
    np.savez('norm_stats.npz', mean=global_mean, std=global_std)
    
    # 4. Process and save both datasets using these global stats
    normalize_and_save(whisper_wav_paths, raw_whisper_dir, output_trainA_dir, global_mean, global_std, sampling_rate, n_fft, hop_length, n_mels)
    normalize_and_save(normal_wav_paths, raw_normal_dir, output_trainB_dir, global_mean, global_std, sampling_rate, n_fft, hop_length, n_mels)