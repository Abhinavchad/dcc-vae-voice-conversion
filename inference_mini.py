# In dcc_vae_project/inference.py

import torch
import numpy as np
import librosa
import soundfile as sf
import argparse
# Make sure to import all necessary models
from models_new import DCC_VAE_Encoder, DCC_VAE_Decoder, PitchPredictor

def inference(args):
    """
    Main inference function to convert a whispered audio file to normal speech
    using the full pitch-aware architecture.
    """
    # --- 1. Setup and Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all three trained models
    print(f"Loading encoder checkpoint from: {args.encoder_checkpoint}")
    encoder = DCC_VAE_Encoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device, weights_only=True))
    
    print(f"Loading decoder checkpoint from: {args.decoder_checkpoint}")
    decoder = DCC_VAE_Decoder(content_dim=args.content_dim, speaker_dim=args.speaker_dim).to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=device, weights_only=True))
    
    print(f"Loading pitch predictor checkpoint from: {args.pitch_predictor_checkpoint}")
    pitch_predictor = PitchPredictor(args.content_dim).to(device)
    pitch_predictor.load_state_dict(torch.load(args.pitch_predictor_checkpoint, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    pitch_predictor.eval()

    # --- Load the Vocoder ---
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
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=args.sampling_rate, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Load the specified normalization stats file
    print(f"Loading normalization stats from: {args.norm_stats}")
    norm_stats = np.load(args.norm_stats)
    spec_mean, spec_std = norm_stats['mean'], norm_stats['std']
    
    normalized_spec = (log_mel_spectrogram - spec_mean) / spec_std
    normalized_spec = torch.from_numpy(normalized_spec).unsqueeze(0).to(device)

    # --- 3. Run the Full Model Pipeline ---
    with torch.no_grad():
        # Encode the whisper spectrogram
        mu_c, _, mu_s, _ = encoder(normalized_spec)
        # Predict the pitch contour from the content
        pred_f0 = pitch_predictor(mu_c)
        # Define the target phonation style (normal)
        phonation_code = torch.tensor([[0.0, 1.0]], device=device)
        # Decode using all inputs to get the converted spectrogram
        converted_spec_normalized = decoder(mu_c, mu_s, phonation_code, pred_f0).squeeze(0)

        # --- 4. De-normalize and Convert to Waveform ---
        spec_mean_tensor = torch.tensor(spec_mean, device=device)
        spec_std_tensor = torch.tensor(spec_std, device=device)
        converted_spec = (converted_spec_normalized * spec_std_tensor) + spec_mean_tensor
        
        # Use the vocoder to synthesize the audio
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
    # Add all required arguments
    parser.add_argument('--pitch_predictor_checkpoint', type=str, required=True, help='Path to the trained pitch predictor checkpoint (.pth).')
    parser.add_argument('--norm_stats', type=str, default='norm_stats.npz', help='Path to the .npz file with normalization stats.')
    
    # Other parameters
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--content_dim', type=int, default=256)
    parser.add_argument('--speaker_dim', type=int, default=64)
    args = parser.parse_args()
    inference(args)