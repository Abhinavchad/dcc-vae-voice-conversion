# Unsupervised Whispered-to-Normal Speech Conversion using a Pitch-Aware DCC-VAE

This project is an implementation of a novel unsupervised framework for converting whispered speech into normal (phonated) speech. [cite_start]The core of the project is a **Disentangled Cycle-Consistent Variational Autoencoder (DCC-VAE)**, a hybrid model that synergizes the strengths of Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs)[cite: 6, 35].

[cite_start]The primary challenge in this task is synthesizing a realistic fundamental frequency (F0 or pitch), which is entirely absent in the whispered source audio[cite: 4, 17]. This implementation addresses that by explicitly predicting the F0 contour and using it to condition the synthesis process. [cite_start]The model is designed to work with non-parallel corpora, meaning it does not require paired whisper-normal recordings, which are expensive and difficult to obtain[cite: 25, 27].

## Key Features

* [cite_start]**Unsupervised Training:** The model is trained on two separate, non-parallel sets of data: a collection of whispered speech and a collection of normal speech[cite: 28].
* [cite_start]**Hybrid VAE-GAN Architecture:** Integrates the structured representation learning of VAEs with the high-fidelity synthesis capabilities of GANs[cite: 35].
* [cite_start]**Tri-Factor Disentanglement:** The speech signal is explicitly decomposed into three independent latent representations[cite: 36, 37]:
    * [cite_start]**Content (`z_content`):** Phonetic and linguistic information[cite: 38].
    * [cite_start]**Speaker Identity (`z_speaker`):** The unique vocal characteristics of the speaker[cite: 39].
    * [cite_start]**Phonation Style (`z_phonation`):** A code representing whisper or normal phonation[cite: 40].
* **Pitch-Aware Synthesis:** A dedicated `PitchPredictor` network first predicts a plausible F0 contour from the whisper's content embedding. The Decoder then uses this predicted pitch to condition the generation, dramatically simplifying the synthesis task.
* [cite_start]**Latent Cycle-Consistency Loss:** To ensure robust preservation of linguistic content, cycle consistency is enforced in the low-dimensional latent space rather than the high-dimensional spectrogram space[cite: 181, 182].

## Architecture Overview

The pipeline consists of several key neural network components:

1.  [cite_start]**Multi-Head Encoder:** A single encoder body with specialized heads to disentangle the input spectrogram into `z_content` and `z_speaker`[cite: 121].Instance Normalization is used in the content encoder to help remove speaker-specific statistics.
2.  **Pitch Predictor:** An LSTM-based network that takes the `z_content` embedding and predicts a corresponding F0 contour.
3.  **Conditional Decoder:** A generative network that synthesizes the output spectrogram conditioned on the content, speaker, phonation style, and predicted pitch.It uses **Adaptive Instance Normalization (AdaIN)** to inject the speaker style (`z_speaker`) at multiple levels of the synthesis process
4.  **Dual Discriminators:**
    * [cite_start]**Spectrogram Discriminator (`D_spec`):** A PatchGAN discriminator that enforces the realism of the generated spectrogram output[cite: 145].
    * [cite_start]**Latent Discriminator (`D_latent_spk`):** An MLP that works on the content embedding (`z_content`) to adversarially purge it of speaker information, improving disentanglement.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhinavchad/dcc-vae-voice-conversion.git
    cd your-repo-name
    ```

2.  **Create and activate a Conda environment:** The project was developed using Python 3.9.
    ```bash
    conda create --name dcc_vae python=3.9
    conda activate dcc_vae
    ```

3.  **Install dependencies:** Install all required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Git LFS:** This project uses Git LFS to handle large audio and model files.
    ```bash
    git lfs install
    git lfs pull
    ```

## Usage Workflow

### 1. Data Preparation

1.  Place your raw normal speech audio files (e.g., LibriSpeech) inside the `raw_audio/raw_audio/normal_files/` directory.
2.  Place your raw whispered speech audio files inside the `raw_audio/raw_audio/whisper_files/` directory.
3.  Run the preprocessing script. This will analyze all audio, extract spectrograms and F0 contours, compute normalization statistics, and save the processed features into the `data/` directory.
    ```bash
    python preprocess_mini.py
    ```

### 2. Training

1.  The training script `train_mini.py` is configured with hyperparameters that were found to be stable, including a KL-divergence warm-up schedule.
2.  To start training, run:
    ```bash
    python train_mini.py
    ```
3.  You can monitor the training progress using TensorBoard:
    ```bash
    tensorboard --logdir runs
    ```

### 3. Inference

Once a model is trained, checkpoints will be saved in the `checkpoints/` directory. You can use the `inference_mini.py` script to convert a new whisper file.

```bash
python inference_mini.py `
    --input_wav "path/to/your/whisper.wav" `
    --output_wav "converted_speech.wav" `
    --encoder_checkpoint "checkpoints/YOUR_RUN_NAME/encoder_epoch_XXX.pth" `
    --decoder_checkpoint "checkpoints/YOUR_RUN_NAME/decoder_epoch_XXX.pth" `
    --pitch_predictor_checkpoint "checkpoints/YOUR_RUN_NAME/pitch_predictor_epoch_XXX.pth" `
    --norm_stats "norm_stats_mini.npz"
```
