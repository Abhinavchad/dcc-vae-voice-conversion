# In dcc_vae_project/train.py

import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Import our custom modules
from models import DCC_VAE_Encoder, DCC_VAE_Decoder, PatchGANDiscriminator, LatentMLPDiscriminator
# from dataset import UnalignedSpectrogramDataset
from dataset import UnalignedSpectrogramDataset, get_speaker_info

def main():
    # torch.autograd.set_detect_anomaly(True)
    # --- 1. Hyperparameters ---
    # These are crucial for training and should be tuned carefully.
    # The values here are common starting points.
    root_dir = 'data'
    num_epochs = 100
    batch_size = 1
    lr_g = 2e-5  # TO THIS (0.00002)
    lr_d = 1e-5  # TO THIS (0.00001)
    content_dim = 256
    speaker_dim = 64
    phonation_dim = 2
    num_speakers = 251 # IMPORTANT: Set this to the number of unique speakers in your dataset

    # Loss weights (the λ values from the paper)
    lambda_recon = 5.0      # Was 10.0
    lambda_kl = 0.1         # Was 1.0 - This is the most important change
    lambda_adv_spec = 1.0   # Stays the same
    lambda_adv_latent = 0.1 # Stays the same
    lambda_cyc = 2.5        # Was 5.0
    lambda_id = 2.5         # Stays the same
    # --- ADD THESE NEW HYPERPARAMETERS FOR KL WARM-UP ---
    kl_warmup_epochs = 30  # Number of epochs to gradually increase KL weight
    target_lambda_kl = 0.1   # The final value we want lambda_kl to reach
    # Logging and checkpointing
    log_step = 100
    save_epoch_freq = 10
    run_name = 'dcc_vae_experiment_1'

    # --- 2. Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(f'runs/{run_name}')
    os.makedirs(f'checkpoints/{run_name}', exist_ok=True)
    # --- ADD THIS SECTION TO AUTOMATICALLY GET SPEAKER INFO ---
    print("Discovering speaker information from dataset...")
    num_speakers, speaker_to_id_map = get_speaker_info(root_dir)
    if num_speakers == 0:
        print("Error: No speakers found. Please check your data directories and filenames.")
        exit() # Exit if no data is found
    # --- END OF ADDED SECTION ---
    # Models
    # Models (This part is the same, but now uses the discovered num_speakers)
    encoder = DCC_VAE_Encoder(content_dim, speaker_dim).to(device)
    decoder = DCC_VAE_Decoder(content_dim, speaker_dim, phonation_dim).to(device)
    d_spec_A = PatchGANDiscriminator().to(device)
    d_spec_B = PatchGANDiscriminator().to(device)
    d_latent_spk = LatentMLPDiscriminator(content_dim, num_speakers).to(device) # Uses the discovered value

    # Optimizers (using Adam as is common for GANs)
    # The generator optimizer updates both the encoder and decoder
    optimizer_G = optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=lr_g, betas=(0.5, 0.999)
    )
    # The discriminator optimizer updates all three discriminators
    optimizer_D = optim.Adam(
        itertools.chain(d_spec_A.parameters(), d_spec_B.parameters(), d_latent_spk.parameters()),
        lr=lr_d, betas=(0.5, 0.999)
    )

    # Loss Functions
    recon_loss = nn.L1Loss() # L1 loss for reconstruction, cycle, and identity
    adv_spec_loss = nn.MSELoss() # Least-Squares GAN loss for spectrogram discriminator
    latent_loss = nn.CrossEntropyLoss() # For the latent speaker discriminator

    # Dataloader
    # --- FIX THIS LINE ---
    # Dataloader (Pass the speaker_to_id_map to the dataset)
    dataset = UnalignedSpectrogramDataset(
        root_dir=root_dir,
        speaker_to_id_map=speaker_to_id_map, # Pass the map here
        mode='train',
        max_len=128
    )
    # --- END OF FIX ---
    # Note: batch_size > 1 may require adjustments to speaker_id handling if not uniform in batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Starting Training Loop...")
    # --- 3. Training Loop ---
    for epoch in range(num_epochs):
        current_lambda_kl = min(target_lambda_kl, target_lambda_kl * (epoch / kl_warmup_epochs))
        
        # This will print the current KL weight for each epoch, so you can see it working
        print(f"--- Epoch {epoch+1}/{num_epochs} | Current lambda_kl: {current_lambda_kl:.4f} ---")
        for i, batch in enumerate(dataloader):
            # Load data and move to device
            real_A = batch['A'].to(device) # Whisper spectrogram
            real_B = batch['B'].to(device) # Normal spectrogram
            phon_A = batch['phon_A'].to(device) # Whisper phonation code
            phon_B = batch['phon_B'].to(device) # Normal phonation code
            spk_id_A = batch['speaker_id_A'].to(device)
            spk_id_B = batch['speaker_id_B'].to(device)
            if torch.isnan(real_A).any() or torch.isinf(real_A).any():
                print(f"!!!!!!!!!!!!! CORRUPT DATA DETECTED IN real_A AT BATCH {i} !!!!!!!!!!!!!")
                continue # Skip this bad batch and move to the next
            if torch.isnan(real_B).any() or torch.isinf(real_B).any():
                print(f"!!!!!!!!!!!!! CORRUPT DATA DETECTED IN real_B AT BATCH {i} !!!!!!!!!!!!!")
                continue # Skip this bad batch and move to the next
            # --- Forward pass through the models ---
            # Encode real spectrograms
            mu_c_A, logvar_c_A, mu_s_A, logvar_s_A = encoder(real_A)
            mu_c_B, logvar_c_B, mu_s_B, logvar_s_B = encoder(real_B)

            # Reparameterization trick to get latent vectors
            z_c_A = encoder.reparameterize(mu_c_A, logvar_c_A)
            z_s_A = encoder.reparameterize(mu_s_A, logvar_s_A)
            z_c_B = encoder.reparameterize(mu_c_B, logvar_c_B)
            z_s_B = encoder.reparameterize(mu_s_B, logvar_s_B)

            # Decode for reconstruction and conversion
            recon_A = decoder(z_c_A, z_s_A, phon_A)
            recon_B = decoder(z_c_B, z_s_B, phon_B)
            fake_A = decoder(z_c_B, z_s_B, phon_A) # Converted B->A
            fake_B = decoder(z_c_A, z_s_A, phon_B) # Converted A->B

            # ====================================================================
            #                      (A) Train Discriminators
            # ====================================================================
            optimizer_D.zero_grad()

            # --- D_spec Loss (Spectrogram Adversarial Loss) ---
            # Real loss: D(real) should be 1
            pred_real_A = d_spec_A(real_A)
            pred_real_B = d_spec_B(real_B)
            loss_D_real = adv_spec_loss(pred_real_A, torch.ones_like(pred_real_A)) + \
                          adv_spec_loss(pred_real_B, torch.ones_like(pred_real_B))
            # Fake loss: D(G(z)) should be 0. We detach fake_A/B to prevent gradients flowing to the generator.
            pred_fake_A = d_spec_A(fake_A.detach())
            pred_fake_B = d_spec_B(fake_B.detach())
            loss_D_fake = adv_spec_loss(pred_fake_A, torch.zeros_like(pred_fake_A)) + \
                          adv_spec_loss(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_spec = (loss_D_real + loss_D_fake) * 0.5

            # --- D_latent_spk Loss (Latent Adversarial Disentanglement Loss) ---
            # The latent discriminator tries to correctly predict the speaker ID from the content embedding
            pred_spk_A = d_latent_spk(z_c_A.detach())
            pred_spk_B = d_latent_spk(z_c_B.detach())
            loss_D_latent = latent_loss(pred_spk_A, spk_id_A) + latent_loss(pred_spk_B, spk_id_B)

            # Total discriminator loss
            loss_D = loss_D_spec + loss_D_latent
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(d_spec_A.parameters(), d_spec_B.parameters(), d_latent_spk.parameters()), 1.0)
            optimizer_D.step()

            # ====================================================================
            #                      (B) Train Generator/Encoder
            # ====================================================================
            optimizer_G.zero_grad()

            # We need to re-run the parts of the forward pass that were detached
            pred_fake_A = d_spec_A(fake_A)
            pred_fake_B = d_spec_B(fake_B)
            pred_spk_A_gen = d_latent_spk(z_c_A)
            pred_spk_B_gen = d_latent_spk(z_c_B)

            # --- 1. L_adv_spec (Generator's perspective) ---
            # The generator wants D(G(z)) to be 1 (to fool the discriminator)
            loss_G_adv_spec = adv_spec_loss(pred_fake_A, torch.ones_like(pred_fake_A)) + \
                              adv_spec_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            # --- 2. L_adv_latent (Encoder's perspective) ---
            # The encoder wants to fool the latent speaker discriminator.
            # It tries to make it predict the wrong speaker, maximizing its loss.
            loss_G_adv_latent = latent_loss(pred_spk_A_gen, spk_id_B) + latent_loss(pred_spk_B_gen, spk_id_A)
            # A common alternative is `-loss_D_latent`. This approach is also effective.

            # --- 3. L_kl (KL Divergence Loss) ---
            kl_div_A = 0.5 * torch.sum(1 + logvar_c_A - mu_c_A.pow(2) - logvar_c_A.exp()) + \
                       0.5 * torch.sum(1 + logvar_s_A - mu_s_A.pow(2) - logvar_s_A.exp())
            kl_div_B = 0.5 * torch.sum(1 + logvar_c_B - mu_c_B.pow(2) - logvar_c_B.exp()) + \
                       0.5 * torch.sum(1 + logvar_s_B - mu_s_B.pow(2) - logvar_s_B.exp())
            loss_kl = -(kl_div_A + kl_div_B) # Maximize ELBO -> Minimize negative KL

            # --- 4. L_recon (Reconstruction Loss) ---
            # Auto-encoding loss: E(x) -> D(z) should be x
            loss_recon = recon_loss(recon_A, real_A) + recon_loss(recon_B, real_B)

            # --- 5. L_id (Identity Mapping Loss) ---
            # Identity loss: G(x) should be x if domain is the same.
            # Here, we ask the decoder to reconstruct using its *own* speaker/phonation style.
            # This is essentially the same as reconstruction loss in this VAE setup.
            # In CycleGAN it's G_B(B) = B. Here, D(E(B), z_B_phon) = B.
            id_A = decoder(z_c_A, z_s_A, phon_A)
            id_B = decoder(z_c_B, z_s_B, phon_B)
            loss_id = recon_loss(id_A, real_A) + recon_loss(id_B, real_B)

            # --- 6. L_cyc_latent (Latent Cycle-Consistency Loss) ---
            # Cycle A->B->A'
            mu_c_cycled_A, _, _, _ = encoder(fake_B) # Re-encode the fake spectrogram
            z_c_cycled_A = encoder.reparameterize(mu_c_cycled_A, logvar_c_A) # Use original logvar
            loss_cycle_A = recon_loss(z_c_cycled_A, z_c_A)
            # Cycle B->A->B'
            mu_c_cycled_B, _, _, _ = encoder(fake_A)
            z_c_cycled_B = encoder.reparameterize(mu_c_cycled_B, logvar_c_B)
            loss_cycle_B = recon_loss(z_c_cycled_B, z_c_B)

            loss_cyc = loss_cycle_A + loss_cycle_B

            # --- Total Generator/Encoder Loss ---
            loss_G = (lambda_adv_spec * loss_G_adv_spec +
                      lambda_adv_latent * loss_G_adv_latent +
                      current_lambda_kl * loss_kl +  # USE THE CURRENT KL WEIGHT HERE
                      lambda_recon * loss_recon +
                      lambda_cyc * loss_cyc +
                      lambda_id * loss_id)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), decoder.parameters()), 1.0)
            optimizer_G.step()

            # --- 4. Logging ---
            # In train.py

            # --- 4. Logging ---
            if (i + 1) % log_step == 0:
                global_step = epoch * len(dataloader) + i
                
                # --- UPDATE THIS PRINT STATEMENT ---
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i+1}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[Recon: {loss_recon.item():.4f}] [KL: {loss_kl.item():.4f}] "
                      f"[Cyc: {loss_cyc.item():.4f}] [ID: {loss_id.item():.4f}] "
                      f"[AdvSpec: {loss_G_adv_spec.item():.4f}] [AdvLat: {loss_G_adv_latent.item():.4f}]")

                # Log to TensorBoard
                writer.add_scalar('loss/discriminator', loss_D.item(), global_step)
                writer.add_scalar('loss/generator_total', loss_G.item(), global_step)
                # Log individual generator loss components
                writer.add_scalar('loss_components/G_reconstruction', loss_recon.item(), global_step)
                writer.add_scalar('loss_components/G_kl_divergence', loss_kl.item(), global_step)
                writer.add_scalar('loss_components/G_cycle_latent', loss_cyc.item(), global_step)
                writer.add_scalar('loss_components/G_identity', loss_id.item(), global_step)
                writer.add_scalar('loss_components/G_adv_spec', loss_G_adv_spec.item(), global_step)
                writer.add_scalar('loss_components/G_adv_latent', loss_G_adv_latent.item(), global_step)

        # Save model checkpoints
        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(encoder.state_dict(), f'checkpoints/{run_name}/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'checkpoints/{run_name}/decoder_epoch_{epoch+1}.pth')
            print(f"Saved models at epoch {epoch+1}")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()