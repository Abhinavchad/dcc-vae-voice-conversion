# File 3: train_mini.py

import os
import itertools
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models_new import DCC_VAE_Encoder, DCC_VAE_Decoder, PitchPredictor, PatchGANDiscriminator, LatentMLPDiscriminator
from dataset_new import UnalignedSpectrogramDataset, get_speaker_info

def main():
    # --- 1. Hyperparameters ---
    root_dir = 'data_mini'
    run_name = 'dcc_vae_with_pitch_predictor_1'
    resume_epoch = 0
    num_epochs = 500
    batch_size = 1
    lr_g = 2e-5
    lr_d = 1e-5
    content_dim = 256
    speaker_dim = 64
    phonation_dim = 2

    lambda_recon = 20.0
    lambda_cyc = 10.0
    lambda_adv_spec = 0.5
    lambda_adv_latent = 0.01
    lambda_id = 5.0
    lambda_f0 = 10.0
    kl_warmup_epochs = 30
    target_lambda_kl = 0.1
    log_step = 20
    save_epoch_freq = 50

    # --- 2. Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    writer = SummaryWriter(f'runs/{run_name}')
    os.makedirs(f'checkpoints/{run_name}', exist_ok=True)
    print("Discovering speaker information from dataset...")
    num_speakers, speaker_to_id_map = get_speaker_info(root_dir)
    if num_speakers == 0:
        print("Error: No speakers found.")
        exit()

    encoder = DCC_VAE_Encoder(content_dim, speaker_dim).to(device)
    decoder = DCC_VAE_Decoder(content_dim, speaker_dim, phonation_dim).to(device)
    pitch_predictor = PitchPredictor(content_dim).to(device)
    d_spec_A = PatchGANDiscriminator().to(device)
    d_spec_B = PatchGANDiscriminator().to(device)
    d_latent_spk = LatentMLPDiscriminator(content_dim, num_speakers).to(device)
    
    optimizer_G = optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters(), pitch_predictor.parameters()),
        lr=lr_g, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        itertools.chain(d_spec_A.parameters(), d_spec_B.parameters(), d_latent_spk.parameters()),
        lr=lr_d, betas=(0.5, 0.999)
    )
    
    recon_loss = nn.L1Loss()
    adv_spec_loss = nn.MSELoss()
    latent_loss = nn.CrossEntropyLoss()
    
    dataset = UnalignedSpectrogramDataset(root_dir=root_dir, speaker_to_id_map=speaker_to_id_map, mode='train', max_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("Starting Training Loop with New Pitch-Aware Architecture...")
    # --- 3. Training Loop ---
    for epoch in range(resume_epoch, num_epochs):
        current_lambda_kl = min(target_lambda_kl, target_lambda_kl * (epoch / kl_warmup_epochs))
        print(f"--- Epoch {epoch+1}/{num_epochs} | Current lambda_kl: {current_lambda_kl:.4f} ---")

        for i, batch in enumerate(dataloader):
            real_A_spec = batch['A_spec'].to(device)
            real_B_spec = batch['B_spec'].to(device)
            real_A_f0 = batch['A_f0'].to(device)
            real_B_f0 = batch['B_f0'].to(device)
            phon_A = batch['phon_A'].to(device)
            phon_B = batch['phon_B'].to(device)
            spk_id_A = batch['speaker_id_A'].to(device)
            spk_id_B = batch['speaker_id_B'].to(device)
            
            mu_c_A, logvar_c_A, mu_s_A, logvar_s_A = encoder(real_A_spec)
            mu_c_B, logvar_c_B, mu_s_B, logvar_s_B = encoder(real_B_spec)
            z_c_A = encoder.reparameterize(mu_c_A, logvar_c_A)
            z_s_A = encoder.reparameterize(mu_s_A, logvar_s_A)
            z_c_B = encoder.reparameterize(mu_c_B, logvar_c_B)
            z_s_B = encoder.reparameterize(mu_s_B, logvar_s_B)

            pred_f0_A = pitch_predictor(z_c_A)
            pred_f0_B = pitch_predictor(z_c_B)

            recon_A = decoder(z_c_A, z_s_A, phon_A, real_A_f0)
            recon_B = decoder(z_c_B, z_s_B, phon_B, real_B_f0)
            fake_A = decoder(z_c_B, z_s_B, phon_A, real_A_f0)
            fake_B = decoder(z_c_A, z_s_A, phon_B, pred_f0_A)

            optimizer_D.zero_grad()
            pred_real_A, pred_real_B = d_spec_A(real_A_spec), d_spec_B(real_B_spec)
            loss_D_real = adv_spec_loss(pred_real_A, torch.ones_like(pred_real_A)) + adv_spec_loss(pred_real_B, torch.ones_like(pred_real_B))
            pred_fake_A, pred_fake_B = d_spec_A(fake_A.detach()), d_spec_B(fake_B.detach())
            loss_D_fake = adv_spec_loss(pred_fake_A, torch.zeros_like(pred_fake_A)) + adv_spec_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_D_spec = (loss_D_real + loss_D_fake) * 0.5
            pred_spk_A, pred_spk_B = d_latent_spk(z_c_A.detach()), d_latent_spk(z_c_B.detach())
            loss_D_latent = latent_loss(pred_spk_A, spk_id_A) + latent_loss(pred_spk_B, spk_id_B)
            loss_D = loss_D_spec + loss_D_latent
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(d_spec_A.parameters(), d_spec_B.parameters(), d_latent_spk.parameters()), 1.0)
            optimizer_D.step()

            optimizer_G.zero_grad()
            real_B_f0_downsampled = F.interpolate(real_B_f0.unsqueeze(1), size=pred_f0_B.size(1), mode='linear', align_corners=False).squeeze(1)
            # Now calculate the loss between two tensors of the same size
            loss_f0 = recon_loss(pred_f0_B, real_B_f0_downsampled)
            pred_fake_A, pred_fake_B = d_spec_A(fake_A), d_spec_B(fake_B)
            pred_spk_A_gen, pred_spk_B_gen = d_latent_spk(z_c_A), d_latent_spk(z_c_B)
            loss_G_adv_spec = adv_spec_loss(pred_fake_A, torch.ones_like(pred_fake_A)) + adv_spec_loss(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_G_adv_latent = latent_loss(pred_spk_A_gen, spk_id_B) + latent_loss(pred_spk_B_gen, spk_id_A)
            kl_div_A = 0.5 * torch.sum(1 + logvar_c_A - mu_c_A.pow(2) - logvar_c_A.exp()) + 0.5 * torch.sum(1 + logvar_s_A - mu_s_A.pow(2) - logvar_s_A.exp())
            kl_div_B = 0.5 * torch.sum(1 + logvar_c_B - mu_c_B.pow(2) - logvar_c_B.exp()) + 0.5 * torch.sum(1 + logvar_s_B - mu_s_B.pow(2) - logvar_s_B.exp())
            loss_kl = -(kl_div_A + kl_div_B)
            loss_recon = recon_loss(recon_A, real_A_spec) + recon_loss(recon_B, real_B_spec)
            id_A, id_B = decoder(z_c_A, z_s_A, phon_A, real_A_f0), decoder(z_c_B, z_s_B, phon_B, real_B_f0)
            loss_id = recon_loss(id_A, real_A_spec) + recon_loss(id_B, real_B_spec)
            mu_c_cycled_A, _, _, _ = encoder(fake_B)
            z_c_cycled_A = encoder.reparameterize(mu_c_cycled_A, logvar_c_A)
            loss_cycle_A = recon_loss(z_c_cycled_A, z_c_A)
            mu_c_cycled_B, _, _, _ = encoder(fake_A)
            z_c_cycled_B = encoder.reparameterize(mu_c_cycled_B, logvar_c_B)
            loss_cycle_B = recon_loss(z_c_cycled_B, z_c_B)
            loss_cyc = loss_cycle_A + loss_cycle_B
            loss_G = (lambda_adv_spec * loss_G_adv_spec + lambda_adv_latent * loss_G_adv_latent + current_lambda_kl * loss_kl + lambda_recon * loss_recon + lambda_cyc * loss_cyc + lambda_id * loss_id + lambda_f0 * loss_f0)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(encoder.parameters(), decoder.parameters(), pitch_predictor.parameters()), 1.0)
            optimizer_G.step()

            if (i + 1) % log_step == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i+1}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                      f"[Recon: {loss_recon.item():.4f}] [KL: {loss_kl.item():.4f}] "
                      f"[Cyc: {loss_cyc.item():.4f}] [ID: {loss_id.item():.4f}] "
                      f"[AdvSpec: {loss_G_adv_spec.item():.4f}] [AdvLat: {loss_G_adv_latent.item():.4f}] "
                      f"[F0: {loss_f0.item():.4f}]")
                writer.add_scalar('loss/discriminator', loss_D.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss/generator_total', loss_G.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_reconstruction', loss_recon.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_kl_divergence', loss_kl.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_cycle_latent', loss_cyc.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_identity', loss_id.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_adv_spec', loss_G_adv_spec.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/G_adv_latent', loss_G_adv_latent.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss_components/F0_prediction', loss_f0.item(), epoch * len(dataloader) + i)

        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(encoder.state_dict(), f'checkpoints/{run_name}/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'checkpoints/{run_name}/decoder_epoch_{epoch+1}.pth')
            torch.save(pitch_predictor.state_dict(), f'checkpoints/{run_name}/pitch_predictor_epoch_{epoch+1}.pth')
            print(f"Saved models at epoch {epoch+1}")
    
    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()