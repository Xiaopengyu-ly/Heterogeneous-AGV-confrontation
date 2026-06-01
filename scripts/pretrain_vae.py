"""Pretrain ActionVAE on uniformly sampled action space to learn a 36-dim embedding."""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vae.action_vae import ActionVAE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    action_dim = 5
    embed_dim = 36
    latent_dim = 18

    model = ActionVAE(action_dim, embed_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Generate 100k uniform random actions from [-1, 1]^5
    n_samples = 300_000
    batch_size = 512
    n_epochs = 150

    print(f"Sampling {n_samples} uniform actions from [-1, 1]^{action_dim}...")
    all_actions = torch.FloatTensor(n_samples, action_dim).uniform_(-1.0, 1.0)

    # 80/20 train/test split
    split = int(n_samples * 0.8)
    train_data = all_actions[:split]
    test_data = all_actions[split:]

    print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_data))
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for i in range(0, len(train_data), batch_size):
            idx = perm[i:i + batch_size]
            batch = train_data[idx].to(device)

            optimizer.zero_grad()
            total_loss, recon_loss, kl_loss = model.loss(batch, beta=0.1)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        if (epoch + 1) % 20 == 0:
            n_batches = (len(train_data) + batch_size - 1) // batch_size
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            print(f"Epoch {epoch + 1:3d} | loss={avg_loss:.6f} recon={avg_recon:.6f} kl={avg_kl:.6f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_batch = test_data.to(device)
        recon, embed, mu, logvar = model(test_batch)
        test_recon = nn.functional.mse_loss(recon, test_batch.to(device))

        # Per-dimension MSE
        per_dim_mse = nn.functional.mse_loss(recon, test_batch.to(device), reduction='none').mean(0)
        dim_names = ["rou_scale", "phi", "e_theta", "v_r", "w_r"]
        print(f"\nTest recon MSE: {test_recon.item():.6f}")
        for name, mse in zip(dim_names, per_dim_mse):
            print(f"  {name}: {mse.item():.6f}")

    # Save model
    save_path = "models/vae/action_vae_pretrained.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'action_dim': action_dim,
        'embed_dim': embed_dim,
        'latent_dim': latent_dim,
        'test_recon_mse': test_recon.item(),
    }, save_path)
    print(f"\nModel saved to {save_path}")

    if test_recon.item() < 0.01:
        print("PASS: reconstruction MSE < 0.01 (avg per-dim error < 1%)")
    else:
        print(f"WARNING: reconstruction MSE {test_recon.item():.6f} >= 0.01, consider more training")


if __name__ == "__main__":
    main()
