import torch
import torch.nn as nn
import numpy as np


class ActionVAE(nn.Module):
    """VAE that encodes a 5-dim action into a 36-dim embedding aligned with lidar space."""

    def __init__(self, action_dim=5, embed_dim=36, latent_dim=18):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, action_dim)
        )

        self.embed_proj = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        embed = self.embed_proj(z)
        return recon, embed, mu, logvar

    @torch.no_grad()
    def get_embedding(self, x):
        """Get 36-dim embedding using mean (no sampling noise at inference)."""
        mu, logvar = self.encode(x)
        return self.embed_proj(mu)

    def loss(self, x, beta=0.1):
        recon, embed, mu, logvar = self.forward(x)
        recon_loss = nn.functional.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
