import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


@dataclass
class VAEOutput:
    z: torch.Tensor          # sampled latent space        [batch, 8]
    
    mu: torch.Tensor         # encoder mean               [batch, 8]
    std: torch.Tensor        # encoder std deviation       [batch, 8]
    
    x_recon: torch.Tensor    # reconstructed trajectory    [batch, 400]
    loss: torch.Tensor       # total loss (scalar)
    recon_loss: torch.Tensor # reconstruction component    (scalar)
    kl_loss: torch.Tensor    # KL divergence component     (scalar)

class TacticVAE(nn.Module):
    # input dim = [Batch size, 50 timesteps, 8 actions] (for time based GRU)
    # hidden dim = 64 (~53K params)
    # latent dim = 8 (400 input dims are compressed to an 8 dim vector representing tactic axes)
    def __init__(self, input_dim: int = 400, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 2 hidden layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # 400 → 128
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim), # 128 → 128
            nn.LeakyReLU(0.01),
        )
        
        # mu and std as heads to represent probability distributions of the 
        # latent space from the hidden layers above (i.e. a distilled representation
        # of the tactic trajectory)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)
        
        # mirrors the encoder (backwards)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),      # 8 → 128
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),      # 128 → 128
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, input_dim),       # 128 → 400
            nn.Sigmoid(),                           # bound to [0, 1] (probability distributions)
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Compute random noise
        
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        
        # Compute mean and log variance (to guarantee positive values)
        mu = self.fc_mu(encoded)
        log_var = self.fc_std(encoded)

        # Reparameterize the latent variable so we can run GD
        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(z)
        return decoded, mu, log_var
    
    # Loss function uses Evidence Lower Bound (ELBO) 
    # - CE - tries to get the input to the encoder and output of the decoder to match
    # - KL divergence - tries to get the distribution of the latent representation to match the gaussian distribution
    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vae(X_train, learning_rate=1e-3, num_epochs=10, batch_size=32):
    X_train = X_train.view(X_train.shape[0], -1).to(device)

    # Set up params
    model = TacticVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True
    )

    # Train in batches
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            # Move batch of data to device
            x_batch = data[0].to(device)

            # Zero out gradients from previous batch
            optimizer.zero_grad()
            x_recon, mu, log_var = model(x_batch)
            loss = model.loss_function(x_recon, x_batch, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}")
    
    return model