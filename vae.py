import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path


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

        # Encoder returns the mu and std as heads to represent probability distributions of the 
        # latent space from the hidden layers above (i.e. a distilled representation
        # of the tactic trajectory)
        self.encoder = nn.GRU(
            input_size=8,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder requires initial hidden state which is uses to predict next states recursively
        # The output of the hidden layer is projected to the original 8 size feature dim
        self.get_h0 = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(
            input_size=8,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_dim, 8)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Compute random noise
        
        return mu + eps * std

    def forward(self, x):
        # Returns list of full encoder output (enc_out) and output at final timestamp (h_n)
        enc_out, h_n = self.encoder(x)
        
        # The last hidden state summarizes the entire input sequence. We project it into 
        # the mean and (logged) variance to get the latent Gaussian distribution q(z|x) 
        # for the VAE encoder.
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        log_var = self.fc_std(h_last)

        # Reparameterize the latent variable so we can run GD
        z = self.reparameterize(mu, log_var)

        # Initialize decoder with the ground truth output from  previous states as the 
        # input to the decoder so model can make predictions from real values (teacher forcing)
        h0 = self.get_h0(z).unsqueeze(0)
        dec_out, h_n = self.decoder(x, h0)
        logits = self.output_head(dec_out)
        return logits, mu, log_var
    
    # Loss function uses timestep based cross entropy loss to evaluate if the model
    # made correct predictions about the probability of future outcomes
    def loss_function(self, logits, x, mu, log_var, beta=1.0):
        # Return the expected action from the one hot vector
        target_idx = x.argmax(dim=-1)

        # collapse into 2 dimensions so CE can process
        logits_flat = logits.reshape(-1, 8)
        target_flat = target_idx.reshape(-1)

        # Loss is a combination of CE (answer accuracy) and KLD (quality of prediction)
        # Beta is updated dynamically during training to prevent kld from overpowering loss
        recon = F.cross_entropy(logits_flat, target_flat, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon + beta * kld
        return loss

def load_data(X, device, batch_size=32):
    # Stratified sampling (80/20) with reproducible splits
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    dataset = TensorDataset(X_tensor)

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    X_train, X_test = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(
        X_train, batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        X_test, batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader

def evaluate_vae(model, device, test_loader):
    # Track validation/test loss
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (x_batch, ) in test_loader:
            x_batch = x_batch.to(device)
            logits, mu, log_var = model(x_batch)
            loss = model.loss_function(logits, x_batch, mu, log_var)
            total_loss += loss.item()
    
        loss_norm = total_loss / len(test_loader.dataset)
    print(f"total loss: {loss_norm}")
    return loss_norm

def train_vae(model, train_loader, test_loader, device, learning_rate=1e-3, num_epochs=10, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = Path("artifacts/checkpoints/vae_best.pt")
    best_test_loss = float("inf")

    # Train in batches
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for data in train_loader:
            # Move batch of data to device
            x_batch = data[0].to(device)

            # Zero out gradients from previous batch
            optimizer.zero_grad()
            logits, mu, log_var = model(x_batch)
            loss = model.loss_function(logits, x_batch, mu, log_var)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader.dataset)

        # Save best performing model
        if test_loader is not None:
            test_loss = evaluate_vae(model, device, test_loader)
            print(f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss:.6f} | val_loss={test_loss:.6f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": test_loss,
                        "model_config": {
                            "input_dim": model.input_dim,
                            "hidden_dim": model.hidden_dim,
                            "latent_dim": model.latent_dim,
                        },
                    },
                    checkpoint_path,
                )
                print(f"Saved new best checkpoint: {checkpoint_path}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss:.6f}")
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = TacticVAE().to(device)
        X = np.load("data/processed/trajectories_training.npy")
        train_loader, test_loader = load_data(X, device)
        train_vae(model, train_loader, test_loader, device)
    except Exception as e:
        print(f"error during training process {e}")