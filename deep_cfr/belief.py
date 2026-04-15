"""
Encodes the current observed state of the world from the blue force.
This includes the state of the battlefield and believed enemy intent.

A unique latent representation is endowed to each observation (info set) at time t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObsEmbedding(nn.Module):
    # Observed features of the battlefield as continuous state vectors
    def __init__(self, obs_dim: int, output_dim: int):
        super().__init__()
        self.obs_layer = nn.Linear(obs_dim, output_dim) # Project observations into new dimension
    def forward(self, obs):
        return F.relu(self.obs_layer(obs))

class ActionEmbedding(nn.Module):
    # Discrete actions performed within an observed state
    def __init__(self, action_dim: int, output_dim: int):
        super().__init__()
        self.action_layer = nn.Linear(action_dim, output_dim)
    def foward(self, actions):
        return F.relu(self.action_layer(actions))

class BeliefEncoder(nn.Module):
    # Input: observation and action history
    # Output: latent representation of belief vector summarizing known info set
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int = 32, # Matches obs dim
        hidden_dim: int = 64, # Maintain sufficient memory of historic actions
        belief_dim: int = 16, # Distill belief history to a 16 dim vector
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_part = ObsEmbedding(obs_dim, embed_dim)
        self.action_part = ActionEmbedding(action_dim, embed_dim)
        self.step = nn.Linear(obs_dim + action_dim, embed_dim) # 64 dim in combination

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_belief = nn.Linear(hidden_dim, belief_dim)

    def forward(self, obs_history, action_history):
        obs_x = self.obs_part(obs_history)
        act_x = self.action_part(action_history)

        step_x = torch.cat([obs_x, act_x], dim=-1)
        step_x = F.relu(self.step(step_x))

        _, h_n = self.gru(step_x)

        h_last = h_n[-1]
        belief = F.relu(self.to_belief(h_last))
        return belief