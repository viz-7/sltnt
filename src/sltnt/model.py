import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def encode(self, x):
        return self.encoder(x / 255.0)

    def get_value(self, x):
        return self.critic(self.encoder(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class StateTransition(nn.Module):
    """FFN -> LSTM -> FFN. (state_t, action_t) -> (state_{t+1}, game-over logit)."""

    def __init__(self, n_actions, state_dim=512, hidden_dim=512):
        super().__init__()
        self.n_actions = n_actions
        indim = state_dim + n_actions
        self.norm = nn.LayerNorm(indim)
        self.in_ffn = nn.Sequential(
            layer_init(nn.Linear(indim, hidden_dim)), nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.out_ffn = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, state_dim + 1), std=1.0),  # last output = done logit
        )

    def forward(self, states, actions):
        # states: (T, N, state_dim), actions: (T, N)
        a = nn.functional.one_hot(actions.long(), self.n_actions).float()
        x = self.in_ffn(self.norm(torch.cat([states, a], dim=-1)))
        x, _ = self.lstm(x)
        out = self.out_ffn(x)
        return out[..., :-1], out[..., -1]  # next_state_pred (T, N, D), done_logits (T, N)

def state_transition_loss(pred_state, done_logits, target_state, done_target):
    nonterminal = 1.0 - done_target
    per_step_mse = ((pred_state - target_state) ** 2).mean(dim=-1)
    recon = (per_step_mse * nonterminal).sum() / nonterminal.sum().clamp(min=1.0)
    bce = nn.functional.binary_cross_entropy_with_logits(done_logits, done_target)
    return recon + bce, recon, bce
