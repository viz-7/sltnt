import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def encode(self, x):  # accepts uint8 or float frames
        return self.encoder(x / 255.0)

    def get_value(self, x):
        return self.critic(self.encode(x))

    def get_action_and_value(self, x, action=None, noise_std = 0.0):
        hidden = self.encode(x)
        if noise_std > 0:  # reparameterization: gradients flow through the encoder
            hidden = hidden + noise_std * torch.randn_like(hidden)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class StateTransition(nn.Module):
    """Predicts next encoded state + game-over logit.

    states: (T, N, D), actions: (T, N), episode_starts: (T, N) bool.
    episode_starts[t, n] resets only env n's memory before step t.
    Memory is a tuple (h, c), each (N, hidden_dim).
    """

    def __init__(self, n_actions, state_dim=512, hidden_dim=512):
        super().__init__()
        self.n_actions, self.hidden_dim = n_actions, hidden_dim
        indim = state_dim + n_actions
        self.norm = nn.LayerNorm(indim)
        self.in_ffn = nn.Sequential(layer_init(nn.Linear(indim, hidden_dim)), nn.ReLU())
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.out_ffn = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, state_dim + 2), std=1.0),
        )

    def init_hidden(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(
            batch_size, self.hidden_dim, device=device
        )

    def forward(self, states, actions, hidden=None, episode_starts=None):
        """Returns (next_state_pred (T,N,D), done_logits (T,N), (h, c))."""
        T, N, _ = states.shape
        a = F.one_hot(actions.long(), self.n_actions).to(states.dtype)
        x = self.in_ffn(self.norm(torch.cat([states, a], dim=-1)))

        h, c = hidden if hidden is not None else self.init_hidden(N, states.device)
        keep = (
            torch.ones(T, N, 1, device=x.device, dtype=x.dtype)
            if episode_starts is None
            else (~episode_starts.bool()).to(x.dtype).unsqueeze(-1)
        )

        outs = []
        for t in range(T):
            h, c = self.lstm(x[t], (h * keep[t], c * keep[t]))
            outs.append(h)

        y = self.out_ffn(torch.stack(outs))
        return y[..., :-2], y[..., -2], y[..., -1], (h, c)

def state_transition_loss(pred_state, done_logits, pred_reward, target_state, done_target,
                          reward_target, done_coef=1.0, reward_coef=1.0):
    """MSE on nonterminal next states + BCE on game-over + MSE on (clipped) reward."""
    keep = 1.0 - done_target
    mse = ((pred_state - target_state) ** 2).mean(-1)
    recon = (mse * keep).sum() / keep.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(done_logits, done_target)
    rew = F.binary_cross_entropy_with_logits(pred_reward, reward_target)
    return recon + done_coef * bce + reward_coef * rew, recon, bce, rew
