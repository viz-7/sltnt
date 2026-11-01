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
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def encode(self, x):
        """Encode uint8 or float Atari observations into latent states."""
        return self.encoder(x.float() / 255.0)

    def get_value(self, x):
        return self.critic(self.encode(x))

    def get_action_and_value(self, x, action=None, noise_std=0.0):
        hidden = self.encode(x)
        if noise_std > 0:
            hidden = hidden + noise_std * torch.randn_like(hidden)
        probs = Categorical(logits=self.actor(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class StateTransition(nn.Module):
    """Two-layer FFN predicting the next state, game-over, and reward.

    `states` has shape (..., 3, state_dim), ordered as:
        [state(t-2), state(t-1), state(t)]

    `actions` has shape matching the leading dimensions of `states`.
    """

    def __init__(self, n_actions, state_dim=512, hidden_dim=512):
        super().__init__()
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        input_dim = 3 * state_dim + n_actions
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, state_dim + 2), std=1.0)

    def forward(self, states, actions):
        if states.shape[-2] != 3:
            raise ValueError(
                f"Expected 3 states in the input window, got {states.shape[-2]}"
            )
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got {states.shape[-1]}"
            )

        action_one_hot = F.one_hot(
            actions.long(), num_classes=self.n_actions
        ).to(dtype=states.dtype)

        x = torch.cat(
            [states.flatten(start_dim=-2), action_one_hot],
            dim=-1,
        )
        x = F.relu(self.fc1(self.norm(x)))
        output = self.fc2(x)

        return output[..., :-2], output[..., -2], output[..., -1]


def state_transition_loss(
    pred_state,
    done_logits,
    pred_reward,
    target_state,
    done_target,
    reward_target,
    done_coef=1.0,
    reward_coef=1.0,
):
    """Nonterminal next-state MSE plus game-over and binary-reward BCE."""
    done_target = done_target.float()
    reward_target = reward_target.float()

    keep = 1.0 - done_target
    mse = ((pred_state - target_state) ** 2).mean(dim=-1)
    recon = (mse * keep).sum() / keep.sum().clamp(min=1.0)

    done_bce = F.binary_cross_entropy_with_logits(done_logits, done_target)
    reward_bce = F.binary_cross_entropy_with_logits(
        pred_reward, reward_target
    )

    total = recon + done_coef * done_bce + reward_coef * reward_bce
    return total, recon, done_bce, reward_bce
