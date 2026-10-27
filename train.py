import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from src.sltnt.model import *

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    resume: Optional[str] = None
    """path to a checkpoint (saved on Ctrl+C under runs/<run_name>/ckpt.pt) to resume from"""
    hist_every: int = 10
    """log weight / gradient histograms every this many iterations (0 disables)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 200
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # State transition model arguments
    st_learning_rate: float = 1e-3
    """the learning rate of the state transition model optimizer"""
    st_epochs: int = 8
    """the number of state transition update epochs per iteration (run after the PPO update)"""
    st_num_minibatches: int = 8
    """the number of mini-batches (split across envs so the LSTM sees full time sequences)"""
    st_done_coef: float = 1.0
    """weight of the game-over BCE loss"""
    st_hidden_dim: int = 512
    """hidden size of the state transition model"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics:
    """Plain wrapper around an envpool Gymnasium pool.

    It deliberately does not subclass gym.Wrapper: newer Gymnasium versions require the
    wrapped object to be a gymnasium.Env, which envpool pools are not. Everything not
    defined here is forwarded to the underlying pool.
    """

    def __init__(self, env):
        self.env = env
        self.num_envs = env.spec.config.num_envs
        self.episode_returns = None
        self.episode_lengths = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        # `infos["reward"]` is the unclipped game reward, `infos["terminated"]` marks true game over
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        game_over = infos["terminated"].astype(bool)
        self.episode_returns[game_over] = 0
        self.episode_lengths[game_over] = 0
        infos["r"] = self.returned_episode_returns.copy()
        infos["l"] = self.returned_episode_lengths.copy()
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.env.close()


def mean_item(tensors):
    return torch.stack(tensors).mean().item()


@torch.no_grad()
def encode_in_chunks(agent, x, chunk=1024):
    return torch.cat([agent.encode(x[i : i + chunk]) for i in range(0, len(x), chunk)], dim=0)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    run_dir = f"runs/{run_name}"
    writer = SummaryWriter(run_dir, flush_secs=10)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    # envpool exposes single-env spaces as `action_space` / `observation_space`
    single_action_space = getattr(envs, "single_action_space", envs.action_space)
    single_observation_space = getattr(envs, "single_observation_space", envs.observation_space)
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    state_dim = agent.actor.in_features
    state_transition = StateTransition(single_action_space.n, state_dim=state_dim, hidden_dim=args.st_hidden_dim).to(device)
    # Only the state transition parameters live in this optimizer, so it can never update the Agent.
    st_optimizer = optim.Adam(state_transition.parameters(), lr=args.st_learning_rate, eps=1e-5)

    # Resume from checkpoint (weights, optimizer state, progress counters)
    start_iteration = 1
    global_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "state_transition" in ckpt:
            state_transition.load_state_dict(ckpt["state_transition"])
            st_optimizer.load_state_dict(ckpt["st_optimizer"])
        start_iteration = ckpt["iteration"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from {args.resume}: iteration={ckpt['iteration']}, global_step={global_step}")
        writer.add_text("resume", f"resumed from `{args.resume}` at iteration {ckpt['iteration']}, global_step {global_step}", global_step)
    resume_global_step = global_step

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)
    total_episodes = 0

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    completed_iteration = start_iteration - 1

    def save_checkpoint(reason):
        ckpt_path = f"{run_dir}/ckpt.pt"
        torch.save(
            {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state_transition": state_transition.state_dict(),
                "st_optimizer": st_optimizer.state_dict(),
                "iteration": completed_iteration,
                "global_step": global_step,
                "args": vars(args),
            },
            ckpt_path,
        )
        writer.add_text("checkpoint", f"saved `{ckpt_path}` ({reason}) at iteration {completed_iteration}", global_step)
        print(f"Saved checkpoint to {ckpt_path} ({reason})\nResume with: python {os.path.basename(__file__)} --resume {ckpt_path}")

    try:
        for iteration in range(start_iteration, args.num_iterations + 1):
            iter_start = time.time()
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            iter_episodes = 0
            iter_lives_lost = 0
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
                next_done_np = np.logical_or(terminations, truncations)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

                for idx, d in enumerate(next_done_np):
                    if d:
                        if info["lives"][idx] == 0:
                            iter_episodes += 1
                            total_episodes += 1
                            print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                            avg_returns.append(info["r"][idx])
                            writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            writer.add_scalar("charts/total_episodes", total_episodes, global_step)
                        else:
                            iter_lives_lost += 1
            rollout_time = time.time() - iter_start

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Rollout statistics
            writer.add_scalar("rollout/reward_mean", rewards.mean().item(), global_step)
            writer.add_scalar("rollout/reward_sum", rewards.sum().item(), global_step)
            writer.add_scalar("rollout/reward_max", rewards.max().item(), global_step)
            writer.add_scalar("rollout/reward_min", rewards.min().item(), global_step)
            writer.add_scalar("rollout/done_fraction", dones.mean().item(), global_step)
            writer.add_scalar("rollout/episodes_finished", iter_episodes, global_step)
            writer.add_scalar("rollout/lives_lost", iter_lives_lost, global_step)
            writer.add_scalar("rollout/value_mean", b_values.mean().item(), global_step)
            writer.add_scalar("rollout/value_std", b_values.std().item(), global_step)
            writer.add_scalar("rollout/return_mean", b_returns.mean().item(), global_step)
            writer.add_scalar("rollout/return_std", b_returns.std().item(), global_step)
            writer.add_scalar("rollout/advantage_mean", b_advantages.mean().item(), global_step)
            writer.add_scalar("rollout/advantage_std", b_advantages.std().item(), global_step)
            writer.add_scalar("rollout/advantage_max", b_advantages.max().item(), global_step)
            writer.add_scalar("rollout/advantage_min", b_advantages.min().item(), global_step)
            writer.add_scalar("rollout/logprob_mean", b_logprobs.mean().item(), global_step)
            writer.add_histogram("rollout/actions", b_actions, global_step)

            # Optimizing the policy and value network
            update_start = time.time()
            b_inds = np.arange(args.batch_size)
            pg_losses, v_losses, entropies = [], [], []
            old_approx_kls, approx_kls, clipfracs = [], [], []
            ratio_maxs, ratio_mins, grad_norms = [], [], []
            epochs_run = 0
            for epoch in range(args.update_epochs):
                epochs_run += 1
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        old_approx_kls.append(old_approx_kl)
                        approx_kls.append(approx_kl)
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean())
                        ratio_maxs.append(ratio.max())
                        ratio_mins.append(ratio.min())

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.detach())
                    v_losses.append(v_loss.detach())
                    entropies.append(entropy_loss.detach())
                    grad_norms.append(grad_norm.detach())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            update_time = time.time() - update_start

            # State transition update (after the PPO update, on the same rollout buffer).
            # The Agent is only used to *produce* encoder states under no_grad + detach, so the
            # state transition loss can never send gradients into Agent.encoder.
            st_start = time.time()
            with torch.no_grad():
                enc = encode_in_chunks(agent, obs.reshape((-1,) + single_observation_space.shape))
                enc = enc.view(args.num_steps, args.num_envs, -1)
                all_states = torch.cat([enc, agent.encode(next_obs).unsqueeze(0)], dim=0).detach()  # (T+1, N, D)
                st_states = all_states[:-1]  # s_t
                st_targets = all_states[1:]  # s_{t+1}
                st_actions = actions.long()  # a_t
                # transition t -> t+1 ends the game iff the next step is flagged done
                st_done = torch.cat([dones[1:], next_done.unsqueeze(0)], dim=0)  # (T, N)
            assert not st_states.requires_grad and not st_targets.requires_grad, "encoder states must be detached from the Agent"

            st_losses, st_recons, st_bces, st_accs = [], [], [], []
            st_env_mb = args.num_envs // args.st_num_minibatches
            env_inds = np.arange(args.num_envs)
            for st_epoch in range(args.st_epochs):
                np.random.shuffle(env_inds)
                for start in range(0, args.num_envs, st_env_mb):
                    mb = torch.as_tensor(env_inds[start : start + st_env_mb], device=device)
                    pred_state, done_logits = state_transition(st_states[:, mb], st_actions[:, mb])
                    _, recon, bce = state_transition_loss(pred_state, done_logits, st_targets[:, mb], st_done[:, mb])
                    st_loss = recon + args.st_done_coef * bce

                    st_optimizer.zero_grad()
                    st_loss.backward()
                    nn.utils.clip_grad_norm_(state_transition.parameters(), args.max_grad_norm)
                    st_optimizer.step()

                    with torch.no_grad():
                        acc = ((done_logits > 0).float() == st_done[:, mb]).float().mean()
                    st_losses.append(st_loss.detach())
                    st_recons.append(recon.detach())
                    st_bces.append(bce.detach())
                    st_accs.append(acc)
            st_time = time.time() - st_start

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            total_time = time.time() - iter_start
            sps = int((global_step - resume_global_step) / (time.time() - start_time))
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/iteration", iteration, global_step)
            writer.add_scalar("charts/epochs_run", epochs_run, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("time/rollout_seconds", rollout_time, global_step)
            writer.add_scalar("time/update_seconds", update_time, global_step)
            writer.add_scalar("time/iteration_seconds", total_time, global_step)
            writer.add_scalar("losses/value_loss", mean_item(v_losses), global_step)
            writer.add_scalar("losses/policy_loss", mean_item(pg_losses), global_step)
            writer.add_scalar("losses/entropy", mean_item(entropies), global_step)
            writer.add_scalar("losses/old_approx_kl", mean_item(old_approx_kls), global_step)
            writer.add_scalar("losses/approx_kl", mean_item(approx_kls), global_step)
            writer.add_scalar("losses/clipfrac", mean_item(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("losses/grad_norm", mean_item(grad_norms), global_step)
            writer.add_scalar("time/state_transition_seconds", st_time, global_step)
            writer.add_scalar("state_transition/loss", mean_item(st_losses), global_step)
            writer.add_scalar("state_transition/recon_loss", mean_item(st_recons), global_step)
            writer.add_scalar("state_transition/done_bce", mean_item(st_bces), global_step)
            writer.add_scalar("state_transition/done_accuracy", mean_item(st_accs), global_step)
            writer.add_scalar("state_transition/terminal_fraction", st_done.mean().item(), global_step)
            writer.add_scalar("losses/ratio_max", torch.stack(ratio_maxs).max().item(), global_step)
            writer.add_scalar("losses/ratio_min", torch.stack(ratio_mins).min().item(), global_step)
            if args.hist_every > 0 and iteration % args.hist_every == 0:
                for name, p in agent.named_parameters():
                    writer.add_histogram(f"weights/{name}", p.detach().cpu(), global_step)
                    if p.grad is not None:
                        writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), global_step)
            print("SPS:", sps)

            completed_iteration = iteration
    except KeyboardInterrupt:
        print("\nCtrl+C received, saving checkpoint...")
        save_checkpoint("interrupted")
    else:
        save_checkpoint("training completed")
    finally:
        envs.close()
        writer.close()
