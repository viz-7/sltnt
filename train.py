import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Support both `python src/sltnt/train.py` and running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sltnt.model import Agent, StateTransition, state_transition_loss


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    resume: Optional[str] = None
    hist_every: int = 10

    # PPO
    env_id: str = "Breakout-v5"
    total_timesteps: int = 3750000
    learning_rate: float = 2.5e-4
    num_envs: int = 512
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    noise_std: float = 0.1
    target_kl: Optional[float] = None

    # Feed-forward state transition model
    st_learning_rate: float = 1e-3
    st_epochs: int = 8
    st_num_minibatches: int = 32
    st_done_coef: float = 1.0
    st_reward_coef: float = 1.0
    st_hidden_dim: int = 512
    st_encode_batch_size: int = 128

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@torch.no_grad()
def encode_observations_half(agent, observations, device, chunk_size=128):
    """Encode CPU/GPU observations in small chunks and retain FP16 latents.

    The observation rollout stays on CPU. Only one small image chunk is copied
    to the GPU at a time, avoiding a second full float image buffer.
    """
    leading_shape = observations.shape[:-3]
    flat = observations.reshape(-1, *observations.shape[-3:])
    state_dim = agent.actor.in_features

    encoded = torch.empty(
        (flat.shape[0], state_dim),
        dtype=torch.float16,
        device=device,
    )

    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        image_batch = flat[start:end].to(device, non_blocking=True)
        encoded[start:end].copy_(agent.encode(image_batch).to(torch.float16))
        del image_batch

    return encoded.reshape(*leading_shape, state_dim)


@torch.no_grad()
def make_state_windows(enc, history_obs, game_overs, env_indices, agent, device, encode_batch_size):
    """Build a small group's [s(t-2), s(t-1), s(t)] windows.

    `history_obs` is raw uint8 CPU storage with shape (2, N, C, H, W).
    The first two observations are re-encoded after the PPO update, so every
    latent in this rollout uses the same, updated Agent encoder.
    """
    batch_size = len(env_indices)
    T = enc.shape[0]
    state_dim = agent.actor.in_features

    env_indices_cpu = env_indices.cpu()
    previous_two = encode_observations_half(
        agent,
        history_obs[0, env_indices_cpu],
        device,
        encode_batch_size,
    ).float()
    previous_one = encode_observations_half(
        agent,
        history_obs[1, env_indices_cpu],
        device,
        encode_batch_size,
    ).float()

    windows = torch.empty(
        (T, batch_size, 3, state_dim),
        dtype=torch.float32,
        device=device,
    )

    for t in range(T):
        current = enc[t, env_indices].float()

        if t > 0:
            reset = game_overs[t - 1, env_indices].bool().unsqueeze(-1)
            next_previous_two = torch.where(reset, current, previous_one)
            next_previous_one = torch.where(
                reset,
                current,
                enc[t - 1, env_indices].float(),
            )
            previous_two = next_previous_two
            previous_one = next_previous_one

        windows[t, :, 0].copy_(previous_two)
        windows[t, :, 1].copy_(previous_one)
        windows[t, :, 2].copy_(current)

    return windows


def update_raw_history(history_obs, obs, next_obs_cpu, game_overs):
    """Keep the two observations preceding the next PPO buffer's first state.

    This history is raw uint8 CPU data, only two frames per environment.
    On a true game-over, history is reset to the autoreset observation.
    Life-loss PPO boundaries do not reset it.
    """
    T = obs.shape[0]

    if T >= 2:
        history_obs[0].copy_(obs[-2])
        history_obs[1].copy_(obs[-1])

        # If transition T-2 ended a game, obs[T-1] is the new game's first
        # observation, so duplicate it into both history slots.
        ended_before_last_state = game_overs[-2].bool().cpu()
        if ended_before_last_state.any():
            history_obs[0, ended_before_last_state] = obs[-1, ended_before_last_state]
            history_obs[1, ended_before_last_state] = obs[-1, ended_before_last_state]
    else:
        history_obs[0].copy_(history_obs[1])
        history_obs[1].copy_(obs[0])

    # The last transition ended a game. next_obs_cpu is the autoreset state.
    ended_at_buffer_end = game_overs[-1].bool().cpu()
    if ended_at_buffer_end.any():
        history_obs[0, ended_at_buffer_end] = next_obs_cpu[ended_at_buffer_end]
        history_obs[1, ended_at_buffer_end] = next_obs_cpu[ended_at_buffer_end]


if __name__ == "__main__":
    args = tyro.cli(Args)

    N, T = args.num_envs, args.num_steps
    if N < 1 or T < 1:
        raise ValueError("num_envs and num_steps must both be >= 1")
    if args.num_minibatches < 1 or args.st_num_minibatches < 1:
        raise ValueError("PPO and ST minibatch counts must both be >= 1")
    if args.st_encode_batch_size < 1:
        raise ValueError("st_encode_batch_size must be >= 1")

    args.batch_size = N * T
    args.num_minibatches = min(args.num_minibatches, args.batch_size)
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("total_timesteps must be at least num_envs * num_steps")

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
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir, flush_secs=10)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    torch.set_float32_matmul_precision("high")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # ---------------- Environment ----------------
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=N,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    act_space = getattr(envs, "single_action_space", envs.action_space)
    obs_space = getattr(envs, "single_observation_space", envs.observation_space)
    assert isinstance(act_space, gym.spaces.Discrete), (
        "Only discrete action spaces are supported"
    )

    # ---------------- Models ----------------
    agent = Agent(act_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    st = StateTransition(
        act_space.n,
        agent.actor.in_features,
        args.st_hidden_dim,
    ).to(device)
    st_optimizer = optim.Adam(st.parameters(), lr=args.st_learning_rate, eps=1e-5)

    # ---------------- Resume ----------------
    start_iter, global_step = 1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])

        saved_st = ckpt.get("state_transition")
        current_st = st.state_dict()
        compatible_st = (
            saved_st is not None
            and saved_st.keys() == current_st.keys()
            and all(
                saved_st[key].shape == current_st[key].shape
                for key in current_st
            )
        )
        if compatible_st:
            st.load_state_dict(saved_st)
            if "st_optimizer" in ckpt:
                try:
                    st_optimizer.load_state_dict(ckpt["st_optimizer"])
                except (ValueError, KeyError):
                    print("ST optimizer state was incompatible; reinitializing it.")
            print("Loaded compatible StateTransition weights.")
        else:
            print(
                "Checkpoint StateTransition architecture is incompatible; "
                "starting the two-layer FFN from scratch. PPO weights were loaded."
            )

        start_iter = ckpt["iteration"] + 1
        global_step = ckpt["global_step"]
        print(
            f"Resumed from {args.resume}: iteration={ckpt['iteration']}, "
            f"global_step={global_step}"
        )

    # ---------------- Rollout storage ----------------
    # The large image buffer lives on CPU, not GPU. Atari frames are uint8.
    # PPO transfers only the selected minibatch to the GPU.
    obs = torch.empty(
        (T, N) + obs_space.shape,
        dtype=torch.uint8,
        device="cpu",
    )
    actions = torch.zeros(T, N, dtype=torch.long, device=device)
    logprobs = torch.zeros(T, N, device=device)
    rewards = torch.zeros(T, N, device=device)
    dones = torch.zeros(T, N, device=device)
    values = torch.zeros(T, N, device=device)
    game_overs = torch.zeros(T, N, device=device)

    reset_obs_np = envs.reset()[0]
    next_obs_cpu = torch.as_tensor(reset_obs_np).clone()
    next_obs = next_obs_cpu.to(device, non_blocking=True)
    next_done = torch.zeros(N, device=device)

    # [two-steps-ago, previous-step] raw observations for every environment.
    # At initialization the requested context is [current, current, current].
    history_obs = torch.stack((next_obs_cpu, next_obs_cpu), dim=0)

    resume_step = global_step
    start_time = time.time()
    ep_ret = np.zeros(N, np.float32)
    ep_len = np.zeros(N, np.int32)
    avg_returns = deque(maxlen=20)
    total_episodes = 0
    completed_iter = start_iter - 1

    def save_checkpoint(reason):
        path = f"{run_dir}/ckpt.pt"
        torch.save(
            {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "state_transition": st.state_dict(),
                "st_optimizer": st_optimizer.state_dict(),
                "iteration": completed_iter,
                "global_step": global_step,
                "args": vars(args),
            },
            path,
        )
        writer.add_text("checkpoint", f"saved `{path}` ({reason})", global_step)
        print(f"Saved checkpoint: {path} ({reason})")

    try:
        for iteration in range(start_iter, args.num_iterations + 1):
            t0 = time.time()
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (
                    1.0 - (iteration - 1.0) / args.num_iterations
                ) * args.learning_rate

            iter_episodes = 0
            iter_lives_lost = 0

            # ---------- 1. PPO rollout ----------
            for step in range(T):
                global_step += N
                obs[step].copy_(next_obs_cpu)
                dones[step].copy_(next_done)

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term, trunc)
                # EnvPool's info flag represents true game-over, not episodic-life.
                over = np.asarray(info["terminated"], dtype=bool)

                ep_ret += info["reward"]
                ep_len += 1
                for i in np.flatnonzero(over):
                    total_episodes += 1
                    avg_returns.append(ep_ret[i])
                    for key, value_to_log in (
                        ("avg_episodic_return", np.mean(avg_returns)),
                        ("episodic_return", ep_ret[i]),
                        ("episodic_length", ep_len[i]),
                        ("total_episodes", total_episodes),
                    ):
                        writer.add_scalar(f"charts/{key}", value_to_log, global_step)

                ep_ret[over] = 0
                ep_len[over] = 0
                iter_episodes += int(over.sum())
                iter_lives_lost += int((done_np & ~over).sum())

                flags = torch.as_tensor(
                    np.stack([reward, over, done_np]).astype(np.float32),
                    device=device,
                )
                rewards[step] = flags[0]
                game_overs[step] = flags[1]
                next_done = flags[2]

                # Keep the raw copy on CPU for rollout storage/history and only
                # one current observation batch on GPU for acting/GAE.
                next_obs_cpu = torch.as_tensor(obs_np).clone()
                next_obs = next_obs_cpu.to(device, non_blocking=True)

            rollout_time = time.time() - t0

            # ---------- 2. GAE ----------
            with torch.no_grad():
                next_value = agent.get_value(next_obs).flatten()
                nonterminal = 1.0 - torch.cat([dones[1:], next_done[None]], dim=0)
                next_values = torch.cat([values[1:], next_value[None]], dim=0)
                deltas = rewards + args.gamma * next_values * nonterminal - values

                advantages = torch.zeros_like(rewards)
                last = 0
                for t in reversed(range(T)):
                    last = advantages[t] = (
                        deltas[t]
                        + args.gamma * args.gae_lambda * nonterminal[t] * last
                    )
                returns = advantages + values

            # These are views. The actual image storage remains on CPU.
            b_obs = obs.view(T * N, *obs_space.shape)
            b_logprobs = logprobs.flatten()
            b_actions = actions.flatten()
            b_advantages = advantages.flatten()
            b_returns = returns.flatten()
            b_values = values.flatten()

            # ---------- 3. PPO update ----------
            t1 = time.time()
            clip_coef = args.clip_coef
            stats = []

            for epoch in range(args.update_epochs):
                permutation = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb = permutation[start : start + args.minibatch_size]
                    mb_cpu = mb.cpu()
                    obs_mb = b_obs.index_select(0, mb_cpu).to(
                        device,
                        non_blocking=True,
                    )

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        obs_mb,
                        b_actions[mb],
                        noise_std=args.noise_std,
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()

                    adv = b_advantages[mb]
                    if args.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    pg_loss = torch.max(
                        -adv * ratio,
                        -adv * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef),
                    ).mean()

                    newvalue = newvalue.view(-1)
                    value_error = (newvalue - b_returns[mb]) ** 2
                    if args.clip_vloss:
                        clipped_value = b_values[mb] + (
                            newvalue - b_values[mb]
                        ).clamp(-clip_coef, clip_coef)
                        value_error = torch.max(
                            value_error,
                            (clipped_value - b_returns[mb]) ** 2,
                        )
                    value_loss = 0.5 * value_error.mean()
                    entropy_mean = entropy.mean()

                    loss = (
                        pg_loss
                        - args.ent_coef * entropy_mean
                        + args.vf_coef * value_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        args.max_grad_norm,
                    )
                    optimizer.step()

                    with torch.no_grad():
                        stats.append(
                            torch.stack(
                                [
                                    pg_loss,
                                    value_loss,
                                    entropy_mean,
                                    (-logratio).mean(),
                                    ((ratio - 1.0) - logratio).mean(),
                                    ((ratio - 1.0).abs() > clip_coef).float().mean(),
                                    ratio.max(),
                                    ratio.min(),
                                    grad_norm,
                                ]
                            )
                        )

                    del obs_mb, loss

                if args.target_kl is not None and stats[-1][4] > args.target_kl:
                    break

            stats_tensor = torch.stack(stats)
            pg, vl, ent, old_kl, kl, clipfrac, _, _, grad_norm = (
                stats_tensor.mean(0).tolist()
            )
            ratio_max = stats_tensor[:, 6].max().item()
            ratio_min = stats_tensor[:, 7].min().item()
            update_time = time.time() - t1

            # Release the Agent's last PPO gradients before ST training.
            optimizer.zero_grad(set_to_none=True)
            del stats, stats_tensor

            # ---------- 4. Encode rollout using the UPDATED encoder ----------
            # Crucially, do not concatenate obs[1:] with next_obs. The raw
            # rollout is already on CPU, and only compact FP16 latents are kept.
            t2 = time.time()
            agent.eval()
            with torch.no_grad():
                encoded_obs = encode_observations_half(
                    agent,
                    obs,
                    device,
                    args.st_encode_batch_size,
                )
                final_encoded = agent.encode(next_obs).to(torch.float16)

            # ---------- 5. Train StateTransition in small environment groups ----------
            st_stats = []
            env_group_size = max(
                1,
                (N + args.st_num_minibatches - 1) // args.st_num_minibatches,
            )

            st.train()
            for _ in range(args.st_epochs):
                env_permutation = torch.randperm(N, device=device)
                for start in range(0, N, env_group_size):
                    env_indices = env_permutation[start : start + env_group_size]

                    state_windows = make_state_windows(
                        encoded_obs,
                        history_obs,
                        game_overs,
                        env_indices,
                        agent,
                        device,
                        args.st_encode_batch_size,
                    )

                    batch_envs = len(env_indices)
                    target_states = torch.empty(
                        (T, batch_envs, agent.actor.in_features),
                        dtype=torch.float32,
                        device=device,
                    )
                    if T > 1:
                        target_states[:-1].copy_(
                            encoded_obs[1:, env_indices].float()
                        )
                    target_states[-1].copy_(final_encoded[env_indices].float())

                    action_batch = actions[:, env_indices]
                    done_target = game_overs[:, env_indices]
                    reward_target = rewards[:, env_indices]

                    pred_state, done_logits, pred_reward = st(
                        state_windows,
                        action_batch,
                    )
                    st_loss, recon_loss, done_bce, reward_bce = state_transition_loss(
                        pred_state,
                        done_logits,
                        pred_reward,
                        target_states,
                        done_target,
                        reward_target,
                        args.st_done_coef,
                        args.st_reward_coef,
                    )

                    st_optimizer.zero_grad(set_to_none=True)
                    st_loss.backward()
                    nn.utils.clip_grad_norm_(st.parameters(), args.max_grad_norm)
                    st_optimizer.step()

                    with torch.no_grad():
                        done_acc = (
                            (done_logits > 0) == done_target.bool()
                        ).float().mean()
                        st_stats.append(
                            torch.stack(
                                [
                                    st_loss.detach(),
                                    recon_loss.detach(),
                                    done_bce.detach(),
                                    done_acc,
                                    reward_bce.detach(),
                                ]
                            )
                        )

                    del state_windows, target_states
                    del pred_state, done_logits, pred_reward, st_loss

            st_optimizer.zero_grad(set_to_none=True)
            st_l, st_recon, st_bce, st_acc, st_rew = (
                torch.stack(st_stats).mean(0).tolist()
            )
            st_time = time.time() - t2
            del encoded_obs, final_encoded, st_stats

            # ---------- 6. Carry raw history into the next PPO buffer ----------
            update_raw_history(history_obs, obs, next_obs_cpu, game_overs)
            agent.train()

            # ---------- 7. Logging ----------
            variance = torch.var(b_returns, correction=0)
            explained_variance = (
                float("nan")
                if variance == 0
                else (
                    1.0
                    - torch.var(b_returns - b_values, correction=0) / variance
                ).item()
            )
            sps = int((global_step - resume_step) / (time.time() - start_time))

            logs = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/iteration": iteration,
                "charts/epochs_run": epoch + 1,
                "charts/SPS": sps,
                "time/rollout_seconds": rollout_time,
                "time/update_seconds": update_time,
                "time/state_transition_seconds": st_time,
                "time/iteration_seconds": time.time() - t0,
                "rollout/reward_mean": rewards.mean().item(),
                "rollout/reward_sum": rewards.sum().item(),
                "rollout/reward_max": rewards.max().item(),
                "rollout/reward_min": rewards.min().item(),
                "rollout/done_fraction": dones.mean().item(),
                "rollout/game_over_fraction": game_overs.mean().item(),
                "rollout/episodes_finished": iter_episodes,
                "rollout/lives_lost": iter_lives_lost,
                "losses/value_loss": vl,
                "losses/policy_loss": pg,
                "losses/entropy": ent,
                "losses/old_approx_kl": old_kl,
                "losses/approx_kl": kl,
                "losses/clipfrac": clipfrac,
                "losses/explained_variance": explained_variance,
                "losses/grad_norm": grad_norm,
                "losses/ratio_max": ratio_max,
                "losses/ratio_min": ratio_min,
                "state_transition/loss": st_l,
                "state_transition/recon_loss": st_recon,
                "state_transition/done_bce": st_bce,
                "state_transition/reward_bce": st_rew,
                "state_transition/done_accuracy": st_acc,
                "state_transition/game_over_fraction": game_overs.mean().item(),
            }
            for key, value_to_log in logs.items():
                writer.add_scalar(key, value_to_log, global_step)

            if args.hist_every > 0 and iteration % args.hist_every == 0:
                for name, parameter in agent.named_parameters():
                    writer.add_histogram(
                        f"weights/{name}",
                        parameter.detach().cpu(),
                        global_step,
                    )

            print(
                f"Iteration {iteration} | SPS: {sps} | "
                f"Avg return: {np.mean(avg_returns) if avg_returns else float('nan'):.1f} | "
                f"ST loss: {st_l:.5f} | ST recon: {st_recon:.5f} | "
                f"ST done acc: {st_acc:.3f}"
            )
            completed_iter = iteration

    except KeyboardInterrupt:
        print("\nCtrl+C received, saving checkpoint...")
        save_checkpoint("interrupted")
    else:
        save_checkpoint("training completed")
    finally:
        envs.close()
        writer.close()
