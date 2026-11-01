import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import envpool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from src.sltnt.model import Agent, StateTransition, state_transition_loss

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
    target_kl: Optional[float] = None

    # State transition
    st_learning_rate: float = 1e-3
    st_epochs: int = 8
    st_num_minibatches: int = 32
    st_done_coef: float = 1.0
    st_reward_coef: float = 1.0
    st_hidden_dim: int = 512

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    N, T = args.num_envs, args.num_steps
    args.batch_size = N * T
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    if N % args.st_num_minibatches:
        raise ValueError("num_envs must be divisible by st_num_minibatches")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args), name=run_name,
            monitor_gym=True, save_code=True,
        )

    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir, flush_secs=10)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------- Environment ----------------
    envs = envpool.make(
        args.env_id, env_type="gymnasium", num_envs=N,
        episodic_life=True, reward_clip=True, seed=args.seed,
    )
    act_space = getattr(envs, "single_action_space", envs.action_space)
    obs_space = getattr(envs, "single_observation_space", envs.observation_space)
    assert isinstance(act_space, gym.spaces.Discrete), "Only discrete action spaces are supported"

    # ---------------- Models ----------------
    agent = Agent(act_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    st = StateTransition(act_space.n, agent.actor.in_features, args.st_hidden_dim).to(device)
    st_optimizer = optim.Adam(st.parameters(), lr=args.st_learning_rate, eps=1e-5)

    # ---------------- Rollout storage (frames kept as uint8) ----------------
    obs = torch.zeros((T, N) + obs_space.shape, dtype=torch.uint8, device=device)
    actions = torch.zeros(T, N, dtype=torch.long, device=device)
    logprobs, rewards, dones, values, game_overs = (
        torch.zeros(T, N, device=device) for _ in range(5)
    )  # game_overs: true game-over only (no life loss / truncation)

    # ---------------- Resume ----------------
    start_iter, global_step = 1, 0
    st_hidden = st.init_hidden(N, device)  # per-env memory carried across rollouts
    next_game_over = torch.zeros(N, dtype=torch.bool, device=device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "state_transition" in ckpt:
            st.load_state_dict(ckpt["state_transition"])
            st_optimizer.load_state_dict(ckpt["st_optimizer"])
        if "st_hidden" in ckpt:
            st_hidden = tuple(x.to(device) for x in ckpt["st_hidden"])
            next_game_over = ckpt["next_game_over"].to(device)
        start_iter, global_step = ckpt["iteration"] + 1, ckpt["global_step"]
        print(f"Resumed from {args.resume}: iteration={ckpt['iteration']}, global_step={global_step}")

    resume_step, start_time = global_step, time.time()
    next_obs = torch.as_tensor(envs.reset()[0], device=device)
    next_done = torch.zeros(N, device=device)
    ep_ret, ep_len = np.zeros(N, np.float32), np.zeros(N, np.int32)
    avg_returns, total_episodes, completed_iter = deque(maxlen=20), 0, start_iter - 1

    def save_checkpoint(reason):
        path = f"{run_dir}/ckpt.pt"
        torch.save(
            {
                "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                "state_transition": st.state_dict(), "st_optimizer": st_optimizer.state_dict(),
                "st_hidden": st_hidden, "next_game_over": next_game_over,
                "iteration": completed_iter, "global_step": global_step, "args": vars(args),
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

            iter_episodes = iter_lives_lost = 0
            rollout_start = next_game_over  # did the previous rollout end a game?

            # ---------- 1. Rollout ----------
            for step in range(T):
                global_step += N
                obs[step], dones[step] = next_obs, next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step], actions[step], logprobs[step] = value.flatten(), action, logprob

                obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term, trunc)  # PPO: termination or truncation
                over = np.asarray(info["terminated"], dtype=bool)  # ST: true game over

                ep_ret += info["reward"]
                ep_len += 1
                for i in np.flatnonzero(over):
                    total_episodes += 1
                    avg_returns.append(ep_ret[i])
                    for k, v in (
                        ("avg_episodic_return", np.mean(avg_returns)),
                        ("episodic_return", ep_ret[i]),
                        ("episodic_length", ep_len[i]),
                        ("total_episodes", total_episodes),
                    ):
                        writer.add_scalar(f"charts/{k}", v, global_step)
                ep_ret[over], ep_len[over] = 0, 0
                iter_episodes += int(over.sum())
                iter_lives_lost += int((done_np & ~over).sum())

                # one host->device copy for all per-step flags
                flags = torch.as_tensor(
                    np.stack([reward, over, done_np]).astype(np.float32), device=device
                )
                rewards[step], game_overs[step], next_done = flags[0], flags[1], flags[2]
                next_game_over = flags[1].bool()
                next_obs = torch.as_tensor(obs_np, device=device)

            rollout_time = time.time() - t0

            # ---------- 2. GAE ----------
            with torch.no_grad():
                next_value = agent.get_value(next_obs).flatten()
                nonterm = 1.0 - torch.cat([dones[1:], next_done[None]])
                nextvals = torch.cat([values[1:], next_value[None]])
                deltas = rewards + args.gamma * nextvals * nonterm - values
                advantages, last = torch.zeros_like(rewards), 0
                for t in reversed(range(T)):
                    last = advantages[t] = (
                        deltas[t] + args.gamma * args.gae_lambda * nonterm[t] * last
                    )
                returns = advantages + values

            b_obs = obs.flatten(0, 1)
            b_logprobs, b_actions = logprobs.flatten(), actions.flatten()
            b_adv, b_returns, b_values = advantages.flatten(), returns.flatten(), values.flatten()

            # ---------- 3. PPO update ----------
            t1 = time.time()
            c, stats = args.clip_coef, []
            for epoch in range(args.update_epochs):
                perm = torch.randperm(args.batch_size, device=device)
                for s in range(0, args.batch_size, args.minibatch_size):
                    mb = perm[s : s + args.minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb], b_actions[mb]
                    )
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()

                    adv = b_adv[mb]
                    if args.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c, 1 + c)).mean()

                    newvalue = newvalue.view(-1)
                    v_err = (newvalue - b_returns[mb]) ** 2
                    if args.clip_vloss:
                        v_clip = b_values[mb] + (newvalue - b_values[mb]).clamp(-c, c)
                        v_err = torch.max(v_err, (v_clip - b_returns[mb]) ** 2)
                    v_loss = 0.5 * v_err.mean()

                    ent = entropy.mean()
                    loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        stats.append(
                            torch.stack(
                                [
                                    pg_loss, v_loss, ent, (-logratio).mean(),
                                    ((ratio - 1) - logratio).mean(),
                                    ((ratio - 1).abs() > c).float().mean(),
                                    ratio.max(), ratio.min(), gn,
                                ]
                            )
                        )

                if args.target_kl is not None and stats[-1][4] > args.target_kl:
                    break

            stats = torch.stack(stats)
            pg, vl, ent, old_kl, kl, clipfrac, _, _, gn = stats.mean(0).tolist()
            ratio_max, ratio_min = stats[:, 6].max().item(), stats[:, 7].min().item()
            update_time = time.time() - t1

            # ---------- 4. Encode rollout with the updated agent ----------
            t2 = time.time()
            with torch.no_grad():
                flat = obs.flatten(0, 1)
                enc = torch.cat(
                    [agent.encode(flat[i : i + 2048]) for i in range(0, len(flat), 2048)]
                ).view(T, N, -1)
                st_targets = torch.cat([enc[1:], agent.encode(next_obs)[None]])
                # reset memory before a state if the previous transition ended the game
                starts = torch.cat([rollout_start[None], game_overs[:-1].bool()])
            hidden_in = st_hidden  # frozen rollout-boundary memory for all epochs

            # ---------- 5. Train StateTransition on full per-env sequences ----------
            env_mb, st_stats = N // args.st_num_minibatches, []
            for _ in range(args.st_epochs):
                perm = torch.randperm(N, device=device)
                for s in range(0, N, env_mb):
                    mb = perm[s : s + env_mb]
                    pred, logits, pred_r, _ = st(
                        enc[:, mb], actions[:, mb],
                        (hidden_in[0][mb], hidden_in[1][mb]), starts[:, mb],
                    )
                    st_loss, recon, bce, rew = state_transition_loss(
                        pred, logits, pred_r, st_targets[:, mb], game_overs[:, mb], rewards[:, mb],
                        args.st_done_coef, args.st_reward_coef,
                    )
                    st_optimizer.zero_grad()
                    st_loss.backward()
                    nn.utils.clip_grad_norm_(st.parameters(), args.max_grad_norm)
                    st_optimizer.step()
                    with torch.no_grad():
                        acc = ((logits > 0).float() == game_overs[:, mb]).float().mean()
                        st_stats.append(torch.stack([st_loss, recon, bce, acc, rew]))

            # ---------- 6. Replay with updated weights -> memory for next rollout ----------
            with torch.no_grad():
                st_hidden = st(enc, actions, hidden_in, starts)[3]
            st_time = time.time() - t2
            st_l, st_recon, st_bce, st_acc, st_rew = torch.stack(st_stats).mean(0).tolist()

            # ---------- 7. Logging ----------
            var_y = torch.var(b_returns, correction=0)
            explained_var = (
                float("nan") if var_y == 0
                else 1 - (torch.var(b_returns - b_values, correction=0) / var_y).item()
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
                "losses/explained_variance": explained_var,
                "losses/grad_norm": gn,
                "losses/ratio_max": ratio_max,
                "losses/ratio_min": ratio_min,
                "state_transition/loss": st_l,
                "state_transition/recon_loss": st_recon,
                "state_transition/done_bce": st_bce,
                "state_transition/reward_bce": st_rew,
                "state_transition/done_accuracy": st_acc,
                "state_transition/game_over_fraction": game_overs.mean().item(),
                "state_transition/memory_h_norm": st_hidden[0].norm().item(),
                "state_transition/memory_c_norm": st_hidden[1].norm().item(),
            }
            for k, v in logs.items():
                writer.add_scalar(k, v, global_step)

            if args.hist_every > 0 and iteration % args.hist_every == 0:
                for name, p in agent.named_parameters():
                    writer.add_histogram(f"weights/{name}", p.detach().cpu(), global_step)
                    if p.grad is not None:
                        writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), global_step)

            print(
                f"Iteration {iteration} | SPS: {sps} | "
                f"Avg return: {np.mean(avg_returns) if avg_returns else float('nan'):.1f} | "
                f"ST loss: {st_l:.5f} | Memory norm: {logs['state_transition/memory_h_norm']:.3f}"
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
