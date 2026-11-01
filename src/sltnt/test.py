"""Load a checkpoint, play an Atari game, render a GIF, and evaluate the
StateTransition model's one-step next-state, done, and reward predictions.

The StateTransition model is a two-layer FFN. Its input is:
    [state(t-2), state(t-1), state(t), one_hot(action(t))]

The initial temporal context is [s0, s0, s0]. In --sltnt mode the actor uses
predicted states, with an optional real-state correction every --sync-every.

Usage:
    python src/sltnt/test.py --checkpoint runs/<run_name>/ckpt.pt
    python src/sltnt/test.py --checkpoint runs/<run_name>/ckpt.pt --sltnt
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import envpool
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from sltnt.model import Agent, StateTransition


@dataclass
class Args:
    checkpoint: str
    sltnt: bool = False
    """actor acts on predicted states, with periodic real-state correction"""
    env_id: Optional[str] = None
    out: str = "rollout.gif"
    scale: int = 4
    fps: int = 15
    gif_every: int = 1
    print_every: int = 1
    max_steps: int = 4500
    fire_action: int = 1
    stall_steps: int = 100
    stochastic: bool = False
    seed: int = 1
    cuda: bool = True
    sync_every: int = 10


def make_env(env_id, seed, **kwargs):
    return envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=1,
        episodic_life=False,
        reward_clip=False,
        seed=seed,
        **kwargs,
    )


def to_rgb(obs):
    frame = np.asarray(obs)[0]

    if frame.shape[-1] == 3:
        return np.ascontiguousarray(frame)

    return np.ascontiguousarray(frame[-3:].transpose(1, 2, 0))


def to_gif_frame(frame, scale, text=""):
    img = Image.fromarray(frame)
    img = img.resize(
        (img.width * scale, img.height * scale),
        Image.Resampling.NEAREST,
    )

    if text:
        try:
            font = ImageFont.load_default(size=max(10, 3 * scale))
        except TypeError:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        box = draw.multiline_textbbox((4, 4), text, font=font)

        draw.rectangle(
            (0, 0, box[2] + 4, box[3] + 4),
            fill=(0, 0, 0),
        )

        draw.multiline_text(
            (4, 4),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    return img.convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=64,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.sync_every < 1:
        sys.exit("--sync-every must be >= 1.")

    if args.gif_every < 1:
        sys.exit("--gif-every must be >= 1.")

    if args.fps < 1:
        sys.exit("--fps must be >= 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    env_id = args.env_id or ckpt_args.get("env_id") or "Breakout-v5"

    mode = (
        f"sltnt (real-state sync every {args.sync_every} steps)"
        if args.sltnt
        else "real states"
    )

    print(
        f"env={env_id} checkpoint={args.checkpoint} "
        f"device={device} mode={mode}"
    )

    if "state_transition" not in ckpt:
        sys.exit("Checkpoint has no 'state_transition' weights.")

    # ---------------------------------------------------------
    # Environments
    # ---------------------------------------------------------

    env = make_env(env_id, args.seed)

    view_env = make_env(
        env_id,
        args.seed,
        img_height=210,
        img_width=160,
        gray_scale=False,
        stack_num=1,
    )

    n_actions = env.action_space.n

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    agent = Agent(n_actions)
    agent.load_state_dict(ckpt["agent"])
    agent.to(device).eval()

    st = StateTransition(
        n_actions,
        agent.actor.in_features,
        ckpt_args.get("st_hidden_dim", 512),
    )

    try:
        st.load_state_dict(ckpt["state_transition"])
    except RuntimeError as exc:
        env.close()
        view_env.close()

        sys.exit(
            "The checkpoint's StateTransition weights do not match "
            "the two-layer FFN architecture. Train and save a new "
            "checkpoint first.\n"
            f"{exc}"
        )

    st.to(device).eval()

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------

    obs, info = env.reset()
    view_obs, _ = view_env.reset()

    frames = [to_rgb(view_obs)]
    captions = [""]

    lives = info["lives"][0]

    fire = (
        args.fire_action
        if 0 <= args.fire_action < n_actions
        else None
    )

    force_fire = fire is not None

    since_reward = 0
    score = 0.0
    steps = 0

    end_reason = "max_steps reached"

    mses = []
    done_hits = []
    reward_errors = []
    reward_hits = []

    # Initial context: [s0, s0, s0].
    #
    # history contains [s(t-2), s(t-1)].
    # state contains s(t).

    with torch.no_grad():
        state = agent.encode(
            torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            )
        )

        history = (
            state.unsqueeze(1)
            .expand(-1, 2, -1)
            .clone()
        )

    print(
        f"\n{'step':>5} {'act':>3} {'state MSE':>10} "
        f"{'pred done':>10} {'pred R':>7} "
        f"{'true R':>7} {'sync':>6}"
    )

    # ---------------------------------------------------------
    # Main rollout
    # ---------------------------------------------------------

    for steps in range(1, args.max_steps + 1):

        with torch.no_grad():

            # Actor chooses action using current state.
            if force_fire:
                action = fire
                force_fire = False

            else:
                logits = agent.actor(state)

                if args.stochastic:
                    action = (
                        torch.distributions.Categorical(
                            logits=logits
                        ).sample().item()
                    )
                else:
                    action = logits.argmax(-1).item()

            # Build [s(t-2), s(t-1), s(t)].
            state_window = torch.cat(
                [
                    history,
                    state.unsqueeze(1),
                ],
                dim=1,
            )

            action_tensor = torch.tensor(
                [action],
                dtype=torch.long,
                device=device,
            )

            # FFN prediction.
            pred_state, done_logit, pred_reward = st(
                state_window,
                action_tensor,
            )

        # -----------------------------------------------------
        # Step real environment
        # -----------------------------------------------------

        actions_np = np.array([action])

        obs, reward, terminated, truncated, info = env.step(
            actions_np
        )

        view_obs, *_ = view_env.step(actions_np)

        raw_reward = float(reward[0])
        score += raw_reward

        true_reward = float(raw_reward > 0)

        since_reward = (
            0 if raw_reward > 0
            else since_reward + 1
        )

        game_over = bool(terminated[0])

        # -----------------------------------------------------
        # Compare prediction against reality
        # -----------------------------------------------------

        with torch.no_grad():

            true_next = agent.encode(
                torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                )
            )

            mse = (
                (pred_state[0] - true_next[0]) ** 2
            ).mean().item()

            done_p = torch.sigmoid(
                done_logit[0]
            ).item()

            reward_p = torch.sigmoid(
                pred_reward[0]
            ).item()

            synced = (
                args.sltnt
                and steps % args.sync_every == 0
            )

            # -------------------------------------------------
            # Update temporal context
            # -------------------------------------------------

            if game_over:

                # New episode: initialize all three states
                # using the new episode's reset observation.

                state = true_next

                history = (
                    true_next.unsqueeze(1)
                    .expand(-1, 2, -1)
                    .clone()
                )

            else:

                # Shift:
                #
                # [s(t-2), s(t-1)] -> [s(t-1), s(t)]

                history = torch.cat(
                    [
                        history[:, 1:],
                        state.unsqueeze(1),
                    ],
                    dim=1,
                )

                # Actor and ST use the same current state.
                if args.sltnt:

                    state = (
                        true_next
                        if synced
                        else pred_state
                    )

                else:
                    state = true_next

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        if not game_over:
            mses.append(mse)

        done_hits.append(
            (done_p > 0.5) == game_over
        )

        reward_errors.append(
            -np.log(
                (reward_p if true_reward else 1.0 - reward_p)
                + 1e-8
            )
        )

        reward_hits.append(
            (reward_p > 0.5) == bool(true_reward)
        )

        if (
            steps % args.print_every == 0
            or game_over
            or true_reward != 0
        ):
            print(
                f"{steps:>5} {action:>3} {mse:>10.5f} "
                f"{done_p:>10.3f} {reward_p:>7.3f} "
                f"{true_reward:>7g} "
                f"{'YES' if synced else '':>6}"
            )

        # -----------------------------------------------------
        # Lives and FIRE handling
        # -----------------------------------------------------

        new_lives = info["lives"][0]

        if fire is not None:

            if new_lives < lives:
                force_fire = True

            if since_reward >= args.stall_steps:
                force_fire = True
                since_reward = 0

        lives = new_lives

        # -----------------------------------------------------
        # GIF frames
        # -----------------------------------------------------

        end = (
            game_over
            or bool(truncated[0])
        )

        if steps % args.gif_every == 0 or end:

            frames.append(to_rgb(view_obs))

            captions.append(
                f"step {steps}\n"
                f"MSE {mse:.4f}\n"
                f"done {done_p:.2f}\n"
                f"R {reward_p:.2f}/{true_reward:g}"
                f"\n{'SYNC' if synced else ''}"
            )

        if end:

            end_reason = (
                "game over (all lives lost)"
                if game_over
                else "truncated by the env"
            )

            break

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    env.close()
    view_env.close()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print(
        f"\nRollout finished: {end_reason} "
        f"after {steps} steps, score={score:g}"
    )

    if mses:
        print(
            "State MSE (non-terminal): "
            f"mean={np.mean(mses):.5f} "
            f"median={np.median(mses):.5f} "
            f"max={np.max(mses):.5f}"
        )

    print(
        f"Done accuracy: {np.mean(done_hits):.4f} | "
        f"Reward BCE: {np.mean(reward_errors):.4f} | "
        f"Reward accuracy: {np.mean(reward_hits):.4f}"
    )

    # ---------------------------------------------------------
    # Save GIF
    # ---------------------------------------------------------

    frame_ms = max(
        10,
        int(round(1000 / args.fps * args.gif_every)),
    )

    durations = (
        [frame_ms] * (len(frames) - 1)
        + [2000]
    )

    first = to_gif_frame(
        frames[0],
        args.scale,
        captions[0],
    )

    first.save(
        args.out,
        save_all=True,
        append_images=(
            to_gif_frame(frame, args.scale, caption)
            for frame, caption in zip(
                frames[1:],
                captions[1:],
            )
        ),
        duration=durations,
        loop=0,
        optimize=False,
    )

    print(
        f"Saved {args.out}: {len(frames)} frames, "
        f"{first.width}x{first.height}, "
        f"{Path(args.out).stat().st_size / 1e6:.1f} MB"
    )
