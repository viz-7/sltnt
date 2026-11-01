"""Load a checkpoint, play one full Atari game, render a GIF, and compare the
StateTransition model's one-step predictions (next state, done, reward) to reality.

With --sltnt the actor primarily sees StateTransition-predicted states.
The first frame is encoded once, and the ST rolls the state forward.

Every 10 steps, the real environment observation is encoded again. This
corrected state is passed to both the actor and StateTransition model.

Usage:
    python src/sltnt/test.py --checkpoint runs/<run_name>/ckpt.pt
    python src/sltnt/test.py --checkpoint runs/<run_name>/ckpt.pt --sltnt
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
    """re-encode the real observation every N steps in --sltnt mode"""


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

    if args.sync_every < 1:
        sys.exit("--sync-every must be >= 1.")

    # Two pools, same seed and same actions.
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

    agent = Agent(n_actions)
    agent.load_state_dict(ckpt["agent"])
    agent.to(device).eval()

    st = StateTransition(
        n_actions,
        agent.actor.in_features,
        ckpt_args.get("st_hidden_dim", 512),
    )

    st.load_state_dict(ckpt["state_transition"])
    st.to(device).eval()

    # LSTM memory is maintained throughout the game.
    hidden = st.init_hidden(1, device)

    fire = (
        args.fire_action
        if 0 <= args.fire_action < n_actions
        else None
    )

    obs, info = env.reset()
    view_obs, _ = view_env.reset()

    frames = [to_rgb(view_obs)]
    captions = [""]

    lives = info["lives"][0]

    force_fire = fire is not None

    since_reward = 0
    score = 0.0
    steps = 0

    end_reason = "max_steps reached"

    mses = []
    done_hits = []
    r_errs = []
    r_hits = []

    # Initial real observation encoding.
    with torch.no_grad():
        state = agent.encode(
            torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            )
        )

    print(
        f"\n{'step':>5} {'act':>3} {'state MSE':>10} "
        f"{'pred done':>10} {'pred R':>7} {'true R':>7} "
        f"{'sync':>6}"
    )

    for steps in range(1, args.max_steps + 1):

        with torch.no_grad():

            # Actor chooses an action using the current state.
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

            # StateTransition predicts the next state.
            pred_state, done_logit, pred_r, hidden = st(
                state[None],
                torch.tensor([[action]], device=device),
                hidden,
            )

        # Execute the same action in the actual environment.
        actions = np.array([action])

        obs, reward, terminated, truncated, info = env.step(actions)

        view_obs, *_ = view_env.step(actions)

        raw_r = float(reward[0])
        score += raw_r

        true_r = float(raw_r > 0)

        since_reward = (
            0 if raw_r > 0 else since_reward + 1
        )

        game_over = bool(terminated[0])

        with torch.no_grad():

            # Encode the actual next observation for diagnostics.
            # Also used as the actor/ST input at synchronization points.
            true_next = agent.encode(
                torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                )
            )

            mse = (
                (pred_state[0, 0] - true_next[0]) ** 2
            ).mean().item()

            done_p = torch.sigmoid(
                done_logit[0, 0]
            ).item()

            pr = torch.sigmoid(
                pred_r[0, 0]
            ).item()

            # Synchronization logic:
            #
            # Every N steps, replace the predicted state with
            # the encoding of the real environment observation.
            #
            # This corrected state becomes the input to BOTH
            # the actor and StateTransition on the next iteration.

            synced = (
                args.sltnt
                and steps % args.sync_every == 0
            )

            if args.sltnt:
                if synced:
                    state = true_next
                else:
                    state = pred_state[0]
            else:
                state = true_next

        if not game_over:
            mses.append(mse)

        done_hits.append(
            (done_p > 0.5) == game_over
        )

        r_errs.append(
            -np.log(
                (pr if true_r else 1 - pr) + 1e-8
            )
        )

        r_hits.append(
            (pr > 0.5) == bool(true_r)
        )

        if (
            steps % args.print_every == 0
            or game_over
            or true_r != 0
        ):
            print(
                f"{steps:>5} {action:>3} {mse:>10.5f} "
                f"{done_p:>10.3f} {pr:>7.3f} "
                f"{true_r:>7g} "
                f"{'YES' if synced else '':>6}"
            )

        # Handle lives and automatic FIRE actions.
        new_lives = info["lives"][0]

        if fire is not None:

            if new_lives < lives:
                force_fire = True

            if since_reward >= args.stall_steps:
                force_fire = True
                since_reward = 0

        lives = new_lives

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
                f"R {pr:.2f}/{true_r:g}"
                f"\n{'SYNC' if synced else ''}"
            )

        if end:

            end_reason = (
                "game over (all lives lost)"
                if game_over
                else "truncated by the env"
            )

            break

    env.close()
    view_env.close()

    print(
        f"\nRollout finished: {end_reason} "
        f"after {steps} steps, score={score:g}"
    )

    if mses:
        print(
            f"State MSE (non-terminal): "
            f"mean={np.mean(mses):.5f} "
            f"median={np.median(mses):.5f} "
            f"max={np.max(mses):.5f}"
        )

    print(
        f"Done accuracy: {np.mean(done_hits):.4f} | "
        f"Reward BCE: {np.mean(r_errs):.4f} | "
        f"Reward accuracy: {np.mean(r_hits):.4f}"
    )

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
            to_gif_frame(f, args.scale, c)
            for f, c in zip(frames[1:], captions[1:])
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
