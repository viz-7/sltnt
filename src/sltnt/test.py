"""Load a PPO checkpoint, play one full Atari game, and render it to a big GIF.

Usage:
    python test.py --checkpoint runs/<run_name>/ckpt.pt
    python test.py --checkpoint runs/<run_name>/ckpt.pt --scale 6 --out breakout.gif

No extra dependencies: envpool (already used for training) plays the game and
Pillow (installed with tensorboard) writes the GIF.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

# allow `python test.py` from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import envpool  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
from PIL import Image  # noqa: E402

from sltnt.model import Agent  # noqa: E402


@dataclass
class Args:
    checkpoint: str
    """path to the checkpoint saved by ppo.py (runs/<run_name>/ckpt.pt)"""
    env_id: Optional[str] = None
    """envpool id, e.g. Breakout-v5. Defaults to the one stored in the checkpoint"""
    out: str = "rollout.gif"
    """output gif path"""
    scale: int = 4
    """integer upscaling factor for the gif (native frame is 160x210, so 4 -> 640x840)"""
    fps: int = 15
    """gif playback speed in frames per second (one frame per agent step = 4 emulator frames, so 15 is real time)"""
    gif_every: int = 1
    """keep every N-th agent step in the gif (raise it to shrink long games)"""
    max_steps: int = 4500
    """safety cap on agent steps; the rollout stops here if the game has not ended"""
    fire_action: int = 1
    """index of the FIRE action used to launch the ball at the start and after a lost life (-1 disables)"""
    stall_steps: int = 100
    """press FIRE if no reward arrives for this many steps (some agents never relaunch the ball)"""
    stochastic: bool = False
    """sample actions from the policy instead of taking the argmax"""
    seed: int = 1
    """env seed"""
    cuda: bool = True
    """use cuda if available"""


def make_env(env_id: str, seed: int, **kwargs):
    # Same defaults as training (84x84 grayscale, 4 stacked frames, frame skip 4), except that
    # episodic life and reward clipping are training tricks, so they are off: we play the real, full game.
    return envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=1,
        episodic_life=False,
        reward_clip=False,
        seed=seed,
        **kwargs,
    )


def load_agent(ckpt, n_actions: int, device: torch.device) -> Agent:
    state = ckpt["agent"] if isinstance(ckpt, dict) and "agent" in ckpt else ckpt
    try:
        agent = Agent(n_actions)
    except (TypeError, AttributeError):
        # model.py still takes an `envs` object, as in the original cleanrl script
        agent = Agent(SimpleNamespace(single_action_space=SimpleNamespace(n=n_actions)))
    agent.load_state_dict(state)
    return agent.to(device).eval()


def to_rgb(obs: np.ndarray) -> np.ndarray:
    """envpool RGB observation (1 env, channels first) -> HxWx3 uint8 frame."""
    frame = np.asarray(obs)[0]
    if frame.shape[-1] == 3:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(frame[-3:].transpose(1, 2, 0))


def to_gif_frame(frame: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(frame)
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)  # crisp pixels, no blur
    return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    env_id = args.env_id or (ckpt.get("args", {}).get("env_id") if isinstance(ckpt, dict) else None) or "Breakout-v5"
    print(f"env={env_id} checkpoint={args.checkpoint} device={device}")

    # Two pools with the same seed that receive the same actions, so they play the identical game:
    #   env      -> what the agent sees (84x84 grayscale, 4 stacked frames)
    #   view_env -> full-resolution 160x210 RGB, used only for the GIF
    env = make_env(env_id, args.seed)
    view_env = make_env(env_id, args.seed, img_height=210, img_width=160, gray_scale=False, stack_num=1)
    agent = load_agent(ckpt, env.action_space.n, device)
    fire = args.fire_action if 0 <= args.fire_action < env.action_space.n else None

    obs, info = env.reset()
    view_obs, _ = view_env.reset()
    frames = [to_rgb(view_obs)]
    lives = info["lives"][0]
    force_fire = fire is not None  # launch the ball at the start of the game
    since_reward = 0
    score = 0.0
    end_reason = "max_steps reached"
    steps = 0

    for steps in range(1, args.max_steps + 1):
        if force_fire:
            action = fire
            force_fire = False
        else:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = agent.actor(agent.network(x / 255.0))
                if args.stochastic:
                    action = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    action = logits.argmax(dim=-1).item()

        actions = np.array([action])
        obs, reward, terminated, truncated, info = env.step(actions)
        view_obs, *_ = view_env.step(actions)
        score += float(reward[0])
        since_reward = 0 if reward[0] > 0 else since_reward + 1

        # after losing a life the ball must be relaunched with FIRE
        new_lives = info["lives"][0]
        if fire is not None:
            if new_lives < lives:
                force_fire = True
            if since_reward >= args.stall_steps:
                force_fire = True
                since_reward = 0
        lives = new_lives

        game_over = bool(terminated[0] or truncated[0])
        if steps % args.gif_every == 0 or game_over:
            frames.append(to_rgb(view_obs))
        if game_over:
            end_reason = "game over (all lives lost)" if terminated[0] else "truncated by the env"
            break

    env.close()
    view_env.close()
    print(f"Rollout finished: {end_reason} after {steps} steps, score={score:g}")

    # GIF: frames are converted lazily so the big upscaled images never sit in memory all at once
    frame_ms = max(10, int(round(1000 / args.fps * args.gif_every)))
    durations = [frame_ms] * (len(frames) - 1) + [2000]  # hold the final frame for 2 s
    first = to_gif_frame(frames[0], args.scale)
    first.save(
        args.out,
        save_all=True,
        append_images=(to_gif_frame(f, args.scale) for f in frames[1:]),
        duration=durations,
        loop=0,
        optimize=False,
    )
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"Saved {args.out}: {len(frames)} frames, {first.width}x{first.height}, {size_mb:.1f} MB")
