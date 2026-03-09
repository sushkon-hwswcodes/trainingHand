"""
example.py — Sanity-check for ArmHandEnv (ArmHand-v0).

Usage:
    MUJOCO_GL=egl python3 example.py           # stats only
    MUJOCO_GL=egl python3 example.py --record  # save output.mp4 + frame.png
"""

import os
import argparse
import numpy as np
import gymnasium as gym

os.environ.setdefault("MUJOCO_GL", "egl")

import arm_hand_env  # noqa: F401 — registers ArmHand-v0


def run(record: bool = False, n_steps: int = 100):
    render_mode = "rgb_array" if record else None
    env = gym.make("ArmHand-v0", render_mode=render_mode)

    print("Observation space:", env.observation_space.shape)
    print("Action space     :", env.action_space.shape)

    obs, _ = env.reset()
    print(f"Reset OK — arm qpos: {obs[:7].round(3)}")

    frames = []
    for step in range(n_steps):
        action = np.zeros(env.action_space.shape)  # hold pose
        obs, reward, terminated, truncated, info = env.step(action)

        if record:
            frames.append(env.render())

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break

    env.close()
    print(f"Ran {step + 1} steps")

    if record and frames:
        import mediapy
        from PIL import Image
        mediapy.write_video("output.mp4", frames, fps=50)
        Image.fromarray(frames[0]).save("frame.png")
        print(f"Saved {len(frames)} frames → output.mp4, frame.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--steps",  type=int, default=100)
    args = parser.parse_args()
    run(record=args.record, n_steps=args.steps)
