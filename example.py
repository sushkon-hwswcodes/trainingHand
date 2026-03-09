"""
example.py — Sanity-check and demo for ArmHandCubeEnv (ArmHandCube-v0).

Usage:
    MUJOCO_GL=egl python3 example.py           # stats only
    MUJOCO_GL=egl python3 example.py --record  # save output.mp4 + frame.png
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")

import arm_hand_env  # noqa: F401 — registers ArmHandCube-v0


def run(record: bool = False, n_steps: int = 200):
    render_mode = "rgb_array" if record else None
    env = gym.make("ArmHandCube-v0", render_mode=render_mode)

    print("Observation space:", env.observation_space.shape)
    print("Action space     :", env.action_space.shape)
    print("Max steps        :", env.spec.max_episode_steps)

    obs, info = env.reset()
    print(f"\nReset OK — obs[:7] (arm qpos): {obs[:7].round(3)}")
    print(f"          obs[7:14] (arm qvel): {obs[7:14].round(3)}")

    frames   = []
    rewards  = []
    contacts = []

    for step in range(n_steps):
        # Zero action = hold arm+hand at current actuator targets
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        contacts.append(info.get("n_contacts", 0))

        if record:
            frame = env.render()
            # Overlay step info
            from PIL import ImageDraw
            img  = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 640, 30], fill=(0, 0, 0, 180))
            draw.text((8, 6), f"step {step+1:3d}  err={info.get('orient_err_deg', 0):.1f}°  contacts={info.get('n_contacts',0)}  r={reward:.2f}", fill=(255, 220, 80))
            frames.append(np.array(img))

        if terminated or truncated:
            print(f"Episode ended at step {step+1}: terminated={terminated} truncated={truncated}")
            break

    env.close()

    print(f"\n── Stats ({len(rewards)} steps) ──")
    print(f"  mean reward  : {np.mean(rewards):.3f}")
    print(f"  total reward : {np.sum(rewards):.3f}")
    print(f"  mean contacts: {np.mean(contacts):.1f}")

    if record and frames:
        import mediapy
        mediapy.write_video("output.mp4", frames, fps=50)
        Image.fromarray(frames[-1]).save("frame.png")
        print(f"\nSaved {len(frames)} frames → output.mp4")
        print("Saved last frame  → frame.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="render and save video")
    parser.add_argument("--steps",  type=int, default=200)
    args = parser.parse_args()
    run(record=args.record, n_steps=args.steps)
