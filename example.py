"""
example.py — Demo and sanity-check for the arm+hand scene.

Without --record  : prints env dimensions and steps 5 physics seconds (no render).
With    --record  : sweeps all 7 arm joints through their full range and saves
                    workspace_demo.mp4.  This is the visual demo of the scene.

Usage:
    MUJOCO_GL=egl python3 example.py
    MUJOCO_GL=egl python3 example.py --record
"""

import os
import argparse
import numpy as np
import mujoco
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")

import arm_hand_env  # noqa: F401 — registers ArmHand-v0
from arm_hand_env import _build_model, _ARM_JOINT_NAMES, _ARM_HOME, _HAND_OPEN


# ── Sanity check ────────────────────────────────────────────────────────────────

def check():
    import gymnasium as gym
    env = gym.make("ArmHand-v0")
    print("Observation space:", env.observation_space.shape)
    print("Action space     :", env.action_space.shape)
    obs, _ = env.reset()
    print(f"Reset OK — arm qpos: {obs[:7].round(3)}")
    for _ in range(250):                        # 250 steps × 10 ms = 2.5 s
        env.step(np.zeros(env.action_space.shape))
    print("250 steps OK")
    env.close()


# ── Joint-sweep demo ─────────────────────────────────────────────────────────────

JOINT_LABELS = {
    "joint1": "Base rotation",
    "joint2": "Shoulder flex",
    "joint3": "Upper arm roll",
    "joint4": "Elbow flex",
    "joint5": "Forearm roll",
    "joint6": "Wrist flex",
    "joint7": "Wrist roll",
}

FRAMES_HOLD  = 25
FRAMES_SWEEP = 90
FPS          = 50
W, H         = 640, 480


def record():
    import mediapy

    model = _build_model()
    data  = mujoco.MjData(model)

    arm_jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES]
    arm_addrs  = [model.jnt_qposadr[j] for j in arm_jids]
    jnt_ranges = [(model.jnt_range[j, 0], model.jnt_range[j, 1]) for j in arm_jids]

    hand_jnames = [
        "allegro/ffj0", "allegro/ffj1", "allegro/ffj2", "allegro/ffj3",
        "allegro/mfj0", "allegro/mfj1", "allegro/mfj2", "allegro/mfj3",
        "allegro/rfj0", "allegro/rfj1", "allegro/rfj2", "allegro/rfj3",
        "allegro/thj0", "allegro/thj1", "allegro/thj2", "allegro/thj3",
    ]
    hand_jids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hand_jnames]
    hand_addrs = [model.jnt_qposadr[j] for j in hand_jids]
    hand_lo    = [model.jnt_range[j, 0] for j in hand_jids]
    hand_hi    = [model.jnt_range[j, 1] for j in hand_jids]

    def set_pose(arm_angles):
        for i, addr in enumerate(arm_addrs):
            data.qpos[addr] = np.clip(arm_angles[i], jnt_ranges[i][0], jnt_ranges[i][1])
        for i, addr in enumerate(hand_addrs):
            data.qpos[addr] = np.clip(_HAND_OPEN[i], hand_lo[i], hand_hi[i])
        obj_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        obj_addr = model.jnt_qposadr[obj_jid]
        data.qpos[obj_addr:obj_addr + 3] = [0, 0, -2.0]
        mujoco.mj_forward(model, data)

    def make_frame(arm_angles, cam, line1, line2):
        set_pose(arm_angles)
        renderer.update_scene(data, camera=cam)
        img  = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 58], fill=(0, 0, 0, 200))
        draw.text((10,  6), line1, fill=(255, 220, 80))
        draw.text((10, 30), line2, fill=(200, 200, 200))
        return np.array(img)

    renderer = mujoco.Renderer(model, height=H, width=W)
    frames   = []

    # Scene overview
    set_pose(_ARM_HOME)
    renderer.update_scene(data, camera="top_down")
    overview_frame = np.array(Image.fromarray(renderer.render().copy()))
    img  = Image.fromarray(overview_frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 58], fill=(0, 0, 0, 200))
    draw.text((10,  6), "Workspace overview — top-down", fill=(255, 220, 80))
    draw.text((10, 30), "8 tables at 45° intervals, heights 0.22–0.56 m", fill=(200, 200, 200))
    for _ in range(60):
        frames.append(np.array(img))

    # Per-joint sweeps (joints 1–5 use "fixed" camera, 6–7 use "hand_close")
    cameras = ["fixed"] * 5 + ["hand_close"] * 2

    for idx, jname in enumerate(_ARM_JOINT_NAMES):
        lo, hi   = jnt_ranges[idx]
        label    = JOINT_LABELS[jname]
        cam      = cameras[idx]
        home_val = _ARM_HOME[idx]

        def f(angle, status):
            return make_frame(
                np.array([angle if i == idx else _ARM_HOME[i] for i in range(7)]),
                cam,
                f"Joint {idx+1}/7  {jname}  —  {label}",
                f"{status}   range {np.degrees(lo):+.0f}° → {np.degrees(hi):+.0f}°"
                f"   now {np.degrees(angle):+.1f}°",
            )

        for _ in range(FRAMES_HOLD):
            frames.append(f(home_val, "home"))
        for t in range(FRAMES_SWEEP):
            frames.append(f(lo + t / (FRAMES_SWEEP - 1) * (hi - lo), "sweeping"))
        for t in range(FRAMES_HOLD // 2):
            frames.append(f(hi, "at max"))
        for t in range(FRAMES_HOLD // 2):
            alpha = t / max(FRAMES_HOLD // 2 - 1, 1)
            frames.append(f(hi + alpha * (home_val - hi), "returning"))

        print(f"  [{idx+1}/7] {jname}  {np.degrees(lo):+.0f}° → {np.degrees(hi):+.0f}°")

    renderer.close()
    mediapy.write_video("workspace_demo.mp4", frames, fps=FPS)
    print(f"Saved {len(frames)} frames → workspace_demo.mp4  ({len(frames)/FPS:.1f} s)")


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true",
                        help="render joint-sweep demo → workspace_demo.mp4")
    args = parser.parse_args()

    if args.record:
        record()
    else:
        check()
