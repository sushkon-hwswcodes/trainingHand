"""
render_workspace.py — Render a video showing the Franka Panda arm sweeping each
of its 7 joints through the full range of motion, inside the table scene.

Uses pure forward kinematics (mj_forward, no physics) for clean geometric motion.

Output: workspace_demo.mp4

Usage:
    MUJOCO_GL=egl python3 render_workspace.py
"""

import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import mediapy

os.environ.setdefault("MUJOCO_GL", "egl")

from arm_hand_env import _build_model, _ARM_JOINT_NAMES, _ARM_HOME, _HAND_OPEN

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "workspace_demo.mp4"
FPS          = 50
FRAMES_HOLD  = 25   # frames held at start/end of each sweep
FRAMES_SWEEP = 90   # frames for one full min→max sweep
WIDTH, HEIGHT = 640, 480

JOINT_LABELS = {
    "joint1": "Base rotation",
    "joint2": "Shoulder flex",
    "joint3": "Upper arm roll",
    "joint4": "Elbow flex",
    "joint5": "Forearm roll",
    "joint6": "Wrist flex",
    "joint7": "Wrist roll",
}

# ── Build model ─────────────────────────────────────────────────────────────────
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
hand_jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hand_jnames]
hand_addrs  = [model.jnt_qposadr[j] for j in hand_jids]
hand_lo     = [model.jnt_range[j, 0] for j in hand_jids]
hand_hi     = [model.jnt_range[j, 1] for j in hand_jids]


def set_pose(arm_angles: np.ndarray):
    """Set arm joints directly (FK only) and hold hand open."""
    for i, addr in enumerate(arm_addrs):
        data.qpos[addr] = np.clip(arm_angles[i], jnt_ranges[i][0], jnt_ranges[i][1])
    for i, addr in enumerate(hand_addrs):
        data.qpos[addr] = np.clip(_HAND_OPEN[i], hand_lo[i], hand_hi[i])
    # Park object far below floor so it doesn't interfere
    obj_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
    obj_addr = model.jnt_qposadr[obj_jid]
    data.qpos[obj_addr:obj_addr + 3] = [0, 0, -2.0]
    mujoco.mj_forward(model, data)


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, 58], fill=(0, 0, 0, 200))
    draw.text((10,  6), lines[0], fill=(255, 220,  80))
    draw.text((10, 30), lines[1], fill=(200, 200, 200))
    return np.array(img)


# ── Cameras to use ──────────────────────────────────────────────────────────────
# Alternate between the overview and hand-close cameras per joint
CAMERAS = ["fixed", "fixed", "fixed", "fixed", "fixed", "hand_close", "hand_close"]

# ── Render ──────────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
frames   = []

total_joints = len(_ARM_JOINT_NAMES)
total_frames = total_joints * (FRAMES_HOLD + FRAMES_SWEEP + FRAMES_HOLD)
print(f"Rendering {total_joints} joints × ({FRAMES_HOLD}+{FRAMES_SWEEP}+{FRAMES_HOLD}) frames "
      f"= {total_frames} frames  ({total_frames / FPS:.1f} s)")

# ── Scene overview (top-down, 60 frames) ────────────────────────────────────────
set_pose(_ARM_HOME)
renderer.update_scene(data, camera="top_down")
for _ in range(60):
    frames.append(overlay(renderer.render().copy(), [
        "Workspace overview — top-down",
        "8 tables at 45° intervals, heights 0.22–0.56 m",
    ]))
print("  [overview] done")

# ── Per-joint sweeps ─────────────────────────────────────────────────────────────
for idx, jname in enumerate(_ARM_JOINT_NAMES):
    lo, hi   = jnt_ranges[idx]
    label    = JOINT_LABELS[jname]
    cam      = CAMERAS[idx]
    home_val = _ARM_HOME[idx]

    def frame_for(angle, status):
        pose = _ARM_HOME.copy()
        pose[idx] = angle
        set_pose(pose)
        renderer.update_scene(data, camera=cam)
        return overlay(renderer.render().copy(), [
            f"Joint {idx+1}/7  {jname}  —  {label}",
            f"{status}   {np.degrees(lo):+.0f}° → {np.degrees(hi):+.0f}°   "
            f"now: {np.degrees(angle):+.1f}°",
        ])

    # Hold at home
    for _ in range(FRAMES_HOLD):
        frames.append(frame_for(home_val, "home"))

    # Sweep min → max
    for t in range(FRAMES_SWEEP):
        alpha = t / (FRAMES_SWEEP - 1)
        angle = lo + alpha * (hi - lo)
        frames.append(frame_for(angle, "sweeping"))

    # Hold at max briefly, then return
    for _ in range(FRAMES_HOLD // 2):
        frames.append(frame_for(hi, "max"))
    for t in range(FRAMES_HOLD // 2):
        alpha = t / max(FRAMES_HOLD // 2 - 1, 1)
        angle = hi + alpha * (home_val - hi)
        frames.append(frame_for(angle, "return"))

    print(f"  [{idx+1}/7]  {jname}  {np.degrees(lo):+.0f}° → {np.degrees(hi):+.0f}°")

# ── Return to home (30 frames) ───────────────────────────────────────────────────
set_pose(_ARM_HOME)
renderer.update_scene(data, camera="fixed")
for _ in range(30):
    frames.append(overlay(renderer.render().copy(), [
        "Home pose",
        "All joints at home position",
    ]))

renderer.close()

mediapy.write_video(VIDEO_PATH, frames, fps=FPS)
print(f"\nSaved {len(frames)} frames → {VIDEO_PATH}  ({len(frames)/FPS:.1f} s)")
