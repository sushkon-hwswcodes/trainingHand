# trainingHand

Franka Panda arm + Wonik Allegro dexterous hand simulation in MuJoCo.
Goal: train a pick-and-place policy using reinforcement learning.

## Robot

- **Arm**: Franka Emika Panda — 7 DOF
- **Hand**: Wonik Allegro (right) — 16 DOF
- **Total**: 23 actuated joints
- Models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## Files

| File | Purpose |
|------|---------|
| `arm_hand_env.py` | Gymnasium environment (`ArmHandCube-v0`) |
| `train.py` | PPO training via Stable-Baselines3 |
| `example.py` | Quick sanity-check / demo |
| `franka_panda/` | Panda arm MJCF + mesh assets |
| `allegro_hand/` | Allegro hand MJCF + mesh assets |

## Setup

```bash
pip install mujoco gymnasium stable-baselines3 tensorboard pillow mediapy
```

## Usage

```bash
# Verify the environment loads
MUJOCO_GL=egl python3 example.py

# Train
python3 train.py --timesteps 2e6 --n-envs 4

# Monitor training
tensorboard --logdir logs

# Evaluate saved policy
python3 train.py --eval
```

## Environment

**Observation** (57-dim): arm qpos/qvel (14) · hand qpos/qvel (32) · cube pos rel palm (3) · cube quat (4) · target quat (4)
**Action** (23-dim, normalized [-1, 1]): first 7 = arm joints, last 16 = hand joints
**Reward**: orientation matching + fingertip contact bonus + drop penalty
