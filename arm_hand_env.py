"""
arm_hand_env.py — Franka Panda arm + Wonik Allegro hand simulation.

Robot:
  - Franka Panda (7 DOF arm) from MuJoCo Menagerie
  - Wonik Allegro right hand (16 DOF) attached to Panda flange via MjSpec.attach()
  - Total: 23 DOF, 23 actuators

Observation (46-dim):
  arm joint positions   (7)
  arm joint velocities  (7)
  hand joint positions  (16)
  hand joint velocities (16)

Action (23-dim, normalized [-1, 1]):
  [0:7]  → arm joints  (joint1–7)
  [7:23] → hand joints (allegro/ffj0–thj3)

Reward: not yet defined — returns 0.0 (task TBD in Phase 2+3).
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_DIR      = os.path.dirname(os.path.abspath(__file__))
_ARM_XML  = os.path.join(_DIR, "franka_panda", "panda_nohand.xml")
_HAND_XML = os.path.join(_DIR, "allegro_hand",  "right_hand.xml")

_ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
]
_HAND_JOINT_NAMES = [
    "allegro/ffj0", "allegro/ffj1", "allegro/ffj2", "allegro/ffj3",
    "allegro/mfj0", "allegro/mfj1", "allegro/mfj2", "allegro/mfj3",
    "allegro/rfj0", "allegro/rfj1", "allegro/rfj2", "allegro/rfj3",
    "allegro/thj0", "allegro/thj1", "allegro/thj2", "allegro/thj3",
]
_ALL_JOINT_NAMES = _ARM_JOINT_NAMES + _HAND_JOINT_NAMES

_N_ARM  = len(_ARM_JOINT_NAMES)   # 7
_N_HAND = len(_HAND_JOINT_NAMES)  # 16
_N_DOF  = _N_ARM + _N_HAND        # 23

# Arm home pose — palm at ≈[0.505, 0, 0.383], facing downward
_ARM_HOME = np.array([0.0, -0.1, 0.0, -2.167, 0.0, 2.0, 0.785])

# Hand open pose
_HAND_OPEN = np.array([
    0.0, 0.0, 0.0, 0.0,   # index  — fully open
    0.0, 0.0, 0.0, 0.0,   # middle — fully open
    0.0, 0.0, 0.0, 0.0,   # ring   — fully open
    0.3, 0.4, 0.0, 0.0,   # thumb  — rotated out, not flexed
])

_MAX_STEPS = 500


def _build_model() -> mujoco.MjModel:
    """Compose Franka arm + Allegro hand, add scene, return MjModel."""
    arm_spec  = mujoco.MjSpec.from_file(_ARM_XML)
    hand_spec = mujoco.MjSpec.from_file(_HAND_XML)

    # Attach Allegro to Panda's flange site
    for s in arm_spec.sites:
        if s.name == "attachment_site":
            arm_spec.attach(hand_spec, prefix="allegro/", site=s)
            break

    arm_spec.option.timestep = 0.002
    arm_spec.option.impratio = 10

    wb = arm_spec.worldbody

    light1 = wb.add_light()
    light1.name    = "main_light"
    light1.pos     = [0, 0, 1.5]
    light1.diffuse = [0.7, 0.7, 0.7]

    light2 = wb.add_light()
    light2.name       = "side_light"
    light2.pos        = [0.5, -0.5, 1.5]
    light2.diffuse    = [0.4, 0.4, 0.4]
    light2.castshadow = False

    floor      = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos  = [0, 0, -0.01]
    floor.size = [0, 0, 0.05]
    floor.rgba = [0.3, 0.35, 0.35, 1]

    # Placeholder object — will be replaced once task/scene is defined
    obj            = wb.add_body()
    obj.name       = "object"
    obj.pos        = [0.5, 0, 0.05]
    fj             = obj.add_freejoint()
    fj.name        = "object_joint"
    geom           = obj.add_geom()
    geom.name      = "object_geom"
    geom.type      = mujoco.mjtGeom.mjGEOM_BOX
    geom.size      = [0.025, 0.025, 0.025]
    geom.rgba      = [0.9, 0.3, 0.1, 1]
    geom.mass      = 0.05
    geom.condim    = 6
    geom.friction  = [1.2, 0.005, 0.0001]

    cam1            = wb.add_camera()
    cam1.name       = "fixed"
    cam1.pos        = [1.5, -1.0, 1.5]
    cam1.mode       = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam1.targetbody = "link4"

    cam2            = wb.add_camera()
    cam2.name       = "hand_close"
    cam2.pos        = [0.9, -0.6, 0.9]
    cam2.mode       = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam2.targetbody = "allegro/palm"

    cam3      = wb.add_camera()
    cam3.name = "top_down"
    cam3.pos  = [0.5, 0.0, 1.5]

    return arm_spec.compile()


class ArmHandEnv(gym.Env):
    """Franka Panda + Wonik Allegro — scene only, task TBD."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.model = _build_model()
        self.data  = mujoco.MjData(self.model)

        def jid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        self._arm_joint_ids  = np.array([jid(n) for n in _ARM_JOINT_NAMES])
        self._hand_joint_ids = np.array([jid(n) for n in _HAND_JOINT_NAMES])
        self._all_joint_ids  = np.concatenate([self._arm_joint_ids, self._hand_joint_ids])

        self._arm_qpos_addrs  = np.array([self.model.jnt_qposadr[j] for j in self._arm_joint_ids])
        self._arm_qvel_addrs  = np.array([self.model.jnt_dofadr[j]  for j in self._arm_joint_ids])
        self._hand_qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in self._hand_joint_ids])
        self._hand_qvel_addrs = np.array([self.model.jnt_dofadr[j]  for j in self._hand_joint_ids])

        self._act_min = self.model.actuator_ctrlrange[:, 0]
        self._act_max = self.model.actuator_ctrlrange[:, 1]

        self._jnt_lo = np.array([self.model.jnt_range[j, 0] for j in self._all_joint_ids])
        self._jnt_hi = np.array([self.model.jnt_range[j, 1] for j in self._all_joint_ids])

        obs_dim = _N_ARM + _N_ARM + _N_HAND + _N_HAND  # 46
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_DOF,), dtype=np.float32
        )

        self._renderer   = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set arm to home pose
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = _ARM_HOME[i]

        # Open hand
        hand_lo = self._jnt_lo[_N_ARM:]
        hand_hi = self._jnt_hi[_N_ARM:]
        for i, addr in enumerate(self._hand_qpos_addrs):
            self.data.qpos[addr] = np.clip(_HAND_OPEN[i], hand_lo[i], hand_hi[i])

        # Set actuators to match the initial pose
        init_angles = np.concatenate([_ARM_HOME, _HAND_OPEN])
        norm = 2.0 * (init_angles - self._jnt_lo) / (self._jnt_hi - self._jnt_lo + 1e-8) - 1.0
        self.data.ctrl[:] = 0.5 * (norm + 1.0) * (self._act_max - self._act_min) + self._act_min

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        ctrl = 0.5 * (action + 1.0) * (self._act_max - self._act_min) + self._act_min
        self.data.ctrl[:] = ctrl

        for _ in range(5):   # 5 × 2 ms = 10 ms per env step
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs      = self._get_obs()
        reward   = 0.0          # TODO: define when task is specified
        info     = {}
        truncated = self._step_count >= _MAX_STEPS

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="fixed")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self._arm_qpos_addrs].astype(np.float32),
            self.data.qvel[self._arm_qvel_addrs].astype(np.float32),
            self.data.qpos[self._hand_qpos_addrs].astype(np.float32),
            self.data.qvel[self._hand_qvel_addrs].astype(np.float32),
        ])


gym.register(
    id="ArmHand-v0",
    entry_point="arm_hand_env:ArmHandEnv",
    max_episode_steps=_MAX_STEPS,
)
