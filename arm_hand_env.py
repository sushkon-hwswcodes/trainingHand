"""
arm_hand_env.py — Franka Panda arm + Wonik Allegro hand, in-hand cube reorientation.

Architecture:
  - Franka Panda (7 DOF arm) from MuJoCo Menagerie
  - Wonik Allegro right hand (16 DOF) attached to Panda flange via MjSpec.attach()
  - Total: 23 DOF, 23 actuators
  - Arm pre-grasp pose: palm faces downward over a cube on the floor

Observation space (57-dim):
  arm joint positions      (7)
  arm joint velocities     (7)
  hand joint positions     (16)
  hand joint velocities    (16)
  cube position rel palm   (3)
  cube orientation quat    (4)
  target orientation quat  (4)

Action space (23-dim, normalized [-1,1]):
  first 7  → arm joints (joint1-7)
  last 16  → hand joints (allegro/ffj0-3, mfj0-3, rfj0-3, thj0-3)

Reward (same structure as hand_env.py):
  orientation_reward : 1 - orient_err/π  ∈ [0, 1]
  contact_bonus      : up to 0.1 for fingers touching cube
  drop_penalty       : -10 if cube falls below floor level
  success_bonus      : +5 when orientation error < 0.2 rad (~11°)
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_DIR = os.path.dirname(os.path.abspath(__file__))
_ARM_XML  = os.path.join(_DIR, "franka_panda", "panda_nohand.xml")
_HAND_XML = os.path.join(_DIR, "allegro_hand",  "right_hand.xml")

# Joint name lists
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

_N_ARM   = len(_ARM_JOINT_NAMES)   # 7
_N_HAND  = len(_HAND_JOINT_NAMES)  # 16
_N_DOF   = _N_ARM + _N_HAND        # 23

# Touch sensors created in _build_model(), prefixed after attach
_TOUCH_NAMES = [
    "allegro/touch_ff_tip", "allegro/touch_mf_tip",
    "allegro/touch_rf_tip", "allegro/touch_th_tip",
]

# Arm pre-grasp pose — palm at ≈[0.505, 0, 0.383], normal ≈[0,0,-1]
_ARM_GRASP_POSE = np.array([0.0, -0.1, 0.0, -2.167, 0.0, 2.0, 0.785])

# Hand open pose for reset (same structure as hand_env.py)
_HAND_OPEN_POSE = np.array([
    0.0, 0.0, 0.0, 0.0,   # index  — fully open
    0.0, 0.0, 0.0, 0.0,   # middle — fully open
    0.0, 0.0, 0.0, 0.0,   # ring   — fully open
    0.3, 0.4, 0.0, 0.0,   # thumb  — rotated out, not flexed
])

_CUBE_DROP_Z    = 0.05   # episode ends if cube falls below this z
_SUCCESS_THRESH = 0.2    # radians (~11°)
_MAX_STEPS      = 500


def _build_model() -> mujoco.MjModel:
    """Compose Franka arm + Allegro hand, add scene elements, return MjModel."""
    arm_spec  = mujoco.MjSpec.from_file(_ARM_XML)
    hand_spec = mujoco.MjSpec.from_file(_HAND_XML)

    # Add touch sensors to hand_spec before attaching (they get prefixed)
    for site_name in ["ff_tip", "mf_tip", "rf_tip", "th_tip"]:
        touch = hand_spec.add_sensor()
        touch.name    = f"touch_{site_name}"
        touch.type    = mujoco.mjtSensor.mjSENS_TOUCH
        touch.objtype = mujoco.mjtObj.mjOBJ_SITE
        touch.objname = site_name

    # Attach Allegro to Panda's flange site
    for s in arm_spec.sites:
        if s.name == "attachment_site":
            arm_spec.attach(hand_spec, prefix="allegro/", site=s)
            break

    # Physics options
    arm_spec.option.timestep = 0.002
    arm_spec.option.impratio = 10

    # Scene elements added to worldbody
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

    floor = wb.add_geom()
    floor.name  = "floor"
    floor.type  = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos   = [0, 0, -0.01]
    floor.size  = [0, 0, 0.05]
    floor.rgba  = [0.3, 0.35, 0.35, 1]

    # Cube — freejoint body must be a direct worldbody child
    cube_body      = wb.add_body()
    cube_body.name = "cube"
    cube_body.pos  = [0.5, 0, 0.35]    # initial pose; overwritten in reset()
    cube_fj        = cube_body.add_freejoint()
    cube_fj.name   = "cube_joint"
    cube_geom          = cube_body.add_geom()
    cube_geom.name     = "cube_geom"
    cube_geom.type     = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom.size     = [0.025, 0.025, 0.025]
    cube_geom.rgba     = [0.9, 0.3, 0.1, 1]
    cube_geom.mass     = 0.05
    cube_geom.condim   = 6
    cube_geom.friction = [1.2, 0.005, 0.0001]
    cube_geom.priority = 1
    cube_site      = cube_body.add_site()
    cube_site.name = "cube_center"
    cube_site.size = [0.004, 0.004, 0.004]

    # Cameras
    cam1             = wb.add_camera()
    cam1.name        = "fixed"
    cam1.pos         = [1.5, -1.0, 1.5]
    cam1.mode        = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam1.targetbody  = "link4"       # tracks mid-arm → whole system in view

    cam2             = wb.add_camera()
    cam2.name        = "hand_close"
    cam2.pos         = [0.9, -0.6, 0.9]
    cam2.mode        = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam2.targetbody  = "allegro/palm"  # close-up on the hand

    cam3             = wb.add_camera()
    cam3.name        = "top_down"
    cam3.pos         = [0.5, 0.0, 1.5]

    return arm_spec.compile()


class ArmHandCubeEnv(gym.Env):
    """Franka Panda + Wonik Allegro: in-hand cube reorientation."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        target_quat: np.ndarray | None = None,
        randomize_target: bool = True,
    ):
        super().__init__()
        self.render_mode      = render_mode
        self.randomize_target = randomize_target

        self.model = _build_model()
        self.data  = mujoco.MjData(self.model)

        # Cache IDs
        def jid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        def bid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        def sid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        self._arm_joint_ids  = np.array([jid(n) for n in _ARM_JOINT_NAMES])
        self._hand_joint_ids = np.array([jid(n) for n in _HAND_JOINT_NAMES])
        self._all_joint_ids  = np.concatenate([self._arm_joint_ids, self._hand_joint_ids])

        self._arm_qpos_addrs  = np.array([self.model.jnt_qposadr[j] for j in self._arm_joint_ids])
        self._arm_qvel_addrs  = np.array([self.model.jnt_dofadr[j]  for j in self._arm_joint_ids])
        self._hand_qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in self._hand_joint_ids])
        self._hand_qvel_addrs = np.array([self.model.jnt_dofadr[j]  for j in self._hand_joint_ids])

        self._cube_jid      = jid("cube_joint")
        self._cube_qposadr  = self.model.jnt_qposadr[self._cube_jid]
        self._cube_body_id  = bid("cube")
        self._palm_body_id  = bid("allegro/palm")

        self._touch_ids = np.array([sid(n) for n in _TOUCH_NAMES])

        # Actuator ctrl range
        self._act_min = self.model.actuator_ctrlrange[:, 0]
        self._act_max = self.model.actuator_ctrlrange[:, 1]

        # Joint ranges for all 23 joints (in joint order)
        self._jnt_lo = np.array([self.model.jnt_range[j, 0] for j in self._all_joint_ids])
        self._jnt_hi = np.array([self.model.jnt_range[j, 1] for j in self._all_joint_ids])
        self._hand_jnt_lo = self._jnt_lo[_N_ARM:]
        self._hand_jnt_hi = self._jnt_hi[_N_ARM:]

        self._target_quat = (
            target_quat if target_quat is not None else np.array([1., 0., 0., 0.])
        )

        # Spaces
        obs_dim = _N_ARM + _N_ARM + _N_HAND + _N_HAND + 3 + 4 + 4  # 57
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_DOF,), dtype=np.float32
        )

        self._renderer  = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize_target:
            self._target_quat = self._random_quat()

        # ── Step 1: set arm to pre-grasp pose ───────────────────────────
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = _ARM_GRASP_POSE[i]

        # ── Step 2: open hand ───────────────────────────────────────────
        for i, addr in enumerate(self._hand_qpos_addrs):
            self.data.qpos[addr] = np.clip(
                _HAND_OPEN_POSE[i],
                self._hand_jnt_lo[i],
                self._hand_jnt_hi[i],
            )

        mujoco.mj_forward(self.model, self.data)

        # ── Step 3: place cube just below palm center ───────────────────
        palm_pos = self.data.xpos[self._palm_body_id].copy()
        # Palm normal (z-col of xmat) points downward; cube goes below palm
        palm_zaxis = self.data.xmat[self._palm_body_id].reshape(3, 3)[:, 2]
        # Cube at palm center offset 0.06m along palm normal (below palm)
        cube_pos = palm_pos + palm_zaxis * 0.06 + np.array([
            self.np_random.uniform(-0.01, 0.01),
            self.np_random.uniform(-0.01, 0.01),
            0,
        ])
        qs = self._cube_qposadr
        self.data.qpos[qs:qs + 3] = cube_pos
        self.data.qpos[qs + 3:qs + 7] = self._random_quat()

        # ── Step 4: build ctrl for open arm+hand pose ───────────────────
        open_angles = np.concatenate([_ARM_GRASP_POSE, _HAND_OPEN_POSE])
        norm = 2.0 * (open_angles - self._jnt_lo) / (self._jnt_hi - self._jnt_lo + 1e-8) - 1.0
        ctrl = 0.5 * (norm + 1.0) * (self._act_max - self._act_min) + self._act_min
        self.data.ctrl[:] = ctrl

        # ── Step 5: settle physics ──────────────────────────────────────
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Scale normalized action → actuator ctrl range
        ctrl = 0.5 * (action + 1.0) * (self._act_max - self._act_min) + self._act_min
        self.data.ctrl[:] = ctrl

        # 5 substeps × 2 ms = 10 ms per env step
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = info.get("dropped", False) or info.get("success", False)
        truncated  = self._step_count >= _MAX_STEPS

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="fixed")
        frame = self._renderer.render()
        if self.render_mode == "human":
            import cv2
            cv2.imshow("ArmHandCubeEnv", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        arm_qpos  = self.data.qpos[self._arm_qpos_addrs]
        arm_qvel  = self.data.qvel[self._arm_qvel_addrs]
        hand_qpos = self.data.qpos[self._hand_qpos_addrs]
        hand_qvel = self.data.qvel[self._hand_qvel_addrs]

        cube_pos_world = self.data.xpos[self._cube_body_id].copy()
        palm_pos_world = self.data.xpos[self._palm_body_id].copy()
        cube_pos_rel   = cube_pos_world - palm_pos_world

        qs        = self._cube_qposadr
        cube_quat = self.data.qpos[qs + 3:qs + 7].copy()

        return np.concatenate([
            arm_qpos.astype(np.float32),
            arm_qvel.astype(np.float32),
            hand_qpos.astype(np.float32),
            hand_qvel.astype(np.float32),
            cube_pos_rel.astype(np.float32),
            cube_quat.astype(np.float32),
            self._target_quat.astype(np.float32),
        ])

    def _compute_reward(self) -> tuple[float, dict]:
        info = {}

        cube_z = self.data.xpos[self._cube_body_id, 2]
        if cube_z < _CUBE_DROP_Z:
            info["dropped"] = True
            return -10.0, info
        info["dropped"] = False

        qs        = self._cube_qposadr
        cube_quat = self.data.qpos[qs + 3:qs + 7]
        orient_err = self._quat_error(cube_quat, self._target_quat)
        orient_reward = 1.0 - orient_err / np.pi

        touches   = np.array([self.data.sensordata[i] for i in self._touch_ids])
        n_contacts = np.sum(touches > 0.01)
        contact_bonus = 0.1 * min(n_contacts, 4) / 4.0

        if orient_err < _SUCCESS_THRESH:
            info["success"]  = True
            success_bonus = 5.0
        else:
            info["success"]  = False
            success_bonus = 0.0

        reward = orient_reward + contact_bonus + success_bonus
        info["orient_err_deg"] = np.degrees(orient_err)
        info["n_contacts"]     = int(n_contacts)
        return float(reward), info

    @staticmethod
    def _quat_error(q1: np.ndarray, q2: np.ndarray) -> float:
        q1  = q1 / (np.linalg.norm(q1) + 1e-8)
        q2  = q2 / (np.linalg.norm(q2) + 1e-8)
        dot = np.abs(np.dot(q1, q2))
        return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))

    @staticmethod
    def _random_quat() -> np.ndarray:
        u = np.random.uniform(0, 1, 3)
        q = np.array([
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
        ])
        return q / np.linalg.norm(q)


# Register with gymnasium
gym.register(
    id="ArmHandCube-v0",
    entry_point="arm_hand_env:ArmHandCubeEnv",
    max_episode_steps=_MAX_STEPS,
)
