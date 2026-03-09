# Project Status

**Goal**: Train a dexterous robotic arm to pick up objects from one table and place them on another,
using a Franka Panda arm + Wonik Allegro hand in MuJoCo simulation.

---

## High-Level Plan

| # | Phase | Description | Status |
|---|-------|-------------|--------|
| 1 | **Simulation setup** | Get a realistic arm + hand model running in MuJoCo | ✅ Done |
| 2 | **Scene design** | Add tables of varying heights around the arm | ✅ Done |
| 3 | **Task definition** | Define pick-and-place task, reward function, reset logic | 🔲 Next |
| 4 | **RL training** | Train PPO policy, tune hyperparameters | 🔲 Pending |
| 5 | **Evaluation** | Benchmark success rate, generalization across table heights | 🔲 Pending |
| 6 | **Sim-to-real prep** | Domain randomization, policy distillation (stretch goal) | 🔲 Pending |

---

## Session Log

### Session 1–2 — Simulation Setup ✅
- Researched SOTA in robotic hand manipulation (MuJoCo Menagerie, VLA models, sim-to-real)
- Replaced hand-crafted capsule model with real **Wonik Allegro** hand from MuJoCo Menagerie
- Built `HandCube-v0`: Allegro-only in-hand cube reorientation (16 DOF)
- Fixed cube physics (drop threshold, settle sequence, finger opening)

### Session 3 — Arm + Hand Composition ✅
- Composed **Franka Panda** (7 DOF) + **Allegro Hand** (16 DOF) = **23 DOF total**
- Used MuJoCo 3.5 `MjSpec.attach()` API for clean programmatic composition
- Built `ArmHandCube-v0` Gymnasium environment ([arm_hand_env.py](arm_hand_env.py))
  - 57-dim observation, 23-dim action, orientation-matching reward
- Cleaned up codebase, removed all prototyping scaffolding
- Created GitHub repo: [sushkon-hwswcodes/trainingHand](https://github.com/sushkon-hwswcodes/trainingHand)

---

## Current State of the Code

```
trainingHand/
├── arm_hand_env.py     ← model + bare env (ArmHand-v0); reward=0, task TBD
├── example.py          ← sanity check: loads env, steps 100 times, optionally records
├── franka_panda/       ← Panda arm MJCF + meshes (from MuJoCo Menagerie)
└── allegro_hand/       ← Allegro hand MJCF + meshes (from MuJoCo Menagerie)
```

All task-specific code removed (orientation reward, drop penalty, touch sensors, target quaternion, success threshold).
Observation is now just joint positions + velocities (46-dim). Reward returns 0.0 as a stub.
`train.py` and `task.py` will be added once the scene and task are defined (Phases 2 + 3).

---

## Session 4 — Scene Design ✅
- Added 8 tables at 45° intervals around the arm base, heights 0.22–0.56 m, all within reach
- Verified workspace envelope: radial reach ~0.90 m, z range −0.34 to +1.22 m
- `render_workspace.py` renders a 21 s video sweeping all 7 arm joints through full range

## Next Session — Task Definition (Phase 3)

**Decisions needed:**
- [ ] Pick which table is the **source** (object starts here) and which is the **target** (goal)
- [ ] Fixed source/target pair, or randomized each episode?
- [ ] What counts as "placed": object center within X cm of target table surface?
- [ ] Reward shaping strategy (dense vs sparse)

**To implement once decisions are made:**
- [ ] Add touch sensors back to `_build_model()` (needed to detect grasp)
- [ ] Add a target-zone marker (semi-transparent box) on the target table
- [ ] Update `reset()`: spawn object on source table surface with small random offset
- [ ] Write `_compute_reward()` for pick-and-place:
  - Phase 1: approach — reward ∝ 1/(1 + dist(palm, object))
  - Phase 2: grasp — bonus when ≥2 fingertips contact object
  - Phase 3: lift — reward when object z > source table height + 0.05 m
  - Phase 4: transport — reward ∝ 1/(1 + dist(object, target_zone))
  - Phase 5: place — success bonus when object rests on target table
- [ ] Add `train.py` once reward is defined

---

## Key Technical Notes

- **Model composition**: `MjSpec.attach(hand_spec, prefix="allegro/", site=attachment_site)` — mesh paths resolve relative to each source file automatically
- **Arm pre-grasp pose**: `[0, -0.1, 0, -2.167, 0, 2.0, 0.785]` → palm at ~[0.505, 0, 0.383], facing down
- **Joint naming after composition**: arm joints = `joint1`–`joint7`; hand joints = `allegro/ffj0`–`allegro/thj3`
- **Touch sensors**: add to `hand_spec` before `attach()` call, then prefixed → `allegro/touch_ff_tip` etc. (removed for now, add back in Phase 3)
- **Tables**: 8 static bodies added in `_build_model()`, each with a top geom + leg geom; heights 0.22–0.56 m at r ≈ 0.38–0.55 m
- **Rendering**: EGL headless (`MUJOCO_GL=egl`), cameras use `mjCAMLIGHT_TARGETBODYCOM` mode
