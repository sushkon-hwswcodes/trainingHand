# Project Status

**Goal**: Train a dexterous robotic arm to pick up objects from one table and place them on another,
using a Franka Panda arm + Wonik Allegro hand in MuJoCo simulation.

---

## High-Level Plan

| # | Phase | Description | Status |
|---|-------|-------------|--------|
| 1 | **Simulation setup** | Get a realistic arm + hand model running in MuJoCo | ✅ Done |
| 2 | **Scene design** | Add tables of varying heights around the arm | 🔲 Next |
| 3 | **Task definition** | Define pick-and-place task, reward function, reset logic | 🔲 Pending |
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
├── arm_hand_env.py     ← main environment (ArmHandCube-v0), needs task redesign for pick-and-place
├── train.py            ← PPO training script, ready to use once env task is updated
├── example.py          ← quick demo / sanity check
├── franka_panda/       ← Panda arm MJCF + meshes (from MuJoCo Menagerie)
└── allegro_hand/       ← Allegro hand MJCF + meshes (from MuJoCo Menagerie)
```

The environment currently runs a **cube reorientation** task (placeholder).
This will be replaced with **pick-and-place** in the next session.

---

## Next Session — Scene Design (Phase 2)

**To do:**
- [ ] Add 2–3 tables of different heights around the arm in `_build_model()`
- [ ] Decide table layout (positions, heights) relative to the arm base
- [ ] Place a cube (or multiple objects) on one table as the source
- [ ] Define a target zone on another table as the goal
- [ ] Update `reset()` to spawn objects on source table
- [ ] Rewrite reward function for pick-and-place:
  - Phase 1: approach object (distance arm → object)
  - Phase 2: grasp (fingertip contact)
  - Phase 3: lift and transport (object height + object → target distance)
  - Phase 4: place (object within target zone, arm releases)

---

## Key Technical Notes

- **Model composition**: `MjSpec.attach(hand_spec, prefix="allegro/", site=attachment_site)` — mesh paths resolve relative to each source file automatically
- **Arm pre-grasp pose**: `[0, -0.1, 0, -2.167, 0, 2.0, 0.785]` → palm at ~[0.505, 0, 0.383], facing down
- **Joint naming after composition**: arm joints = `joint1`–`joint7`; hand joints = `allegro/ffj0`–`allegro/thj3`
- **Touch sensors**: added to `hand_spec` before `attach()` call, then prefixed → `allegro/touch_ff_tip` etc.
- **Rendering**: EGL headless (`MUJOCO_GL=egl`), cameras use `mjCAMLIGHT_TARGETBODYCOM` mode
