# Parallelism Policy Play Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing play entrypoint so a trained `parallelism_tracking_flat` policy can run in real IsaacLab physics while displaying its 24-frame Parallelism reference, trajectory markers, app-panel commands, and per-termination reset suppression.

**Architecture:** `scripts/play.py` remains the only runtime entrypoint. It selects the tracking PLAY config, obtains the manager already used by tracking observations, and adds play-only utilities for an ImGui command panel, non-colliding reference Go2 articulation, marker updates, and a pre-reset termination filter. The training environment, planner, PPO configuration, and raw projects remain untouched.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab ManagerBasedRLEnv, Isaac Sim Omni UI/Usd APIs, RSL-RL.

## Global Constraints

- Work on branch `Parallelism-flat-rl`; do not modify `raw/`.
- Use the same body-frame `base_velocity` tensor for panel input, policy observations, and Parallelism replanning.
- Planner reuses the existing `ParallelismReferenceManager`: reset and 24-frame replan semantics must not change.
- Reference Go2 is visual-only, fully non-colliding, exactly overlaps the physical policy robot, and has a translucent cyan material.
- Every existing configured termination remains diagnostically computed; a checked `不终止` UI item suppresses only its reset effect.
- Suppression must occur before IsaacLab consumes `termination_manager` results to reset environments.
- Default play mode is one environment and the panel starts with all `不终止` items checked.

---

## File Structure

- Modify: `Go2Pvcnn/scripts/play.py`
  - Parser/task selection; panel state; command override; reference robot and marker lifecycle; pre-reset termination filter; runtime loop integration.
- Modify: `Go2Pvcnn/tracking/register_envs.py`
  - Register a play task only if gym construction cannot accept the explicit PLAY cfg with the existing task id.
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`
  - Verify the play entrypoint exposes the new experiment.
- Create: `Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py`
  - Pure-Python coverage for command clamping, diagnostic termination mask, and manager-frame visualization data extraction.

## Task 1: Add pure runtime state and tests

**Files:**
- Create: `Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py`
- Modify: `Go2Pvcnn/scripts/play.py`

**Interfaces:**
- Produces `ParallelismPlayPanelState` with `vx`, `vy`, `vyaw`, and `suppress_termination: dict[str, bool]`.
- Produces `_panel_command_tensor(state, command)` returning a tensor with the command shape/dtype/device.
- Produces `_filter_termination_masks(raw_masks, suppress)` returning `(effective_done, raw_masks)`.

- [x] **Step 1: Write failing tests for panel command and termination filtering**

```python
def test_panel_command_has_signed_parallelism_limits():
    state = play.ParallelismPlayPanelState(vx=3.0, vy=-2.0, vyaw=-4.0)
    command = torch.zeros(1, 3)
    assert torch.equal(play._panel_command_tensor(state, command), torch.tensor([[1.0, -0.5, -1.0]]))


def test_suppressed_termination_stays_diagnostic_but_not_done():
    raw = {"base_contact": torch.tensor([True]), "parallelism_ref_joint_pos_too_far": torch.tensor([True])}
    done, diagnostics = play._filter_termination_masks(raw, {"base_contact": True, "parallelism_ref_joint_pos_too_far": False})
    assert diagnostics["base_contact"].item() is True
    assert done.item() is True
```

- [x] **Step 2: Run the tests to verify failure**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: FAIL because the play runtime interfaces do not exist.

- [x] **Step 3: Implement the pure helpers in `scripts/play.py`**

```python
@dataclass
class ParallelismPlayPanelState:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    suppress_termination: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(PARALLELISM_TERMINATION_NAMES, True))


def _panel_command_tensor(state, command):
    values = command.new_tensor([[state.vx, state.vy, state.vyaw]])
    return torch.clamp(values, min=command.new_tensor([-1.0, -0.5, -1.0]), max=command.new_tensor([1.0, 0.5, 1.0]))
```

Implement `_filter_termination_masks` with a Torch logical OR over only non-suppressed raw masks. Do not call `.item()` in the filter.

- [x] **Step 4: Run focused tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: PASS.

- [x] **Step 5: Commit the first testable utility**

```bash
git add Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py
git commit -m "feat: add parallelism play panel runtime state"
```

## Task 2: Select the tracking PLAY environment and share panel command

**Files:**
- Modify: `Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py`

**Interfaces:**
- Consumes `ParallelismTrackingFlatEnvCfg_PLAY` and `get_parallelism_reference_manager(base_env)`.
- Produces `--experiment parallelism_tracking_flat` and a manager-backed play loop.

- [x] **Step 1: Write failing parser/map tests**

```python
def test_play_parser_accepts_parallelism_tracking_flat():
    parser = play.build_arg_parser()
    experiment = next(action for action in parser._actions if action.dest == "experiment")
    assert "parallelism_tracking_flat" in experiment.choices


def test_parallelism_play_command_updates_env0_only():
    command = torch.zeros(2, 3)
    play._apply_panel_velocity_command(fake_env(command), ParallelismPlayPanelState(vx=0.4))
    assert torch.equal(command, torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]))
```

- [x] **Step 2: Run the parser/map tests to verify failure**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: FAIL because `play.py` has no tracking experiment mapping or panel command application.

- [x] **Step 3: Add the tracking branch to play setup**

Import `ParallelismTrackingFlatEnvCfg_PLAY`, map `parallelism_tracking_flat` to `Isaac-Go2-Parallelism-Tracking-Flat-v0`, and create that cfg with `num_envs=1` by default. After `gym.make`, obtain the manager with:

```python
from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager
parallelism_manager = get_parallelism_reference_manager(base_env)
```

Add `_apply_panel_velocity_command(base_env, state)` that updates only `command_manager.get_command("base_velocity")[0, :3]`. Call it before `wrapped_env.get_observations()` and before `wrapped_env.step(actions)`.

- [x] **Step 4: Run focused tests**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: PASS.

- [x] **Step 5: Commit task**

```bash
git add Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/tracking
git commit -m "feat: play parallelism tracking policy"
```

## Task 3: Add reference Go2 and manager-backed planning markers

**Files:**
- Modify: `Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py`

**Interfaces:**
- Consumes manager `root_pos_w`, `root_rpy_w`, `joint_pos`, `foot_pos_w`, `contact_state`, and `phase`.
- Produces `_ParallelismPlayVisualizer.update(base_env, manager)` and a reference articulation under `/World/envs/env_0/ParallelismReferenceGo2`.

- [x] **Step 1: Write failing extraction tests**

```python
def test_reference_visual_frame_uses_current_manager_phase():
    manager = fake_manager(phase=3)
    frame = play._parallelism_visual_frame(manager, env_id=0)
    assert torch.equal(frame.joint_pos, manager.joint_pos[0, 3])
    assert torch.equal(frame.foot_pos_w, manager.foot_pos_w[0, 3])
    assert frame.future_foot_pos_w.shape[0] == manager.horizon - 3
```

- [x] **Step 2: Run the test to verify failure**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py::test_reference_visual_frame_uses_current_manager_phase -q`

Expected: FAIL because no visual frame extraction exists.

- [x] **Step 3: Implement reference visualization**

Define a small `ParallelismVisualFrame` dataclass and `_parallelism_visual_frame`. Use IsaacLab marker APIs to create root trajectory, four foot trajectory marker sets, touchdown markers, and four actual-to-reference foot error-line marker sets. Spawn a second copy of the robot USD at the fixed reference prim path, set collision disabled on every reference collider, set a translucent cyan PreviewSurface material, and update root/joint state every loop. The reference robot must not be inserted into `base_env.scene`, action manager, or sensors.

- [x] **Step 4: Run the focused test**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: PASS.

- [x] **Step 5: Commit task**

```bash
git add Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py
git commit -m "feat: visualize parallelism policy reference"
```

## Task 4: Add ImGui controls and pre-reset per-termination masking

**Files:**
- Modify: `Go2Pvcnn/scripts/play.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py`

**Interfaces:**
- Consumes `base_env.termination_manager` raw per-term masks and `ParallelismPlayPanelState.suppress_termination`.
- Produces an ImGui window with signed speed sliders, `全部不终止`, seven individual `不终止` checkboxes, and live raw-trigger indicators.

- [x] **Step 1: Write failing tests for all-term suppression and selective reset**

```python
def test_all_suppressed_terminations_produce_no_done():
    raw = {"time_out": torch.tensor([True]), "bad_orientation": torch.tensor([True])}
    done, _ = play._filter_termination_masks(raw, {"time_out": True, "bad_orientation": True})
    assert done.tolist() == [False]


def test_unsuppressed_single_termination_produces_done():
    raw = {"time_out": torch.tensor([True]), "bad_orientation": torch.tensor([True])}
    done, _ = play._filter_termination_masks(raw, {"time_out": True, "bad_orientation": False})
    assert done.tolist() == [True]
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py -q`

Expected: FAIL until all named masks are handled by the runtime filter.

- [x] **Step 3: Implement pre-reset filter and panel**

Wrap the tracking play environment's termination-manager compute path before constructing the RSL-RL wrapper. The wrapper records raw `term_dones` by configured term name, creates the effective Torch OR from unsuppressed terms, and writes the effective values into the manager fields that ManagerBasedRLEnv uses for reset. The panel reads the raw masks after each step for its indicators. It must not mutate cfg termination definitions or replace the training environment's manager.

Use Omni UI `FloatSlider`/`CheckBox` controls. Reconcile `全部不终止` from the seven individual booleans each UI refresh. Update the panel on the app thread only.

- [x] **Step 4: Run focused test suite**

Run: `pytest Go2Pvcnn/tests/tracking/test_parallelism_policy_play_visualization.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q`

Expected: PASS.

- [x] **Step 5: Commit task**

```bash
git add Go2Pvcnn/scripts/play.py Go2Pvcnn/tests/tracking
git commit -m "feat: control parallelism play terminations"
```

## Task 5: Verify the actual IsaacLab playback path

**Files:**
- Modify: `Go2Pvcnn/tests/tracking/parallelism_training_smoke_probe.py` only if a reusable `--play` probe can stay headless.

- [x] **Step 1: Run all tracking and planner unit tests**

Run: `pytest Go2Pvcnn/tests/tracking Go2Pvcnn/tests/parallelism/test_planner.py Go2Pvcnn/tests/parallelism/test_root_candidates_kinematics.py Go2Pvcnn/tests/parallelism/test_swing.py -q`

Expected: PASS.

- [x] **Step 2: Run a headless checkpoint smoke**

Run: `CUDA_VISIBLE_DEVICES=0 /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py --experiment parallelism_tracking_flat --run_dir 2026-07-31_19-48-00 --checkpoint model_16100.pt --num_envs 1 --headless --max-steps 40 --device cuda:0`

Expected: app starts, checkpoint loads, manager replans, and no reference-visualization import/runtime exception occurs.

- [ ] **Step 3: Run non-headless visual acceptance**

Run: launch the same command without `--headless` and with the project IsaacSim environment variables. Set nonzero panel velocities; verify the command changes policy and future planner trajectory. Trigger each relevant raw termination with its `不终止` item checked/un-checked; verify the raw indicator remains visible and reset behavior changes only for that item.

- [x] **Step 4: Commit final verification-related changes**

```bash
git add Go2Pvcnn/tests/tracking/parallelism_training_smoke_probe.py
git commit -m "test: cover parallelism policy play visualization"
```
