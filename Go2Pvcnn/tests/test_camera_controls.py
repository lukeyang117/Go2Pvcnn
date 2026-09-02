from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_ctrl_right_drag_is_translation_and_plain_right_drag_is_rotation():
    from extension.viz.camera_controls import camera_drag_pose

    eye = (0.0, -3.0, 2.0)
    target = (0.0, 0.0, 0.5)
    look_eye, look_target = camera_drag_pose(
        eye, target, 80.0, 0.0, pan=False, width=800.0, height=500.0
    )
    pan_eye, pan_target = camera_drag_pose(
        eye, target, 80.0, 0.0, pan=True, width=800.0, height=500.0
    )

    assert look_target == target
    assert look_eye != eye
    assert pan_target != target
    torch.testing.assert_close(
        torch.tensor(pan_eye) - torch.tensor(pan_target),
        torch.tensor(eye) - torch.tensor(target),
    )


def test_viz_and_play_install_manual_camera_controls_without_follow_reset():
    viz = (ROOT / "extension/viz/go2_foostep_planner.py").read_text()
    play = (ROOT / "scripts/play.py").read_text()

    assert "configure_camera_bindings" in viz
    assert "CameraInputController" in viz
    assert "configure_camera_bindings" in play
    assert "CameraInputController" in play
    assert "extension.viz.camera_controls import" in viz
    assert "extension.viz.camera_controls import" in play


def test_camera_bindings_use_ctrl_right_for_pan():
    source = (ROOT / "extension/viz/camera_controls.py").read_text()
    assert '"PanGesture": "Ctrl RightButton"' in source
    assert '"TumbleGesture": "RightButton"' in source


def test_ctrl_right_drag_is_supported_as_a_pan_alias():
    source = (ROOT / "extension/viz/camera_controls.py").read_text()
    assert "KeyboardInput.LEFT_CONTROL" in source
    assert "KeyboardInput.RIGHT_CONTROL" in source
    assert "_pan_modifier_is_down" in source
    assert "KeyboardInput.LEFT_ALT" not in source
    assert "KeyboardInput.RIGHT_ALT" not in source


def test_camera_controller_is_driven_by_kit_updates_not_policy_loop_only():
    source = (ROOT / "extension/viz/camera_controls.py").read_text()
    assert "get_update_event_stream" in source
    assert "create_subscription_to_pop" in source
    assert "_on_app_update" in source


def test_camera_controller_follows_create_from_view_active_camera():
    source = (ROOT / "extension/viz/camera_controls.py").read_text()
    assert "camera_path" in source
    assert "ViewportCameraState" in source
    assert "camera_prim_path" in source
    assert "_sync_active_camera" in source
