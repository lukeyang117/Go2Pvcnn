"""Manual viewport camera controls shared by the planner viewer and play script."""

from __future__ import annotations

import math


VIEWER_CAMERA_BINDINGS = {
    "PanGesture": "Ctrl RightButton",
    "TumbleGesture": "RightButton",
}


def configure_camera_bindings() -> bool:
    """Configure native gestures before the viewport camera is constructed."""
    try:
        from omni.kit.manipulator.camera import gestures
        from omni.kit.manipulator.camera import manipulator as camera_manipulator
    except ImportError:
        return False

    gestures.kDefaultKeyBindings = dict(VIEWER_CAMERA_BINDINGS)
    default_build_gestures = camera_manipulator.build_gestures

    def build_viewer_gestures(model, bindings=None, manager=None, configure_model=None):
        return default_build_gestures(
            model,
            dict(VIEWER_CAMERA_BINDINGS),
            manager=manager,
            configure_model=configure_model,
        )

    camera_manipulator.build_gestures = build_viewer_gestures
    return True


def camera_drag_pose(eye, target, dx, dy, *, pan, width, height):
    """Apply one pixel drag, preserving view direction for panning."""
    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def mul(a, scalar):
        return tuple(x * scalar for x in a)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def unit(value):
        return mul(value, 1.0 / math.sqrt(max(1.0e-12, dot(value, value))))

    def rotate(value, axis, angle):
        axis = unit(axis)
        cosine, sine = math.cos(angle), math.sin(angle)
        return add(add(mul(value, cosine), mul(cross(axis, value), sine)), mul(axis, dot(axis, value) * (1.0 - cosine)))

    offset = sub(eye, target)
    distance = math.sqrt(max(1.0e-12, dot(offset, offset)))
    forward = unit(sub(target, eye))
    right = cross(forward, (0.0, 0.0, 1.0))
    if dot(right, right) <= 1.0e-10:
        right = (1.0, 0.0, 0.0)
    right = unit(right)
    camera_up = unit(cross(right, forward))
    if pan:
        scale = distance * 1.5
        shift = add(
            mul(right, -dx / max(width, 1.0) * scale),
            mul(camera_up, dy / max(height, 1.0) * scale),
        )
        return add(eye, shift), add(target, shift)
    rotated = rotate(offset, (0.0, 0.0, 1.0), -dx / max(width, 1.0) * math.pi)
    rotated = rotate(rotated, right, dy / max(height, 1.0) * math.pi)
    return add(target, rotated), target


def _camera_pose_from_viewport(camera_path, viewport, *, fallback_target_distance=1.0):
    """Read a viewport camera, including cameras without an authored center of interest."""
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    state = ViewportCameraState(camera_path=camera_path, viewport=viewport)
    eye = tuple(float(value) for value in state.position_world)
    try:
        target = tuple(float(value) for value in state.target_world)
    except (AttributeError, RuntimeError, TypeError):
        from pxr import Usd

        transform = state.usd_camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        forward = transform.TransformDir((0.0, 0.0, -1.0))
        length = math.sqrt(sum(float(value) * float(value) for value in forward))
        if length <= 1.0e-12:
            raise RuntimeError(f"camera {camera_path} has no usable forward direction")
        distance = max(float(fallback_target_distance), 1.0e-3)
        target = tuple(
            eye[index] + float(forward[index]) / length * distance
            for index in range(3)
        )
    return eye, target


class CameraInputController:
    """Translate physical Kit input into mutually exclusive camera actions."""

    def __init__(self, set_camera_view, eye, target, viewport_size):
        import carb.input
        import omni.kit.app
        import omni.appwindow

        self._set_camera_view = set_camera_view
        self._eye, self._target = tuple(eye), tuple(target)
        self._width, self._height = (float(value) for value in viewport_size)
        self._right_down = False
        self._last_pos = None
        self._ctrl_down = False
        self._camera_path = self._active_camera_path()
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self._mouse = app_window.get_mouse()
        self._keyboard = app_window.get_keyboard()
        self._mouse_subscription = self._input.subscribe_to_mouse_events(self._mouse, self._on_mouse_event)
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        self._update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_app_update
        )
        self._native_camera_model = None
        self.update_count = 0
        self.pan_update_count = 0
        self.look_update_count = 0
        self._disable_native_rotation_gestures()

    def close(self):
        if self._mouse_subscription is not None:
            self._input.unsubscribe_to_mouse_events(self._mouse, self._mouse_subscription)
            self._mouse_subscription = None
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_subscription)
            self._keyboard_subscription = None
        if self._update_subscription is not None:
            self._update_subscription = None

    def _on_app_update(self, _event):
        self._sync_active_camera()
        self.poll()

    def _active_camera_path(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            return str(viewport.camera_path) if viewport is not None else None
        except (AttributeError, ImportError, ReferenceError):
            return None

    def _sync_active_camera(self):
        """Adopt a camera created/switched through Create From View."""
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            path = str(viewport.camera_path) if viewport is not None else None
            if not path or path == self._camera_path:
                return
            current_distance = math.sqrt(
                sum((self._eye[index] - self._target[index]) ** 2 for index in range(3))
            )
            self._eye, self._target = _camera_pose_from_viewport(
                path,
                viewport,
                fallback_target_distance=current_distance,
            )
            self._camera_path = path
            self._native_camera_model = None
        except (AttributeError, ImportError, ReferenceError, RuntimeError, TypeError):
            return

    def _on_keyboard_event(self, event):
        import carb.input

        if event.input not in (
            carb.input.KeyboardInput.LEFT_CONTROL,
            carb.input.KeyboardInput.RIGHT_CONTROL,
        ):
            return
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._ctrl_down = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._ctrl_down = False

    def _pan_modifier_is_down(self):
        import carb.input

        return self._ctrl_down or bool(
            self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT_CONTROL)
            or self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT_CONTROL)
        )

    def _disable_native_rotation_gestures(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport_window

            model = self._native_camera_model
            if model is None:
                window = get_active_viewport_window()
                layer = window._find_viewport_layer("Camera", "manipulator") if window is not None else None
                native = getattr(getattr(layer, "layer", None), "manipulator", None)
                model = getattr(native, "model", None)
                self._native_camera_model = model
            if model is not None:
                model.set_ints("disable_tumble", [1])
                model.set_ints("disable_look", [1])
                model.set_ints("disable_pan", [1])
        except (AttributeError, ImportError, ReferenceError):
            return

    def _apply_drag(self, current):
        self._sync_active_camera()
        if self._last_pos is None:
            self._last_pos = current
            return
        dx = float(current[0] - self._last_pos[0])
        dy = float(current[1] - self._last_pos[1])
        self._last_pos = current
        if abs(dx) <= 1.0e-6 and abs(dy) <= 1.0e-6:
            return
        pan = self._pan_modifier_is_down()
        self._eye, self._target = camera_drag_pose(
            self._eye, self._target, dx, dy, pan=pan, width=self._width, height=self._height
        )
        self.update_count += 1
        if pan:
            self.pan_update_count += 1
        else:
            self.look_update_count += 1
        try:
            self._set_camera_view(
                self._eye,
                self._target,
                camera_prim_path=self._camera_path,
            )
        except TypeError:
            self._set_camera_view(self._eye, self._target)

    def _on_mouse_event(self, event):
        import carb.input

        if event.type == carb.input.MouseEventType.RIGHT_BUTTON_DOWN:
            self._right_down = True
            self._last_pos = self._input.get_mouse_coords_pixel(self._mouse)
        elif event.type == carb.input.MouseEventType.RIGHT_BUTTON_UP:
            self._right_down = False
            self._last_pos = None
        elif event.type == carb.input.MouseEventType.MOVE and self._right_down:
            self._apply_drag(self._input.get_mouse_coords_pixel(self._mouse))

    def poll(self):
        import carb.input

        self._disable_native_rotation_gestures()
        pressed = self._input.get_mouse_value(self._mouse, carb.input.MouseInput.RIGHT_BUTTON) > 0
        if not pressed:
            self._right_down = False
            self._last_pos = None
            return
        current = self._input.get_mouse_coords_pixel(self._mouse)
        if not self._right_down:
            self._right_down = True
            self._last_pos = current
            return
        self._apply_drag(current)
