from __future__ import annotations

import torch
from torch import Tensor


def swing_curve(start_w: Tensor, touchdown_w: Tensor, *, frames: int, height_m: float) -> Tensor:
    start = torch.as_tensor(start_w)
    touchdown = torch.as_tensor(touchdown_w, dtype=start.dtype, device=start.device)
    tau = torch.linspace(0.0, 1.0, int(frames), dtype=start.dtype, device=start.device)
    tau_view = tau.view(*((1,) * (start.ndim - 1)), int(frames), 1)
    curve = (1.0 - tau_view) * start[..., None, :] + tau_view * touchdown[..., None, :]
    curve = curve.clone()
    curve[..., 2] = curve[..., 2] + float(height_m) * 4.0 * tau * (1.0 - tau)
    return curve
