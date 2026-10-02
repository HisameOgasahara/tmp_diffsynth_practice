# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""ComfyUI/k-diffusion의 2차 Heun 갱신식."""

import torch

from ..profiling import profile_range


@torch.no_grad()
def sample_heun(model_fn, x, sigmas, callback=None):
    """Euler 예측 위치의 속도로 보정하고 마지막 0 구간은 Euler로 끝낸다."""
    for i in range(len(sigmas) - 1):
        velocity = model_fn.velocity(x, sigmas[i])
        sigma = sigmas[i].to(device=x.device, dtype=x.dtype)
        sigma_next = sigmas[i + 1].to(device=x.device, dtype=x.dtype)
        dt = sigma_next - sigma
        with profile_range("sampling/heun_predict"):
            predicted = x + velocity * dt
        if sigmas[i + 1] == 0:
            x = predicted
        else:
            velocity_next = model_fn.velocity(predicted, sigmas[i + 1])
            with profile_range("sampling/heun_correct"):
                x = x + (velocity + velocity_next) * 0.5 * dt
        if callback is not None:
            callback(i)
    return x
