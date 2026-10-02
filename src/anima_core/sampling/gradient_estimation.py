# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""이전 속도와의 차이로 Euler 갱신을 보정한다."""

import torch


@torch.no_grad()
def sample_gradient_estimation(model_fn, x, sigmas, callback=None):
    dtype = x.dtype
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    old_derivative = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model_fn(x, sigma).float()
        derivative = (x.float() - denoised) / sigma
        if sigma_next == 0:
            updated = denoised
        else:
            dt = sigma_next - sigma
            updated = x.float() + derivative * dt
            if old_derivative is not None:
                # ComfyUI 기본 ge_gamma=2: (ge_gamma - 1) = 1.
                updated = updated + (derivative - old_derivative) * dt
        old_derivative = derivative
        x = updated.to(dtype)
        if callback is not None:
            callback(i)
    return x
