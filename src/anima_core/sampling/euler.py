# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""ComfyUI/k-diffusion의 Euler 갱신식과 RF ancestral 갱신식."""

import torch

from ..profiling import profile_range


@torch.no_grad()
def sample_euler(model_fn, x, sigmas, callback=None):
    """속도 예측을 사용해 기존 생성 경로와 같은 Euler 연산을 수행한다."""
    for i in range(len(sigmas) - 1):
        velocity = model_fn.velocity(x, sigmas[i])
        sigma = sigmas[i].to(device=x.device, dtype=x.dtype)
        sigma_next = sigmas[i + 1].to(device=x.device, dtype=x.dtype)
        with profile_range("sampling/euler_update"):
            x = x + velocity * (sigma_next - sigma)
        if callback is not None:
            callback(i)
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model_fn, x, sigmas, callback=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    """ComfyUI의 sample_euler_ancestral_RF를 Anima의 RF 좌표에 적용한다."""
    if noise_sampler is None:
        noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    dtype = x.dtype
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i].to(device=x.device, dtype=torch.float32)
        sigma_next = sigmas[i + 1].to(device=x.device, dtype=torch.float32)
        denoised = model_fn(x, sigma).float()
        with profile_range("sampling/euler_ancestral_update"):
            if sigma_next == 0:
                x = denoised.to(dtype)
            else:
                downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
                sigma_down = sigma_next * downstep_ratio
                alpha_next = 1 - sigma_next
                alpha_down = 1 - sigma_down
                ratio = sigma_down / sigma
                updated = ratio * x.float() + (1 - ratio) * denoised
                if eta > 0:
                    renoise_coeff = (
                        sigma_next.square()
                        - sigma_down.square() * alpha_next.square() / alpha_down.square()
                    ).clamp_min(0).sqrt()
                    updated = (alpha_next / alpha_down) * updated
                    if s_noise > 0:
                        updated = updated + noise_sampler(sigma, sigma_next).float() * s_noise * renoise_coeff
                x = updated.to(dtype)
        if callback is not None:
            callback(i)
    return x
