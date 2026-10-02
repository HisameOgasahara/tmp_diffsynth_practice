# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""이전 denoised 예측을 재사용하는 결정적 RES 다단계 갱신."""

import torch


@torch.no_grad()
def sample_res_multistep(model_fn, x, sigmas, callback=None):
    dtype = x.dtype
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    old_denoised = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model_fn(x, sigma).float()
        if sigma_next == 0:
            updated = denoised
        elif old_denoised is None:
            updated = x.float() + (x.float() - denoised) / sigma * (sigma_next - sigma)
        else:
            h = sigma.log() - sigma_next.log()
            c2 = (sigma.log() - sigmas[i - 1].log()) / h
            phi1 = (-h).expm1() / (-h)
            phi2 = (phi1 - 1) / (-h)
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            updated = (-h).exp() * x.float() + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
        x = updated.to(dtype)
        if callback is not None:
            callback(i)
    return x
