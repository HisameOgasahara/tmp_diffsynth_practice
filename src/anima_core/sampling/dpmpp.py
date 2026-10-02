# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""ComfyUI의 DPM-Solver++ 2M 및 Flow Matching용 2M SDE 갱신식."""

import torch

from .noise import create_brownian_noise_sampler


@torch.no_grad()
def sample_dpmpp_2m(model_fn, x, sigmas, callback=None):
    """ComfyUI 기본 2M의 -log(sigma) 시간축과 이전 x0 예측을 사용한다.

    model_fn은 denoised = x - sigma * velocity를 반환한다.
    """
    old_denoised = None
    h_last = None
    dtype = x.dtype
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model_fn(x, sigma)
        if sigma_next == 0:
            x = denoised
        else:
            h = sigma.log() - sigma_next.log()
            if old_denoised is None:
                denoised_d = denoised
            else:
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - old_denoised / (2 * r)
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d
            h_last = h
        old_denoised = denoised
        x = x.to(dtype)
        if callback is not None:
            callback(i)
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model_fn, x, sigmas, callback=None, eta=1.0, s_noise=1.0, noise_sampler=None
):
    """RF logSNR 시간축에서 ComfyUI midpoint 2M SDE를 실행한다.

    양수 sigma는 1 미만이어야 한다. 시작 sigma=1의 ComfyUI식 보정은
    호출 측에서 수행한다. seed를 유지하려면 noise_sampler를 전달한다.
    """
    if len(sigmas) <= 1:
        return x
    if eta > 0 and s_noise > 0 and noise_sampler is None and len(sigmas) > 2:
        noise_sampler = create_brownian_noise_sampler(x, sigmas)
    old_denoised = None
    h_last = None
    dtype = x.dtype
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model_fn(x, sigma)
        if sigma_next == 0:
            x = denoised
        else:
            lambda_s, lambda_t = -sigma.logit(), -sigma_next.logit()
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            alpha_t = sigma_next * lambda_t.exp()
            phi = -(-h_eta).expm1()
            x = sigma_next / sigma * (-h * eta).exp() * x + alpha_t * phi * denoised
            if old_denoised is not None:
                r = h_last / h
                x = x + 0.5 * alpha_t * phi / r * (denoised - old_denoised)
            if eta > 0 and s_noise > 0:
                x = x + (
                    noise_sampler(sigma, sigma_next)
                    * sigma_next
                    * (-(-2 * h * eta).expm1()).sqrt()
                    * s_noise
                )
            h_last = h
        old_denoised = denoised
        x = x.to(dtype)
        if callback is not None:
            callback(i)
    return x
