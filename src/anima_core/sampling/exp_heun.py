# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""RF logSNR 좌표의 x0 기반 지수 Heun과 SDE 변형."""

import torch


@torch.no_grad()
def _sample_exp_heun(model_fn, x, sigmas, callback, eta, s_noise, noise_sampler):
    dtype = x.dtype
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    inject_noise = eta > 0 and s_noise > 0
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model_fn(x, sigma).float()
        if sigma_next == 0:
            x = denoised.to(dtype)
        else:
            lambda_s, lambda_t = -sigma.logit(), -sigma_next.logit()
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)
            # ComfyUI SEEDS-2의 r=1, solver_type="phi_2" 경로.
            sigma_mid = (-lambda_t).sigmoid()
            alpha_mid = sigma_mid * lambda_t.exp()
            alpha_t = sigma_next * lambda_t.exp()
            phi1 = (-h_eta).expm1()
            predicted = sigma_mid / sigma * (-h * eta).exp() * x.float() - alpha_mid * phi1 * denoised
            if inject_noise:
                sde_noise = (-2 * h * eta).expm1().neg().sqrt() * noise_sampler(sigma, sigma_mid).float()
                predicted = predicted + sde_noise * sigma_mid * s_noise
            denoised_next = model_fn(predicted.to(dtype), sigma_mid).float()
            phi2 = phi1 / (-h_eta) - 1
            b1 = phi1 - phi2
            updated = sigma_next / sigma * (-h * eta).exp() * x.float() - alpha_t * (b1 * denoised + phi2 * denoised_next)
            if inject_noise:
                # r=1의 마지막 구간 계수는 0이지만 원본처럼 RNG를 소비한다.
                tail_noise = noise_sampler(sigma_mid, sigma_next).float()
                updated = updated + (sde_noise + tail_noise * 0) * sigma_next * s_noise
            x = updated.to(dtype)
        if callback is not None:
            callback(i)
    return x


def sample_exp_heun_2_x0(model_fn, x, sigmas, callback=None):
    return _sample_exp_heun(model_fn, x, sigmas, callback, 0.0, 0.0, None)


def sample_exp_heun_2_x0_sde(model_fn, x, sigmas, callback=None, eta=1.0,
                           s_noise=1.0, noise_sampler=None):
    if noise_sampler is None:
        noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    return _sample_exp_heun(model_fn, x, sigmas, callback, eta, s_noise, noise_sampler)
