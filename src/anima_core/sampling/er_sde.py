# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""ComfyUI의 RF ER-SDE 구현을 독립적인 예측 함수에 맞춰 적용한다.

원 논문: https://arxiv.org/abs/2309.06169
원 구현: https://github.com/QinpengCui/ER-SDE-Solver
"""

import torch

from ..profiling import profile_range


def _scale_noise(er_lambda):
    return er_lambda * (er_lambda.pow(0.3).exp() + 10.0)


@torch.no_grad()
def sample_er_sde(model_fn, x, sigmas, callback=None, s_noise=1.0, noise_sampler=None):
    """200점 수치 적분과 최대 3단계 예측 이력으로 RF ER-SDE를 수행한다."""
    if noise_sampler is None:
        noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    er_lambdas = sigmas / (1 - sigmas)
    integration_points = 200
    point_indices = torch.arange(integration_points, device=x.device, dtype=torch.float32)
    old_denoised = None
    old_derivative = None
    dtype = x.dtype

    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i]).float()
        with profile_range("sampling/er_sde_update"):
            if sigmas[i + 1] == 0:
                x = denoised.to(dtype)
            else:
                lambda_s, lambda_t = er_lambdas[i], er_lambdas[i + 1]
                alpha_s, alpha_t = 1 - sigmas[i], 1 - sigmas[i + 1]
                scaled_t = _scale_noise(lambda_t)
                ratio = scaled_t / _scale_noise(lambda_s)
                updated = (alpha_t / alpha_s) * ratio * x.float()
                updated = updated + alpha_t * (1 - ratio) * denoised

                if i >= 1:
                    dt = lambda_t - lambda_s
                    integration_step = -dt / integration_points
                    positions = lambda_t + point_indices * integration_step
                    scaled_positions = _scale_noise(positions)
                    integral = (1 / scaled_positions).sum() * integration_step
                    derivative = (denoised - old_denoised) / (lambda_s - er_lambdas[i - 1])
                    updated = updated + alpha_t * (dt + integral * scaled_t) * derivative

                    if i >= 2:
                        integral_u = ((positions - lambda_s) / scaled_positions).sum() * integration_step
                        second_derivative = (derivative - old_derivative) / ((lambda_s - er_lambdas[i - 2]) / 2)
                        updated = updated + alpha_t * (dt.square() / 2 + integral_u * scaled_t) * second_derivative
                    old_derivative = derivative

                if s_noise > 0:
                    noise_coeff = (lambda_t.square() - lambda_s.square() * ratio.square()).clamp_min(0).sqrt()
                    updated = updated + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]).float() * s_noise * noise_coeff
                x = updated.to(dtype)
        old_denoised = denoised
        if callback is not None:
            callback(i)
    return x
