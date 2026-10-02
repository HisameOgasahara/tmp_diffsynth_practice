# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI sampling.py and sa_solver.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""RF용 Stochastic Adams Solver의 계수 계산과 기본 예측·보정."""

import math
from typing import Callable, Union

import torch


def compute_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float) -> torch.Tensor:
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms_factored = t ** p - s ** p * (-tau_mul * h).exp()
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - recursive_depth_mat * math.log(tau_mul)
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    return recursive_coeff_mat @ product_terms_factored


def compute_simple_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, is_corrector_step: bool=False) -> torch.Tensor:
    tau_mul = 1 + tau_t ** 2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        b_2 = alpha_t * (0.5 * tau_mul * h ** 2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])


def compute_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, simple_order_2: bool=False, is_corrector_step: bool=False) -> torch.Tensor:
    num_timesteps = curr_lambdas.shape[0]
    if simple_order_2 and num_timesteps == 2:
        return compute_simple_stochastic_adams_b_coeffs(sigma_next, curr_lambdas, lambda_s, lambda_t, tau_t, is_corrector_step)
    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_T = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_T, exp_integral_coeffs)
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


def get_tau_interval_func(start_sigma: float, end_sigma: float, eta: float=1.0) -> Callable[[Union[torch.Tensor, float]], float]:

    def tau_func(sigma: Union[torch.Tensor, float]) -> float:
        if eta <= 0:
            return 0.0
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return eta if start_sigma >= sigma >= end_sigma else 0.0
    return tau_func


@torch.no_grad()
def sample_sa_solver(model_fn, x, sigmas, callback=None, s_noise=1.0, noise_sampler=None, tau_func=None):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    if noise_sampler is None:
        noise_sampler = lambda sigma, sigma_next: torch.randn_like(x)
    dtype = x.dtype
    x = x.float()
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    predictor_order, corrector_order = 3, 4
    simple_order_2 = False

    lambdas = -sigmas.logit()
    if tau_func is None:
        raise ValueError("SA Solver의 노이즈 구간 함수를 전달하세요.")

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x  # x: current state, x_pred: predicted next state

    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    # Lower order near the end to improve stability
    lower_order_to_end = sigmas[-1].item() == 0

    for i in range(len(sigmas) - 1):
        # Evaluation
        denoised = model_fn(x_pred.to(dtype), sigmas[i]).float()
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or sigmas[i + 1] == 0:
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            # Update by the predicted state
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)    # (B, K, ...)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                # The noise from the previous predictor step
                x = x + noise


        # Predictor
        if sigmas[i + 1] == 0:
            # Denoising step
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)    # (B, K, ...)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))  # (B, ...)
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]).float() * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
        x_pred = x_pred.to(dtype).float()
        if callback is not None:
            callback(i)
    return x_pred.to(dtype)
