"""샘플러 옵션 검사·스케줄·추가 노이즈 준비와 실행을 연결한다."""

import math

from . import SAMPLERS
from .noise import create_noise_sampler, create_brownian_noise_sampler
from .sa_solver import get_tau_interval_func
from .schedules import create_sigmas, offset_first_sigma, percent_to_sigma


_TWO_EVALUATION_SAMPLERS = {"heun", "exp_heun_2_x0", "exp_heun_2_x0_sde"}
_RF_SNR_SAMPLERS = {"dpmpp_2m_sde", "er_sde", "exp_heun_2_x0",
                    "exp_heun_2_x0_sde", "sa_solver"}
_NORMAL_NOISE_SAMPLERS = {"euler_ancestral", "er_sde", "exp_heun_2_x0_sde", "sa_solver"}
_ETA_SAMPLERS = {"euler_ancestral", "dpmpp_2m_sde", "exp_heun_2_x0_sde"}


def validate_sampling_options(sampler, steps, denoise, shift, eta, s_noise):
    if sampler not in SAMPLERS:
        raise ValueError(f"알 수 없는 sampler: {sampler}, 사용 가능: {list(SAMPLERS)}")
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError("steps는 1 이상의 정수여야 합니다.")
    if not math.isfinite(float(denoise)) or not 0 < float(denoise) <= 1:
        raise ValueError("denoise는 0보다 크고 1 이하여야 합니다.")
    if not math.isfinite(float(shift)) or float(shift) <= 0:
        raise ValueError("shift는 0보다 큰 유한한 값이어야 합니다.")
    if not math.isfinite(float(eta)) or not 0 <= float(eta) <= 1:
        raise ValueError("eta는 0 이상 1 이하여야 합니다.")
    if not math.isfinite(float(s_noise)) or float(s_noise) < 0:
        raise ValueError("s_noise는 0 이상의 유한한 값이어야 합니다.")


def expected_evaluations(sampler, steps):
    validate_sampling_options(sampler, steps, 1.0, 3.0, 1.0, 1.0)
    return 2 * int(steps) - 1 if sampler in _TWO_EVALUATION_SAMPLERS else int(steps)


def steps_for_evaluations(sampler, target):
    expected_evaluations(sampler, target)
    return max(1, (int(target) + 1) // 2) if sampler in _TWO_EVALUATION_SAMPLERS else int(target)


def run_sampler(model_fn, latents, sampler, steps, seed, denoise=1.0,
                shift=3.0, eta=1.0, s_noise=1.0, callback=None):
    validate_sampling_options(sampler, steps, denoise, shift, eta, s_noise)
    sigmas = create_sigmas(steps, denoise, shift)
    options = {}
    if sampler in _NORMAL_NOISE_SAMPLERS:
        options.update(s_noise=float(s_noise), noise_sampler=create_noise_sampler(latents, seed))
    if sampler in _ETA_SAMPLERS:
        options["eta"] = float(eta)
    if sampler == "dpmpp_2m_sde":
        options["s_noise"] = float(s_noise)
        # 기존 Brownian 경로는 시작 sigma 보정 전에 준비한다.
        if int(steps) > 1 and eta > 0 and s_noise > 0:
            options["noise_sampler"] = create_brownian_noise_sampler(latents, sigmas, seed)
    if sampler == "sa_solver":
        options["tau_func"] = get_tau_interval_func(
            percent_to_sigma(0.2, shift), percent_to_sigma(0.8, shift), float(eta))
    if sampler in _RF_SNR_SAMPLERS:
        sigmas = offset_first_sigma(sigmas, shift)
    return SAMPLERS[sampler](model_fn, latents, sigmas, callback=callback, **options)
