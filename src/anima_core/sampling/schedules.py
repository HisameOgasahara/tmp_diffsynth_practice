"""Z-Image sigma 목록과 RF logSNR 끝점 처리."""

import torch


def z_image_schedule(steps, denoise=1.0, shift=3.0):
    sigmas = torch.linspace(float(denoise), 0.0, int(steps) + 1, dtype=torch.float32)[:-1]
    sigmas = float(shift) * sigmas / (1.0 + (float(shift) - 1.0) * sigmas)
    return sigmas, sigmas * 1000.0


def create_sigmas(steps, denoise=1.0, shift=3.0):
    sigmas, _ = z_image_schedule(steps, denoise, shift)
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def offset_first_sigma(sigmas, shift):
    # ComfyUI ModelSamplingDiscreteFlow.percent_to_sigma(1e-4).
    if sigmas[0] >= 1:
        sigmas = sigmas.clone()
        first = 1.0 - 1e-4
        sigmas[0] = float(shift) * first / (1.0 + (float(shift) - 1.0) * first)
    return sigmas


def percent_to_sigma(percent, shift):
    sigma = 1.0 - float(percent)
    return float(shift) * sigma / (1.0 + (float(shift) - 1.0) * sigma)
