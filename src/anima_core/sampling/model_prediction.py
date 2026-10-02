"""Anima velocity·CFG 예측과 샘플러용 denoised 변환."""

import torch

from ..profiling import profile_range


class ModelPrediction:
    def __init__(self, dit, positive, negative, cfg_scale):
        self.dit = dit
        self.positive = positive
        self.negative = negative
        self.cfg_scale = float(cfg_scale)

    def velocity(self, x, sigma):
        # 기존 Euler의 timestep 변환 순서를 유지합니다.
        timestep = (sigma * 1000.0).reshape(1).to(device=x.device, dtype=x.dtype) / 1000.0
        with profile_range("sampling/positive_dit"):
            positive = self.dit(x=x.unsqueeze(2), timesteps=timestep,
                                context=self.positive, t5xxl_ids=None).squeeze(2)
        if self.cfg_scale == 1.0:
            return positive
        with profile_range("sampling/negative_dit"):
            negative = self.dit(x=x.unsqueeze(2), timesteps=timestep,
                                context=self.negative, t5xxl_ids=None).squeeze(2)
        with profile_range("sampling/cfg"):
            return negative + self.cfg_scale * (positive - negative)

    def __call__(self, x, sigma):
        velocity = self.velocity(x, sigma)
        return x.float() - sigma.to(device=x.device, dtype=torch.float32) * velocity.float()
