"""DiffSynth Anima의 Z-Image 시점 분포와 FlowMatchSFTLoss 계산.

참고: DiffSynth-Studio/diffsynth/diffusion/{flow_match,loss}.py (Apache-2.0).
"""

import torch
from torch.nn import functional as F


class FlowMatchingLoss:
    def __init__(self, num_timesteps=1000, shift=3.0):
        sigmas = torch.linspace(1.0, 0.0, num_timesteps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        # DiffSynth FlowMatchScheduler.set_training_weight의 Anima 기본 1,000 시점.
        weights = torch.exp(-2 * ((self.sigmas * num_timesteps - num_timesteps / 2) / num_timesteps) ** 2)
        weights = weights - weights.min()
        self.weights = weights * (num_timesteps / weights.sum())

    def __call__(self, model, latent, prompt_embeds, t5_ids, use_gradient_checkpointing=False):
        index = torch.randint(len(self.sigmas), (1,)).item()
        sigma = self.sigmas[index].to(device=latent.device, dtype=latent.dtype)
        noise = torch.randn_like(latent)
        noisy = (1 - sigma) * latent + sigma * noise
        target = noise - latent
        prediction = model(
            x=noisy, timesteps=sigma.reshape(1), context=prompt_embeds,
            t5xxl_ids=t5_ids, use_gradient_checkpointing=use_gradient_checkpointing,
        )
        weight = self.weights[index].to(device=latent.device)
        return F.mse_loss(prediction.float(), target.float()) * weight
