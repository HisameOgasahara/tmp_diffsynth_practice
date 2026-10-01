import torch
from torch.nn import functional as F


class FlowMatchingLoss:
    def __init__(self, training=None):
        training = training or {}
        self.sigmoid_scale = training.get("sigmoid_scale", 1.0)
        self.sigmoid_bias = training.get("sigmoid_bias", 0.0)

    def __call__(self, model, latent, prompt_embeds, t5_ids, use_gradient_checkpointing=False,
                 target_attention_mask=None, source_attention_mask=None):
        sigma = torch.sigmoid(
            self.sigmoid_scale * torch.randn((latent.shape[0],), device=latent.device) + self.sigmoid_bias
        )
        broadcast_sigma = sigma.reshape(-1, *([1] * (latent.ndim - 1)))
        noise = torch.randn_like(latent)
        noisy = ((1 - broadcast_sigma) * latent + broadcast_sigma * noise).to(latent.dtype)
        target = noise - latent
        masks = {}
        if target_attention_mask is not None:
            masks["target_attention_mask"] = target_attention_mask
        if source_attention_mask is not None:
            masks["source_attention_mask"] = source_attention_mask
        prediction = model(
            x=noisy, timesteps=sigma.to(latent.dtype), context=prompt_embeds,
            t5xxl_ids=t5_ids, use_gradient_checkpointing=use_gradient_checkpointing,
            **masks,
        )
        per_sample_loss = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
        return per_sample_loss.mean()
