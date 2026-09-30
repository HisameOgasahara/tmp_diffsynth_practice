import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


def attention_forward(
    q,
    k,
    v,
    out_pattern="b n s d",
    attn_mask=None,
    scale=None,
    is_causal=False,
    **kwargs,
):
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=scale,
        is_causal=is_causal,
    )
    if out_pattern == "b n s d":
        return out
    return rearrange(out, f"b n s d -> {out_pattern}")


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    if use_gradient_checkpointing and torch.is_grad_enabled():
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return checkpoint(model, use_reentrant=False, **kwargs)
        return checkpoint(model, use_reentrant=False, **kwargs)
    return model(**kwargs)
