import torch

from ..anima_dit import LLMAdapter
from ..runtime import _stream_load


def load_llm_adapter(path, device, dtype):
    adapter = LLMAdapter(device=device, dtype=dtype, operations=torch.nn).eval().requires_grad_(False)
    return _stream_load(
        adapter, path,
        map_key=lambda key: key.removeprefix("net.").removeprefix("llm_adapter."),
        skip_key=lambda key: not key.removeprefix("net.").startswith("llm_adapter."),
    )
