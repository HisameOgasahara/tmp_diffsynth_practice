# SPDX-License-Identifier: GPL-3.0-only
# Adapted from ComfyUI comfy/k_diffusion/sampling.py (1b883beab11c04a2eb82cf6a3ee64294f3303897).
"""추가 노이즈와 ComfyUI 방식의 CPU Brownian 경로를 생성한다."""

import torch


def create_noise_sampler(x, seed=None):
    """독립 정규 노이즈를 CPU에서 생성해 latent 장치로 옮긴다."""
    generator = None
    if seed is not None:
        # 초기 latent와 같은 CPU 난수열을 반복하지 않습니다.
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

    def sample_noise(sigma, sigma_next):
        return torch.randn(
            x.shape, dtype=x.dtype, device="cpu", generator=generator
        ).to(x.device)

    return sample_noise


class _BatchedBrownianTree:
    """ComfyUI의 BrownianTree 배치 및 구간 방향 처리를 유지한다."""

    def __init__(self, x, t0, t1, seed):
        import torchsde

        t0, t1, self.sign = self._sort(t0, t1)
        w0 = torch.zeros_like(x)
        self.batched = isinstance(seed, (tuple, list))
        if self.batched:
            if len(seed) != x.shape[0]:
                raise ValueError("Brownian seed 목록 길이는 배치 크기와 같아야 합니다.")
            w0 = w0[0]
            seeds = seed
        else:
            if seed is None:
                seed = torch.randint(0, 2**63 - 1, ()).item()
            seeds = (seed,)
        t0, t1 = t0.detach().cpu(), t1.detach().cpu()
        w0 = w0.detach().cpu()
        self.trees = tuple(
            torchsde.BrownianTree(t0, w0, t1, entropy=entropy)
            for entropy in seeds
        )

    @staticmethod
    def _sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self._sort(t0, t1)
        device, dtype = t0.device, t0.dtype
        t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        increment = torch.stack([tree(t0, t1) for tree in self.trees])
        increment = increment.to(device=device, dtype=dtype) * (self.sign * sign)
        return increment if self.batched else increment[0]


def create_brownian_noise_sampler(x, sigmas, seed=None):
    """ComfyUI 기본값처럼 sigma 시간축의 CPU Brownian 증분을 정규화한다.

    구간이 겹칠 때 동일한 Brownian 경로를 공유한다. sigma=0의 최종
    복원 구간에는 호출하지 않는다. torchsde는 이 함수를 사용할 때 로드한다.
    """
    positive_sigmas = sigmas[sigmas > 0]
    if positive_sigmas.numel() < 2:
        raise ValueError("Brownian 노이즈에는 서로 다른 양수 sigma 구간이 필요합니다.")
    sigma_min, sigma_max = positive_sigmas.min(), positive_sigmas.max()
    if sigma_min == sigma_max:
        raise ValueError("Brownian 노이즈에는 서로 다른 양수 sigma 구간이 필요합니다.")
    tree = _BatchedBrownianTree(x, sigma_min, sigma_max, seed)

    def sample_noise(sigma, sigma_next):
        sigma = torch.as_tensor(sigma, device=x.device)
        sigma_next = torch.as_tensor(sigma_next, device=x.device)
        return tree(sigma, sigma_next) / (sigma_next - sigma).abs().sqrt()

    return sample_noise
