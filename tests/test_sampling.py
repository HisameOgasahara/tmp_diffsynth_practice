"""작은 CPU 모델로 생성 sampler의 호환성·수치 동작·재현성을 확인합니다."""

from pathlib import Path
import ast
from functools import partial
import math
from typing import Callable, Union
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anima_core.runtime import sample_euler, sample_latents
from anima_core.logging_utils import configure_logging
from anima_core.sampling import SAMPLERS as SAMPLER_FUNCTIONS
from anima_core.sampling.runner import expected_evaluations, steps_for_evaluations
from anima_core.sampling.model_prediction import ModelPrediction
from anima_core.sampling.schedules import create_sigmas, offset_first_sigma
from anima_core.sampling.noise import create_brownian_noise_sampler, create_noise_sampler


SAMPLERS = tuple(SAMPLER_FUNCTIONS)


class AnalyticDiT(nn.Module):
    def __init__(self, mode="nonlinear"):
        super().__init__()
        self.mode = mode
        self.contexts = []
        self.timesteps = []

    def forward(self, x, timesteps, context, t5xxl_ids=None):
        self.contexts.append(float(context.mean()))
        self.timesteps.append(timesteps.detach().clone())
        if self.mode == "constant":
            return torch.full_like(x, 0.25)
        if self.mode == "linear":
            return x
        t = timesteps.reshape(-1, 1, 1, 1, 1)
        return torch.tanh(x * 0.17 + context.mean() * 0.11 + t * 0.23)


def old_euler(dit, positive, negative, height, width, seed, steps, cfg_scale,
              denoise=1.0, shift=3.0, device="cpu", dtype=torch.float32):
    """분리 이전 Euler의 연산 순서와 RNG를 고정한 회귀 기준입니다."""
    generator = torch.Generator("cpu").manual_seed(int(seed))
    latents = torch.randn((1, 16, int(height) // 8, int(width) // 8),
                          generator=generator, device="cpu", dtype=dtype).to(device)
    sigmas = torch.linspace(float(denoise), 0.0, int(steps) + 1, dtype=torch.float32)[:-1]
    sigmas = float(shift) * sigmas / (1.0 + (float(shift) - 1.0) * sigmas)
    timesteps = sigmas * 1000.0
    for i, timestep in enumerate(timesteps):
        t = timestep.reshape(1).to(device=device, dtype=dtype) / 1000.0
        positive_pred = dit(x=latents.unsqueeze(2), timesteps=t,
                            context=positive, t5xxl_ids=None).squeeze(2)
        if float(cfg_scale) == 1.0:
            velocity = positive_pred
        else:
            negative_pred = dit(x=latents.unsqueeze(2), timesteps=t,
                                context=negative, t5xxl_ids=None).squeeze(2)
            velocity = negative_pred + float(cfg_scale) * (positive_pred - negative_pred)
        sigma = sigmas[i].to(device=device, dtype=latents.dtype)
        sigma_next = (sigmas[i + 1].to(device=device, dtype=latents.dtype)
                      if i + 1 < len(sigmas) else torch.zeros((), device=device, dtype=latents.dtype))
        latents = latents + velocity * (sigma_next - sigma)
    return latents


class SamplingTests(unittest.TestCase):
    def setUp(self):
        configure_logging({"level": "CRITICAL", "progress": False})
        torch.set_num_threads(1)
        self.options = dict(positive=torch.full((1, 2, 3), 1.25),
                            negative=torch.full((1, 2, 3), -0.75),
                            height=16, width=24, seed=153, steps=7,
                            cfg_scale=4.0, denoise=0.73, shift=2.4,
                            device="cpu", dtype=torch.float32)

    def test_euler_preserves_previous_results_bitwise(self):
        for dtype in (torch.float32, torch.float16):
            for cfg_scale in (1.0, 4.0):
                with self.subTest(dtype=dtype, cfg_scale=cfg_scale):
                    options = dict(self.options, dtype=dtype, cfg_scale=cfg_scale)
                    expected = old_euler(AnalyticDiT(), **options)
                    actual = sample_latents(AnalyticDiT(), sampler="euler", **options)
                    self.assertTrue(torch.equal(actual, expected),
                                    f"max delta={float((actual - expected).abs().max())}")
                    self.assertTrue(torch.equal(sample_euler(AnalyticDiT(), **options), expected))

    def test_cfg_one_skips_negative_model_calls(self):
        for sampler in SAMPLERS:
            with self.subTest(sampler=sampler):
                model = AnalyticDiT()
                sample_latents(model, sampler=sampler, **dict(self.options, cfg_scale=1.0))
                self.assertGreater(len(model.contexts), 0)
                self.assertEqual(set(model.contexts), {1.25})

    def test_heun_integrates_constant_velocity(self):
        options = dict(self.options, cfg_scale=1.0)
        generator = torch.Generator("cpu").manual_seed(options["seed"])
        initial = torch.randn((1, 16, 2, 3), generator=generator)
        start = options["shift"] * options["denoise"] / (1 + (options["shift"] - 1) * options["denoise"])
        expected = initial - 0.25 * start
        actual = sample_latents(AnalyticDiT("constant"), sampler="heun", **options)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=3e-7)

    def test_heun_reduces_error_for_linear_velocity(self):
        options = dict(self.options, cfg_scale=1.0, steps=24, denoise=0.8, shift=1.0)
        generator = torch.Generator("cpu").manual_seed(options["seed"])
        initial = torch.randn((1, 16, 2, 3), generator=generator)
        expected = initial * torch.exp(torch.tensor(-options["denoise"]))
        euler = sample_latents(AnalyticDiT("linear"), sampler="euler", **options)
        heun = sample_latents(AnalyticDiT("linear"), sampler="heun", **options)
        self.assertLess(float((heun - expected).square().mean()),
                        float((euler - expected).square().mean()) / 4)

    def test_all_samplers_handle_rf_endpoints_and_reproduce_seed(self):
        for sampler in SAMPLERS:
            for steps in (1, 5):
                with self.subTest(sampler=sampler, steps=steps):
                    options = dict(self.options, steps=steps, denoise=1.0)
                    actual = sample_latents(AnalyticDiT(), sampler=sampler, **options)
                    repeated = sample_latents(AnalyticDiT(), sampler=sampler, **options)
                    self.assertEqual(actual.shape, (1, 16, 2, 3))
                    self.assertEqual(actual.dtype, options["dtype"])
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(torch.equal(actual, repeated))

    def test_stochastic_samplers_reproduce_with_partial_denoise(self):
        for sampler in ("euler_ancestral", "dpmpp_2m_sde", "er_sde", "exp_heun_2_x0_sde", "sa_solver"):
            with self.subTest(sampler=sampler):
                actual = sample_latents(AnalyticDiT(), sampler=sampler, **self.options)
                repeated = sample_latents(AnalyticDiT(), sampler=sampler, **self.options)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, repeated))

    def test_dpmpp_rf_integrates_constant_velocity(self):
        options = dict(self.options, cfg_scale=1.0)
        expected = old_euler(AnalyticDiT("constant"), **options)
        actual = sample_latents(AnalyticDiT("constant"), sampler="dpmpp_2m", **options)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_brownian_noise_shares_a_seeded_path_across_intervals(self):
        x = torch.zeros(1, 2, 2, 3)
        sigmas = torch.tensor([0.75, 0.5, 0.25, 0.0])
        noise = create_brownian_noise_sampler(x, sigmas, seed=17)
        start, middle, end = sigmas[:3]
        whole = noise(start, end) * (start - end).sqrt()
        first = noise(start, middle) * (start - middle).sqrt()
        second = noise(middle, end) * (middle - end).sqrt()
        torch.testing.assert_close(whole, first + second, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(noise(start, end), -noise(end, start), rtol=0, atol=0)
        repeated = create_brownian_noise_sampler(x, sigmas, seed=17)
        torch.testing.assert_close(noise(start, end), repeated(start, end), rtol=0, atol=0)

    def test_extra_noise_does_not_repeat_initial_noise(self):
        initial = torch.randn((1, 16, 2, 3), generator=torch.Generator().manual_seed(153))
        extra = create_noise_sampler(initial, seed=153)(torch.tensor(0.8), torch.tensor(0.4))
        repeated = create_noise_sampler(initial, seed=153)(torch.tensor(0.8), torch.tensor(0.4))
        self.assertFalse(torch.equal(initial, extra))
        self.assertTrue(torch.equal(extra, repeated))

    def test_all_samplers_preserve_model_precision(self):
        for sampler in SAMPLERS:
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(sampler=sampler, dtype=dtype):
                    model = AnalyticDiT()
                    actual = sample_latents(model, sampler=sampler,
                                            **dict(self.options, dtype=dtype, denoise=1.0))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(all(t.dtype == dtype for t in model.timesteps))

    def test_runner_preserves_existing_sampling_results(self):
        for sampler in ("euler", "heun", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "er_sde"):
            for dtype in (torch.float32, torch.float16):
                with self.subTest(sampler=sampler, dtype=dtype):
                    options = dict(self.options, dtype=dtype, denoise=1.0)
                    initial = torch.randn((1, 16, 2, 3), dtype=dtype,
                                          generator=torch.Generator("cpu").manual_seed(options["seed"]))
                    sigmas = create_sigmas(options["steps"], options["denoise"], options["shift"])
                    sampler_options = {}
                    if sampler in {"euler_ancestral", "er_sde"}:
                        sampler_options.update(s_noise=1.0, noise_sampler=create_noise_sampler(initial, options["seed"]))
                    if sampler == "euler_ancestral":
                        sampler_options["eta"] = 1.0
                    if sampler == "dpmpp_2m_sde":
                        sampler_options.update(eta=1.0, s_noise=1.0,
                                               noise_sampler=create_brownian_noise_sampler(initial, sigmas, options["seed"]))
                    if sampler in {"dpmpp_2m_sde", "er_sde"}:
                        sigmas = offset_first_sigma(sigmas, options["shift"])
                    prediction = ModelPrediction(AnalyticDiT(), options["positive"],
                                                 options["negative"], options["cfg_scale"])
                    expected = SAMPLER_FUNCTIONS[sampler](prediction, initial, sigmas, **sampler_options)
                    actual = sample_latents(AnalyticDiT(), sampler=sampler, **options)
                    self.assertTrue(torch.equal(actual, expected))

    def test_evaluation_budget_matches_execution(self):
        for sampler in SAMPLERS:
            for target in (1, 10, 30, 31):
                with self.subTest(sampler=sampler, target=target):
                    steps = steps_for_evaluations(sampler, target)
                    planned = expected_evaluations(sampler, steps)
                    self.assertLessEqual(planned, target)
                    self.assertLessEqual(target - planned, 1)
                    model = AnalyticDiT()
                    sample_latents(model, sampler=sampler, **dict(self.options, steps=steps))
                    self.assertEqual(len(model.contexts), planned * 2)

    def test_reject_invalid_sampling_options_before_model_call(self):
        invalid = {"sampler": ["unknown"], "steps": [0, -1, 1.5, True],
                   "denoise": [0, -0.1, 1.1, float("nan")],
                   "shift": [0, -1, float("inf")],
                   "eta": [-0.1, 1.1, float("nan")],
                   "s_noise": [-1, float("inf")]}
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    model = AnalyticDiT()
                    with self.assertRaises(ValueError):
                        sample_latents(model, **dict(self.options, **{name: value}))
                    self.assertEqual(model.contexts, [])

    def test_adapted_equations_match_local_comfy_source(self):
        # ComfyUI를 설치하지 않고 인접한 참고 checkout의 원 함수를 격리합니다.
        source_path = ROOT.parent / "ComfyUI/comfy/k_diffusion/sampling.py"
        if not source_path.exists():
            self.skipTest("ComfyUI 참고 checkout이 없습니다.")
        names = {"sample_euler_ancestral_RF", "sample_dpmpp_2m", "sample_dpmpp_2m_sde", "sample_er_sde", "sample_seeds_2", "sample_exp_heun_2_x0",
                 "sample_exp_heun_2_x0_sde", "sample_sa_solver", "res_multistep",
                 "sample_res_multistep", "sample_gradient_estimation",
                 "get_ancestral_step", "ei_h_phi_1", "ei_h_phi_2", "default_noise_sampler"}
        source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        source_tree.body = [node for node in source_tree.body
                            if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"torch": torch, "partial": partial,
                     "to_d": lambda x, sigma, denoised: (x - denoised) / sigma,
                     "half_log_snr_to_sigma": lambda value, model_sampling: (-value).sigmoid(),
                     "trange": lambda count, **kwargs: range(count),
                     "sigma_to_half_log_snr": lambda sigma, model_sampling: -sigma.logit(),
                     "offset_first_sigma_for_snr": lambda sigmas, model_sampling: sigmas}
        exec(compile(source_tree, str(source_path), "exec"), namespace)
        helper_path = source_path.with_name("sa_solver.py")
        helpers = ast.parse(helper_path.read_text(encoding="utf-8"))
        helpers.body = [node for node in helpers.body if isinstance(node, ast.FunctionDef)]
        helper_namespace = {"torch": torch, "math": math, "Callable": Callable, "Union": Union}
        exec(compile(helpers, str(helper_path), "exec"), helper_namespace)
        namespace["sa_solver"] = SimpleNamespace(**{key: value for key, value in helper_namespace.items()
                                                   if not key.startswith("__")})
        model_sampling = SimpleNamespace(noise_scale=1.0)

        class DenoisedModel:
            inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
                get_model_object=lambda name: model_sampling))

            def __call__(self, x, sigma, **kwargs):
                return x * 0.7 + sigma.sin() * 0.1

        initial = torch.randn((1, 2, 2, 3), generator=torch.Generator().manual_seed(73))
        sigmas = torch.tensor([0.8, 0.64, 0.42, 0.19, 0.07, 0.0])
        noise = lambda sigma, sigma_next: torch.full_like(initial, 0.3)
        for sampler, source_name in (("euler_ancestral", "sample_euler_ancestral_RF"),
                                     ("dpmpp_2m", "sample_dpmpp_2m"),
                                     ("dpmpp_2m_sde", "sample_dpmpp_2m_sde"),
                                     ("er_sde", "sample_er_sde"),
                                     ("exp_heun_2_x0", "sample_exp_heun_2_x0"),
                                     ("exp_heun_2_x0_sde", "sample_exp_heun_2_x0_sde"),
                                     ("sa_solver", "sample_sa_solver"),
                                     ("res_multistep", "sample_res_multistep"),
                                     ("gradient_estimation", "sample_gradient_estimation")):
            with self.subTest(sampler=sampler):
                options = {} if sampler in {"dpmpp_2m", "exp_heun_2_x0", "res_multistep", "gradient_estimation"} else dict(s_noise=0.4, noise_sampler=noise)
                if sampler in {"euler_ancestral", "dpmpp_2m_sde", "exp_heun_2_x0_sde"}:
                    options["eta"] = 0.7
                if sampler == "sa_solver":
                    options["tau_func"] = lambda sigma: 0.7 if 0.3 <= sigma <= 0.7 else 0.0
                expected = namespace[source_name](DenoisedModel(), initial.clone(), sigmas.clone(), **options)
                actual = SAMPLER_FUNCTIONS[sampler](DenoisedModel(), initial.clone(), sigmas.clone(), **options)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
