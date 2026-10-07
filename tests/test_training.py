"""일반 LoRA의 gradient·생성 호환·학습 재개를 작은 CPU 모델로 확인합니다."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anima_core.lora import fuse_lora
from anima_core.ops import gradient_checkpoint_forward
from anima_core.training.config import load_config, write_config
from anima_core.training.dataset import CachedDataset, SampleStream, discover_pairs, prepare_cache, read_image
from anima_core.training.preprocessing import choose_image_size, preprocess_images, resize_image
from anima_core.training.flow_matching import FlowMatchingLoss
from anima_core.training.lora import collect_lora_weights, find_target_modules, inject_lora
from anima_core.training.trainer import create_scheduler, train_model


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return torch.tanh(self.proj(x))


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x, timesteps, context, t5xxl_ids=None, use_gradient_checkpointing=False,
                target_attention_mask=None, source_attention_mask=None):
        value = x.movedim(1, -1) + context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1) + timesteps.reshape(-1, 1, 1, 1, 1)
        for block in self.blocks:
            value = gradient_checkpoint_forward(block, use_gradient_checkpointing, x=value)
        return value.movedim(-1, 1)


def make_config():
    config = load_config(ROOT / "configs/anima_lora.toml")
    config["dataset"].update(batch_size=2, repeat=2)
    config["lora"].update(rank=2, alpha=8, target_modules="proj")
    config["training"].update(max_steps=4, gradient_accumulation_steps=2)
    config["runtime"]["mixed_precision"] = "no"
    config["checkpoint"]["save_steps"] = 2
    return config


def make_dataset(path):
    path.mkdir()
    generator = torch.Generator().manual_seed(100)
    records = []
    for index in range(3):
        filename = f"{index}.safetensors"
        save_file({"latent": torch.randn(4, 1, 2, 2, generator=generator),
                   "crossattn_emb": torch.randn(512, 4, generator=generator)}, str(path / filename))
        records.append({"file": filename})
    save_file({"crossattn_emb": torch.zeros(512, 4)},
              str(path / "empty.safetensors"))
    (path / "manifest.json").write_text(json.dumps({"fingerprint": "tiny-fixture", "records": records}), encoding="utf-8")
    return CachedDataset(path)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_config_roundtrip_and_invalid_fields(self):
        config = make_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.toml"
            write_config(path, config)
            self.assertEqual(load_config(path), config)
            path.write_text(path.read_text(encoding="utf-8") + "unknown = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_diffsynth_auto_target_rule(self):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 128)) for _ in range(2)])
        model.outside = nn.Linear(512, 512)
        self.assertEqual(find_target_modules(model), ["blocks.0.0", "blocks.1.0"])

    def test_adapter_is_excluded_from_auto_and_explicit_lora_targets(self):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(512, 512)) for _ in range(2)])
        model.llm_adapter = nn.Module()
        model.llm_adapter.blocks = nn.ModuleList([nn.Sequential(nn.Linear(512, 512)) for _ in range(2)])
        self.assertEqual(find_target_modules(model), ["blocks.0.0", "blocks.1.0"])
        config = make_config()["lora"]
        config["target_modules"] = ""
        targets = inject_lora(model, config)
        self.assertEqual(targets, ["blocks.0.0", "blocks.1.0"])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.llm_adapter.parameters()))
        latent = torch.randn(2, 512)
        loss = sum(block(latent).square().mean() for block in model.blocks)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.llm_adapter.parameters()))
        self.assertFalse(any("llm_adapter" in name for name in collect_lora_weights(model)))
        explicit = TinyDiT()
        explicit.llm_adapter = TinyBlock()
        config["target_modules"] = "proj"
        self.assertEqual(inject_lora(explicit, config), ["blocks.0.proj", "blocks.1.proj"])
        self.assertTrue(all(not parameter.requires_grad for parameter in explicit.llm_adapter.parameters()))
        config["target_modules"] = "llm_adapter.proj"
        with self.assertRaises(ValueError):
            inject_lora(explicit, config)

    def test_standalone_adapter_loads_only_prefixed_weights_and_freezes(self):
        from anima_core.training.conditioning import load_llm_adapter

        expected = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                    "bias": torch.tensor([1.0, 2.0, 3.0])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dit.safetensors"
            for prefix in ("net.llm_adapter.", "llm_adapter."):
                weights = {prefix + name: tensor for name, tensor in expected.items()}
                weights["net.blocks.0.unrelated"] = torch.ones(7)
                save_file(weights, str(path))
                adapter = nn.Linear(4, 3)
                with patch("anima_core.training.conditioning.LLMAdapter", return_value=adapter):
                    loaded = load_llm_adapter(path, device="cpu", dtype=torch.float32)
                self.assertIs(loaded, adapter)
                self.assertFalse(loaded.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
                for name, tensor in loaded.state_dict().items():
                    torch.testing.assert_close(tensor, expected[name])

    def test_train_removes_adapter_before_optimizer_training(self):
        from anima_core.training.trainer import train

        config = make_config()
        model = TinyDiT()
        model.llm_adapter = TinyBlock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root / "cache")
            weights = root / "dit.safetensors"
            weights.write_bytes(b"fixture")
            stat = weights.stat()
            dataset.manifest["identity"] = {"weights": {"dit": {
                "path": str(weights.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}}
            (dataset.cache_path / "manifest.json").write_text(json.dumps(dataset.manifest), encoding="utf-8")
            with patch("anima_core.runtime.load_dit", return_value=model), \
                 patch("anima_core.training.trainer.train_model", return_value={"step": 4}) as trainer:
                result = train(config, dataset.cache_path, weights, root / "output", device="cpu", tensorboard=False)
            self.assertEqual(result, {"step": 4})
            self.assertIs(trainer.call_args.args[0], model)
            self.assertFalse(hasattr(model, "llm_adapter"))
            self.assertFalse(any("llm_adapter" in name for name, _ in model.named_parameters()))

    def test_zero_init_and_only_lora_gradients_with_checkpoint(self):
        model = TinyDiT()
        frozen = deepcopy(model)
        inject_lora(model, make_config()["lora"])
        latent = torch.randn(1, 4, 1, 2, 2)
        prompt = torch.randn(1, 5, 4)
        ids = torch.ones(1, 3, dtype=torch.long)
        with torch.no_grad():
            torch.testing.assert_close(model(latent, torch.ones(1), prompt), frozen(latent, torch.ones(1), prompt))
            self.assertTrue(all(torch.count_nonzero(parameter) == 0
                                for name, parameter in model.named_parameters() if "lora_B" in name))
        loss = FlowMatchingLoss()(model, latent, prompt, ids, True)
        loss.backward()
        trained = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(all("lora_" in name for name, _ in trained))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for _, parameter in trained))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad))

    def test_export_fuses_with_existing_generation_loader(self):
        model = TinyDiT()
        frozen = deepcopy(model)
        inject_lora(model, make_config()["lora"])
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_()
        latent, prompt = torch.randn(1, 4, 1, 2, 2), torch.randn(1, 5, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lora.safetensors"
            save_file(collect_lora_weights(model), str(path))
            self.assertEqual(fuse_lora(frozen, path), 2)
        with torch.no_grad():
            torch.testing.assert_close(model(latent, torch.ones(1), prompt), frozen(latent, torch.ones(1), prompt))

    def test_sigmoid_timesteps_and_unweighted_loss_match_reference(self):
        normals = torch.tensor([-2.0, 0.0, 2.0])
        latent = torch.randn(3, 4, 1, 2, 2)
        noise = torch.randn_like(latent)
        prompt = torch.randn(3, 5, 4)
        model = TinyDiT()
        for training in (None, {"sigmoid_scale": 1.7, "sigmoid_bias": -0.3}):
            scale = 1.0 if training is None else training["sigmoid_scale"]
            bias = 0.0 if training is None else training["sigmoid_bias"]
            sigma = torch.sigmoid(normals * scale + bias)
            noisy = (1 - sigma.reshape(-1, 1, 1, 1, 1)) * latent + sigma.reshape(-1, 1, 1, 1, 1) * noise
            prediction = model(noisy, sigma, prompt)
            expected = torch.nn.functional.mse_loss(prediction.float(), (noise - latent).float())
            with patch("torch.randn", return_value=normals), patch("torch.randn_like", return_value=noise), \
                 patch.object(model, "forward", wraps=model.forward) as forward:
                actual = FlowMatchingLoss(training)(model, latent, prompt, None)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(forward.call_args.kwargs["timesteps"], sigma)
            torch.testing.assert_close(forward.call_args.kwargs["x"], noisy)

    def test_cosine_scheduler_matches_transformers_warmup(self):
        from transformers import get_cosine_schedule_with_warmup

        for total, ratio in ((11, 0.24), (8, 0.0), (8, 0.25)):
            config = make_config()["training"]
            config.update(max_steps=total, warmup_ratio=ratio, lr_scheduler="cosine")
            optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=0.01)
            reference_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=0.01)
            scheduler = create_scheduler(optimizer, config)
            reference = get_cosine_schedule_with_warmup(reference_optimizer,
                                                       num_warmup_steps=int(total * ratio),
                                                       num_training_steps=total)
            for _ in range(total + 1):
                self.assertAlmostEqual(scheduler.get_last_lr()[0], reference.get_last_lr()[0], places=14)
                optimizer.step()
                reference_optimizer.step()
                scheduler.step()
                reference.step()

    def test_constant_scheduler_keeps_lr_and_rejects_positive_warmup(self):
        config = make_config()["training"]
        config.update(max_steps=8, warmup_ratio=0.0, lr_scheduler="constant")
        optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=0.01)
        scheduler = create_scheduler(optimizer, config)
        for _ in range(config["max_steps"] + 1):
            self.assertEqual(scheduler.get_last_lr()[0], 0.01)
            optimizer.step()
            scheduler.step()
        config["warmup_ratio"] = 0.25
        with self.assertRaises(ValueError):
            create_scheduler(optimizer, config)

    def test_trainer_loss_average_replaces_first_step_at_epoch_boundary(self):
        config = make_config()
        config["dataset"].update(batch_size=2, repeat=4, caption_dropout_rate=0)
        config["training"].update(max_steps=4, gradient_accumulation_steps=2)
        model = TinyDiT()
        inject_lora(model, config["lora"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_model(model, config, make_dataset(root / "cache"), root / "output",
                        device="cpu", tensorboard=False)
            metrics = [json.loads(line) for line in (root / "output/metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(metrics), 4)
        self.assertAlmostEqual(metrics[2]["avr_loss"], sum(entry["loss"] for entry in metrics[:3]) / 3)
        self.assertAlmostEqual(metrics[3]["avr_loss"],
                               (metrics[3]["loss"] + metrics[1]["loss"] + metrics[2]["loss"]) / 3)

    def test_loss_recorder_replaces_epoch_steps_and_restores_average(self):
        from anima_core.training.loss_recorder import LossRecorder

        recorder = LossRecorder()
        self.assertEqual(recorder.moving_average, 0)
        recorder.add(epoch=0, step=0, loss=2.0)
        recorder.add(epoch=0, step=1, loss=4.0)
        self.assertEqual(recorder.moving_average, 3.0)
        recorder.add(epoch=1, step=0, loss=6.0)
        self.assertEqual(recorder.moving_average, 5.0)
        restored = LossRecorder()
        restored.load_state_dict(recorder.state_dict())
        self.assertEqual(restored.moving_average, 5.0)
        for current in (recorder, restored):
            current.add(epoch=1, step=1, loss=8.0)
        self.assertEqual(restored.moving_average, 7.0)
        self.assertEqual(restored.state_dict(), recorder.state_dict())

    def test_batched_loss_and_gradients_match_serial_samples(self):
        batch_model = TinyDiT()
        serial_model = deepcopy(batch_model)
        latent = torch.randn(3, 4, 1, 2, 2)
        prompt = torch.randn(3, 5, 4)
        ids = torch.arange(5).expand(3, -1)
        qwen_mask = torch.arange(5).unsqueeze(0) < torch.tensor([2, 4, 5]).unsqueeze(1)
        t5_mask = qwen_mask.clone()
        normals = torch.tensor([-1.5, -0.2, 1.5])
        noise = torch.randn_like(latent)
        loss_fn = FlowMatchingLoss()
        with patch("torch.randn", return_value=normals), patch("torch.randn_like", return_value=noise):
            batched_loss = loss_fn(batch_model, latent, prompt, ids,
                                   target_attention_mask=t5_mask, source_attention_mask=qwen_mask)
        batched_loss.backward()
        serial_loss = 0
        for index in range(3):
            sample = slice(index, index + 1)
            with patch("torch.randn", return_value=normals[sample]), \
                 patch("torch.randn_like", return_value=noise[sample]):
                serial_loss = serial_loss + loss_fn(serial_model, latent[sample], prompt[sample], ids[sample],
                                                   target_attention_mask=t5_mask[sample],
                                                   source_attention_mask=qwen_mask[sample]) / 3
        serial_loss.backward()
        torch.testing.assert_close(batched_loss, serial_loss)
        for batched, serial in zip(batch_model.parameters(), serial_model.parameters()):
            torch.testing.assert_close(batched.grad, serial.grad)

    def test_anima_adapter_masks_padding_and_ignores_source_padding(self):
        from anima_core.anima_dit import AnimaDiT, LLMAdapter

        model = AnimaDiT.__new__(AnimaDiT)
        nn.Module.__init__(model)
        model.llm_adapter = LLMAdapter(source_dim=16, target_dim=16, model_dim=16,
                                       num_layers=1, num_heads=2, operations=nn, dtype=torch.float32)
        context = torch.randn(2, 5, 16)
        ids = torch.arange(5).expand(2, -1)
        source_mask = torch.arange(5).unsqueeze(0) < torch.tensor([2, 4]).unsqueeze(1)
        target_mask = torch.arange(5).unsqueeze(0) < torch.tensor([3, 5]).unsqueeze(1)
        padded_context = context.masked_fill(~source_mask.unsqueeze(-1), 1000)
        with torch.no_grad():
            expected = model.preprocess_text_embeds(context, ids, target_attention_mask=target_mask,
                                                    source_attention_mask=source_mask)
            actual = model.preprocess_text_embeds(padded_context, ids, target_attention_mask=target_mask,
                                                  source_attention_mask=source_mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2, 512, 16))
        self.assertTrue(torch.all(actual[0, 3:] == 0))
        self.assertTrue(torch.all(actual[1, 5:] == 0))

    def test_encode_prompt_training_masks_and_inference_return(self):
        from anima_core.runtime import encode_prompt

        class FakeTokenizer:
            def __init__(self, token_count):
                self.token_count = token_count
                self.calls = []

            def __call__(self, prompts, **kwargs):
                self.calls.append((prompts, kwargs))
                length = kwargs["max_length"] if kwargs.get("padding") == "max_length" else self.token_count
                return SimpleNamespace(input_ids=torch.arange(length).unsqueeze(0),
                                       attention_mask=(torch.arange(length) < self.token_count).unsqueeze(0).long())

        class FakeTextEncoder:
            def __call__(self, input_ids, attention_mask, output_hidden_states):
                self.attention_mask = attention_mask
                return SimpleNamespace(hidden_states=[torch.ones(*input_ids.shape, 4)])

        qwen, t5, encoder = FakeTokenizer(3), FakeTokenizer(2), FakeTextEncoder()
        result = encode_prompt(encoder, qwen, t5, "caption", device="cpu", dtype=torch.float32,
                               return_attention_masks=True)
        self.assertEqual(len(result), 4)
        embeds, ids, qwen_mask, t5_mask = result
        self.assertEqual(embeds.shape, (1, 512, 4))
        self.assertEqual(ids.shape, (1, 512))
        self.assertEqual(qwen_mask.shape, (1, 512))
        self.assertEqual(t5_mask.shape, (1, 512))
        self.assertEqual(qwen_mask.dtype, torch.bool)
        self.assertEqual(t5_mask.dtype, torch.bool)
        self.assertEqual(qwen_mask.sum(), 3)
        self.assertEqual(t5_mask.sum(), 2)
        self.assertTrue(torch.all(embeds[:, :3] == 1))
        self.assertTrue(torch.all(embeds[:, 3:] == 0))
        self.assertEqual(encoder.attention_mask.dtype, torch.bool)
        for tokenizer in (qwen, t5):
            self.assertEqual(tokenizer.calls[0][1]["padding"], "max_length")
            self.assertEqual(tokenizer.calls[0][1]["max_length"], 512)
            self.assertTrue(tokenizer.calls[0][1]["truncation"])
        inference = encode_prompt(encoder, qwen, t5, "caption", device="cpu", dtype=torch.float32)
        self.assertEqual(len(inference), 2)
        self.assertEqual(inference[0].shape, (1, 512, 4))
        self.assertEqual(inference[1].shape, (1, 2))
        self.assertTrue(torch.all(inference[0] == 1))
        self.assertNotIn("padding", t5.calls[1][1])

    def test_trainer_accumulates_one_image_at_a_time(self):
        config = make_config()
        config["training"]["max_steps"] = 2
        config["dataset"]["caption_dropout_rate"] = 0
        model = TinyDiT()
        inject_lora(model, config["lora"])
        batch_shapes = []

        def record_batch(module, args, kwargs):
            batch_shapes.append((kwargs["x"].shape[0], kwargs["context"].shape))
            self.assertIsNone(kwargs["t5xxl_ids"])
            self.assertNotIn("source_attention_mask", kwargs)
            self.assertNotIn("target_attention_mask", kwargs)

        hook = model.register_forward_pre_hook(record_batch, with_kwargs=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_model(model, config, make_dataset(root / "cache"), root / "output",
                        device="cpu", tensorboard=False)
        hook.remove()
        self.assertEqual(batch_shapes, [(1, torch.Size([1, 512, 4]))] * 8)

    def test_trainer_clips_once_after_accumulation_before_optimizer_step(self):
        config = make_config()
        config["training"].update(max_steps=2, max_grad_norm=1.0)
        config["dataset"]["caption_dropout_rate"] = 0
        model = TinyDiT()
        inject_lora(model, config["lora"])
        events = []
        original_unscale = torch.amp.GradScaler.unscale_
        original_step = torch.amp.GradScaler.step
        original_clip = torch.nn.utils.clip_grad_norm_

        def unscale(scaler, optimizer):
            events.append("unscale")
            return original_unscale(scaler, optimizer)

        def clip(parameters, max_norm, *args, **kwargs):
            events.append("clip")
            parameters = list(parameters)
            self.assertEqual(max_norm, 1.0)
            self.assertTrue(all(parameter.grad is not None for parameter in parameters))
            norm = original_clip(parameters, max_norm, *args, **kwargs)
            clipped_norm = torch.stack([parameter.grad.norm() for parameter in parameters]).norm()
            self.assertLessEqual(float(clipped_norm), max_norm + 1e-6)
            return norm

        def step(scaler, optimizer, *args, **kwargs):
            events.append("step")
            return original_step(scaler, optimizer, *args, **kwargs)

        hook = model.register_forward_pre_hook(lambda *args: events.append("forward"))
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(torch.amp.GradScaler, "unscale_", unscale), \
             patch.object(torch.amp.GradScaler, "step", step), \
             patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
            root = Path(temporary)
            train_model(model, config, make_dataset(root / "cache"), root / "output",
                        device="cpu", tensorboard=False)
        hook.remove()
        self.assertEqual(events, ["forward"] * 4 + ["unscale", "clip", "step"]
                         + ["forward"] * 4 + ["unscale", "clip", "step"])

    def test_existing_dit_blocks_backpropagate_with_checkpoint(self):
        from anima_core.anima_dit import MiniTrainDIT

        model = MiniTrainDIT(
            max_img_h=8, max_img_w=8, max_frames=2, in_channels=4, out_channels=4,
            patch_spatial=2, patch_temporal=1, model_channels=64, num_blocks=2,
            num_heads=4, crossattn_emb_channels=16, pos_emb_cls="rope3d", image_model="anima",
            rope_enable_fps_modulation=False, operations=nn, dtype=torch.float32,
        )
        config = make_config()["lora"]
        config["target_modules"] = "self_attn.q_proj"
        inject_lora(model, config)
        loss = FlowMatchingLoss()(model, torch.randn(2, 4, 1, 4, 4), torch.randn(2, 5, 16), None, True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_cache_preparation_and_reuse(self):
        class FakeVAE:
            def encode(self, image, device):
                return torch.zeros(1, 16, 1, image.shape[-2] // 8, image.shape[-1] // 8)

        adapter_calls = []

        class FakeAdapter(nn.Module):
            def forward(self, source_hidden_states, target_input_ids,
                        target_attention_mask=None, source_attention_mask=None):
                adapter_calls.append((torch.is_grad_enabled(), target_attention_mask.clone(),
                                      source_attention_mask.clone()))
                return torch.full((*target_input_ids.shape, 4), 7.0)

        config = make_config()
        config["dataset"]["max_pixels"] = 262144
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            Image.new("RGB", (64, 32), "white").save(data / "image.png")
            (data / "image.txt").write_text("caption", encoding="utf-8")
            weights = {}
            for name in ("dit", "text_encoder", "vae"):
                weights[name] = root / (name + ".weights")
                weights[name].write_bytes(b"fixture")
            with patch("anima_core.runtime.load_vae", return_value=FakeVAE()) as vae_loader, \
                 patch("anima_core.runtime.load_text_encoder"), \
                 patch("anima_core.runtime.load_tokenizers", return_value=(None, None)), \
                 patch("anima_core.training.conditioning.load_llm_adapter", return_value=FakeAdapter()) as adapter_loader, \
                 patch("anima_core.runtime.encode_prompt", return_value=(
                     torch.zeros(1, 512, 4), torch.ones(1, 512, dtype=torch.long),
                     torch.arange(512).unsqueeze(0) < 3, torch.arange(512).unsqueeze(0) < 2)):
                cache = prepare_cache(config, data, root / "cache", weights, "cpu")
                dataset = CachedDataset(cache)
                width, height = choose_image_size(64, 32, 262144)
                self.assertEqual(dataset[0]["latent"].shape, (16, 1, height // 8, width // 8))
                self.assertEqual(set(dataset[0]), {"latent", "crossattn_emb"})
                self.assertEqual(set(dataset.empty), {"crossattn_emb"})
                self.assertEqual(dataset.manifest["identity"]["version"], 7)
                for tensors in (dataset[0], dataset.empty):
                    self.assertEqual(tensors["crossattn_emb"].shape, (512, 4))
                    self.assertTrue(torch.all(tensors["crossattn_emb"][:2] == 7))
                    self.assertTrue(torch.all(tensors["crossattn_emb"][2:] == 0))
                self.assertEqual(prepare_cache(config, data, root / "cache", weights, "cpu"), cache)
                self.assertEqual(vae_loader.call_count, 1)
                self.assertEqual(adapter_loader.call_count, 1)
                self.assertEqual(len(adapter_calls), 2)
                for grad_enabled, target_mask, source_mask in adapter_calls:
                    self.assertFalse(grad_enabled)
                    self.assertEqual(target_mask.sum(), 2)
                    self.assertEqual(source_mask.sum(), 3)

    def test_resume_matches_uninterrupted_training_and_logs(self):
        config = make_config()
        base = TinyDiT()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root / "cache")
            model = deepcopy(base)
            inject_lora(model, config["lora"])
            result = train_model(model, config, dataset, root / "full", device="cpu", tensorboard=True)
            self.assertEqual(result["step"], 4)
            resumed = deepcopy(base)
            inject_lora(resumed, config["lora"])
            train_model(resumed, config, dataset, root / "resume",
                        resume_from=root / "full/step-0000002", device="cpu", tensorboard=False)
            expected = load_file(str(root / "full/step-0000004/lora.safetensors"))
            actual = load_file(str(root / "resume/step-0000004/lora.safetensors"))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            expected_metrics = [json.loads(line) for line in (root / "full/metrics.jsonl").read_text().splitlines()]
            actual_metrics = [json.loads(line) for line in (root / "resume/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([entry["loss"] for entry in expected_metrics[2:]], [entry["loss"] for entry in actual_metrics])
            self.assertEqual([entry["avr_loss"] for entry in expected_metrics[2:]],
                             [entry["avr_loss"] for entry in actual_metrics])
            checkpoint_state = torch.load(root / "full/step-0000002/state.pt", weights_only=True)
            self.assertIn("loss_recorder", checkpoint_state)
            from anima_core.training.loss_recorder import LossRecorder
            saved_recorder = LossRecorder()
            saved_recorder.load_state_dict(checkpoint_state["loss_recorder"])
            self.assertEqual(saved_recorder.moving_average, expected_metrics[1]["avr_loss"])
            self.assertTrue(list((root / "full/logs").glob("events.out.tfevents.*")))
            incompatible = deepcopy(config)
            incompatible["lora"]["alpha"] = 4
            other = deepcopy(base)
            inject_lora(other, incompatible["lora"])
            with self.assertRaises(ValueError):
                train_model(other, incompatible, dataset, root / "bad", root / "full/step-0000002", "cpu", False)

    def test_images_captions_and_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            Image.new("RGB", (80, 40), "white").save(path / "sample.png")
            with self.assertRaises(ValueError):
                discover_pairs(path)
            (path / "sample.txt").write_text("테스트 캡션", encoding="utf-8")
            self.assertEqual(discover_pairs(path)[0]["caption"], "테스트 캡션")
            tensor = read_image(path / "sample.png")
            self.assertEqual(tensor.shape, (3, 1, 40, 80))
            self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_notebook_has_no_saved_outputs_and_code_compiles(self):
        notebook = json.loads((ROOT / "anima_lora_train_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")

    def test_freefit_size_upscales_and_preserves_head(self):
        image = Image.new("RGB", (64, 128), "white")
        image.paste("red", (0, 0, 64, 16))
        processed = resize_image(image, 262144)
        self.assertEqual(processed.size, choose_image_size(64, 128, 262144))
        self.assertEqual(processed.getpixel((processed.width // 2, 2)), (255, 0, 0))
        self.assertGreater(processed.width, 64)
        for width, height in ((1176, 2160), (2428, 1368), (684, 743), (64, 32)):
            target = choose_image_size(width, height, 262144)
            self.assertTrue(all(side % 16 == 0 for side in target))

    def test_preprocessed_files_reuse_and_caption_changes(self):
        config = make_config()
        config["dataset"]["max_pixels"] = 262144
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original"
            source.mkdir()
            Image.new("RGB", (64, 128), "white").save(source / "image.png")
            caption = source / "image.txt"
            caption.write_text("first", encoding="utf-8")
            before = (source / "image.png").read_bytes()
            result = preprocess_images(config, source, root / "processed")
            with Image.open(result / "0000000.png") as image:
                self.assertEqual(image.size, choose_image_size(64, 128, 262144))
            self.assertEqual((result / "0000000.txt").read_text(), "first")
            self.assertEqual((source / "image.png").read_bytes(), before)
            image_time = (result / "0000000.png").stat().st_mtime_ns
            self.assertEqual(preprocess_images(config, source, root / "processed"), result)
            self.assertEqual((result / "0000000.png").stat().st_mtime_ns, image_time)
            caption.write_text("second", encoding="utf-8")
            changed = preprocess_images(config, source, root / "processed")
            self.assertNotEqual(changed, result)
            self.assertEqual((changed / "0000000.txt").read_text(), "second")

    def test_global_stream_covers_all_samples_and_resumes(self):
        class Dataset:
            bucket_keys = [(32, 64), (64, 32), (32, 64), (80, 16), (16, 80)]

            def __len__(self):
                return len(self.bucket_keys)

            def __getitem__(self, index):
                return index

        dataset = Dataset()
        stream = SampleStream(dataset, repeat=3, seed=42, batch_size=4)
        observed = []
        batches = []
        for _ in range(4):
            batch = stream.take(4)
            self.assertEqual(len(batch), 4)
            batches.append(batch)
            observed.extend(batch)
        self.assertEqual(sorted(observed[:15]), sorted(list(range(5)) * 3))
        self.assertTrue(any(len({dataset.bucket_keys[index] for index in batch}) > 1 for batch in batches))
        self.assertEqual(stream.state_dict(), {"epoch": 1, "cursor": 1})
        restored = SampleStream(dataset, repeat=3, seed=42, batch_size=4, **stream.state_dict())
        for _ in range(10):
            self.assertEqual(stream.take(4), restored.take(4))

    def test_training_accepts_mixed_image_sizes_and_resumes(self):
        config = make_config()
        base = TinyDiT()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root / "cache")
            tensors = dataset[1]
            tensors["latent"] = torch.ones(4, 1, 2, 4)
            save_file(tensors, str(dataset.cache_path / dataset.records[1]["file"]))
            dataset = CachedDataset(dataset.cache_path)
            full = deepcopy(base)
            inject_lora(full, config["lora"])
            train_model(full, config, dataset, root / "full", device="cpu", tensorboard=False)
            resumed = deepcopy(base)
            inject_lora(resumed, config["lora"])
            train_model(resumed, config, dataset, root / "resumed",
                        resume_from=root / "full/step-0000002", device="cpu", tensorboard=False)
            expected = load_file(str(root / "full/step-0000004/lora.safetensors"))
            actual = load_file(str(root / "resumed/step-0000004/lora.safetensors"))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
