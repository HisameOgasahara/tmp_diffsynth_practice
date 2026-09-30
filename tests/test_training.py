"""일반 LoRA의 gradient·생성 호환·학습 재개를 작은 CPU 모델로 확인합니다."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys
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
from anima_core.training.dataset import CachedDataset, discover_pairs, prepare_cache, read_image
from anima_core.training.flow_matching import FlowMatchingLoss
from anima_core.training.lora import collect_lora_weights, find_target_modules, inject_lora
from anima_core.training.trainer import train_model


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

    def forward(self, x, timesteps, context, t5xxl_ids=None, use_gradient_checkpointing=False):
        value = x.movedim(1, -1) + context.mean() + timesteps.reshape(-1, 1, 1, 1, 1)
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
                   "prompt_embeds": torch.randn(5, 4, generator=generator),
                   "t5_ids": torch.arange(index + 2)}, str(path / filename))
        records.append({"file": filename})
    save_file({"prompt_embeds": torch.zeros(5, 4), "t5_ids": torch.zeros(1, dtype=torch.long)},
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

    def test_flow_schedule_matches_reference(self):
        loss = FlowMatchingLoss()
        sigma = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigma = 3 * sigma / (1 + 2 * sigma)
        weights = torch.exp(-2 * ((sigma * 1000 - 500) / 1000) ** 2)
        weights -= weights.min()
        weights *= 1000 / weights.sum()
        torch.testing.assert_close(loss.sigmas, sigma)
        torch.testing.assert_close(loss.weights, weights)

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
        loss = FlowMatchingLoss()(model, torch.randn(1, 4, 1, 4, 4), torch.randn(1, 5, 16), None, True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_cache_preparation_and_reuse(self):
        class FakeVAE:
            def encode(self, image, device):
                return torch.zeros(1, 16, 1, image.shape[-2] // 8, image.shape[-1] // 8)

        config = make_config()
        config["dataset"]["resolution"] = 32
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
                 patch("anima_core.runtime.encode_prompt", return_value=(torch.zeros(1, 5, 4), torch.ones(1, 2, dtype=torch.long))):
                cache = prepare_cache(config, data, root / "cache", weights, "cpu")
                dataset = CachedDataset(cache)
                self.assertEqual(dataset[0]["latent"].shape, (16, 1, 4, 4))
                self.assertEqual(prepare_cache(config, data, root / "cache", weights, "cpu"), cache)
                self.assertEqual(vae_loader.call_count, 1)

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
            tensor = read_image(path / "sample.png", 32)
            self.assertEqual(tensor.shape, (3, 1, 32, 32))
            self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_notebook_has_no_saved_outputs_and_code_compiles(self):
        notebook = json.loads((ROOT / "anima_lora_train_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")


if __name__ == "__main__":
    unittest.main()
