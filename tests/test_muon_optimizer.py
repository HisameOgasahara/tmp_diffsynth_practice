"""AdamW 경로 보존과 Muon 선택·설정·학습 재개를 확인합니다."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_training import ROOT, TinyDiT, make_config, make_dataset
from anima_core.training.config import load_config, validate_config, write_config
from anima_core.training.lora import inject_lora, collect_lora_weights
from anima_core.training.optimizers import create_optimizer
from anima_core.training.trainer import train_model


class MuonOptimizerTests(unittest.TestCase):
    def test_adamw_keeps_existing_options(self):
        training = make_config()["training"]
        parameter = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = create_optimizer([parameter], training)
        reference = torch.optim.AdamW([parameter], lr=training["learning_rate"],
                                      weight_decay=training["weight_decay"])
        self.assertEqual(optimizer.defaults, reference.defaults)

    def test_muon_routes_all_options(self):
        training = load_config(ROOT / "configs/anima_lora_muon.toml")["training"]
        parameters = [torch.nn.Parameter(torch.ones(2, 3)), torch.nn.Parameter(torch.zeros(3, 2))]
        with patch.object(torch.optim, "Muon", create=True) as constructor:
            create_optimizer(parameters, training)
            constructor.assert_called_once_with(
                parameters, lr=0.00005, weight_decay=0.01, momentum=0.95,
                nesterov=True, ns_steps=5, adjust_lr_fn="original")

    def test_muon_config_roundtrip_and_invalid_options(self):
        config = load_config(ROOT / "configs/anima_lora_muon.toml")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.toml"
            write_config(path, config)
            self.assertEqual(load_config(path), config)
        for key, value in (("momentum", 1), ("momentum", float("nan")),
                           ("nesterov", 1), ("ns_steps", 0),
                           ("adjust_lr_fn", "unknown")):
            with self.subTest(key=key, value=value):
                invalid = deepcopy(config)
                invalid["training"][key] = value
                with self.assertRaises(ValueError):
                    validate_config(invalid)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "현재 PyTorch에 Muon이 없습니다.")
    def test_native_muon_training_resume_matches_uninterrupted(self):
        torch.set_num_threads(1)
        config = make_config()
        muon = load_config(ROOT / "configs/anima_lora_muon.toml")["training"]
        config["training"].update({key: muon[key] for key in
                                  ("optimizer", "learning_rate", "momentum", "nesterov",
                                   "ns_steps", "adjust_lr_fn")})
        def make_model():
            torch.manual_seed(9)
            model = TinyDiT()
            inject_lora(model, config["lora"])
            return model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = make_dataset(root / "cache")
            complete = make_model()
            train_model(complete, config, dataset, root / "complete", device="cpu", tensorboard=False)
            resumed = make_model()
            train_model(resumed, config, dataset, root / "resumed",
                        resume_from=root / "complete/step-0000002", device="cpu", tensorboard=False)
            for key, value in collect_lora_weights(complete).items():
                torch.testing.assert_close(value, collect_lora_weights(resumed)[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
