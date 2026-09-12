import unittest
import torch
from unittest.mock import MagicMock

from core.config_schema import TrainingConfig
from engine.dataset import ARBDataset
from engine.trainer import _create_optimizer


class TestLeanFeatures(unittest.TestCase):
    def test_tag_dropout_and_keep_tokens(self):
        dataset = ARBDataset.__new__(ARBDataset)
        dataset.caption_dropout_rate = 0.0
        dataset.keep_tokens = 1
        dataset.tag_dropout_rate = 1.0  # Drop 100% of flexible tags
        dataset.shuffle_captions = False

        caption = "trigger_word, red hair, blue eyes, masterpiece"
        processed = dataset._process_caption(caption)
        self.assertEqual(processed, "trigger_word")

        # Test keep_tokens=2 with 100% tag dropout
        dataset.keep_tokens = 2
        processed = dataset._process_caption(caption)
        self.assertEqual(processed, "trigger_word, red hair")

        # Test zero dropout preserves all tokens
        dataset.tag_dropout_rate = 0.0
        processed = dataset._process_caption(caption)
        self.assertEqual(processed, caption)

    def test_block_weight_multipliers(self):
        class DummyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down_blocks = torch.nn.Linear(4, 4)
                self.mid_block = torch.nn.Linear(4, 4)
                self.up_blocks = torch.nn.Linear(4, 4)

        net = DummyNet()
        config = {
            "learning_rate": 1e-4,
            "text_encoder_lr": 5e-5,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "down_lr_weight": 0.5,
            "mid_lr_weight": 0.0,  # Freeze mid block
            "up_lr_weight": 1.5,
        }

        opt = _create_optimizer({"unet": net}, config)
        # mid_block should have requires_grad set to False
        for p in net.mid_block.parameters():
            self.assertFalse(p.requires_grad)

        # Down block should be scaled by 0.5, up block by 1.5
        self.assertEqual(len(opt.param_groups), 2)
        lrs = [pg["lr"] for pg in opt.param_groups]
        self.assertAlmostEqual(lrs[0], 5e-5)   # 1e-4 * 0.5
        self.assertAlmostEqual(lrs[1], 1.5e-4) # 1e-4 * 1.5

    def test_prodigy_optimizer_configuration(self):
        try:
            from prodigyopt import Prodigy
        except ImportError:
            self.skipTest("prodigyopt not installed")

        p1 = torch.nn.Parameter(torch.randn(4, 4))
        p2 = torch.nn.Parameter(torch.randn(4, 4))
        net = torch.nn.Module()
        net.register_parameter("p1", p1)
        net.register_parameter("p2", p2)

        config = {
            "learning_rate": 1e-4,
            "weight_decay": 0.02,
            "optimizer": "Prodigy",
            "prodigy_d_coef": 2.5,
            "prodigy_use_bias_correction": True,
            "prodigy_safeguard_warmup": True,
            "prodigy_decouple": True,
        }

        opt = _create_optimizer(net, config)
        self.assertIsInstance(opt, Prodigy)
        self.assertEqual(opt.param_groups[0]["d_coef"], 2.5)
        self.assertTrue(opt.param_groups[0]["use_bias_correction"])
        self.assertTrue(opt.param_groups[0]["safeguard_warmup"])
        self.assertTrue(opt.param_groups[0]["decouple"])

    def test_config_defaults(self):
        cfg = TrainingConfig()
        self.assertEqual(cfg.tag_dropout_rate, 0.0)
        self.assertEqual(cfg.down_lr_weight, 1.0)
        self.assertEqual(cfg.mid_lr_weight, 1.0)
        self.assertEqual(cfg.up_lr_weight, 1.0)
        self.assertEqual(cfg.prodigy_d_coef, 1.0)
        self.assertFalse(cfg.prodigy_use_bias_correction)

    def test_sidebar_sections_and_views_alignment(self):
        from ui.sidebar import Sidebar
        self.assertEqual(len(Sidebar.SECTIONS), 5)
        labels = [sec[0] for sec in Sidebar.SECTIONS]
        self.assertEqual(labels, ["MODEL & DATA", "ARCHITECTURE", "TRAINING & HW", "TENSORBOARD", "OUTPUT & LOGS"])
