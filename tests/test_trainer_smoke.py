import os
import sys
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import torch

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from core.config_schema import TrainingConfig
from engine.dataset import create_dataloader, get_dataset_diagnostics
from engine.networks import build_metadata, _extract_standard_state_dict, save_checkpoint, load_checkpoint


class TestTrainerSmoke(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.tmpdir.name) / 'dataset'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        for i in range(8):
            img = Image.new('RGB', (512 + (i % 2) * 128, 512 + ((i + 1) % 2) * 128), color=(i * 30, i * 20, i * 10))
            img_path = self.dataset_dir / f'img_{i:02d}.png'
            img.save(img_path)
            cap_path = self.dataset_dir / f'img_{i:02d}.txt'
            with open(cap_path, 'w', encoding='utf-8') as f:
                f.write(f'photo of subject_{i}, 8k highly detailed')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dataset_and_diagnostics_step_accounting(self):
        dataset, dataloader = create_dataloader(
            dataset_dir=str(self.dataset_dir),
            batch_size=2,
            max_resolution=512,
            repeats=2,
        )
        self.assertEqual(len(dataset), 16)

        diag1 = get_dataset_diagnostics(
            dataset=dataset,
            batch_size=2,
            grad_accum_steps=1,
            epochs=5,
            max_train_steps=0,
        )
        self.assertEqual(diag1['batches_per_epoch'], 8)
        self.assertEqual(diag1['optimizer_steps_per_epoch'], 8)
        self.assertEqual(diag1['total_expected_optimizer_steps'], 40)

        diag2 = get_dataset_diagnostics(
            dataset=dataset,
            batch_size=2,
            grad_accum_steps=2,
            epochs=5,
            max_train_steps=0,
        )
        self.assertEqual(diag2['batches_per_epoch'], 8)
        self.assertEqual(diag2['optimizer_steps_per_epoch'], 4)
        self.assertEqual(diag2['total_expected_optimizer_steps'], 20)

        diag4 = get_dataset_diagnostics(
            dataset=dataset,
            batch_size=2,
            grad_accum_steps=4,
            epochs=5,
            max_train_steps=0,
        )
        self.assertEqual(diag4['batches_per_epoch'], 8)
        self.assertEqual(diag4['optimizer_steps_per_epoch'], 2)
        self.assertEqual(diag4['total_expected_optimizer_steps'], 10)

        diag4_cap = get_dataset_diagnostics(
            dataset=dataset,
            batch_size=2,
            grad_accum_steps=4,
            epochs=5,
            max_train_steps=6,
        )
        self.assertEqual(diag4_cap['total_expected_optimizer_steps'], 6)

    def test_checkpoint_metadata_and_state_consistency(self):
        config = {
            'project_name': 'test_lora',
            'learning_rate': 1e-4,
            'max_train_steps': 100,
            'grad_accum_steps': 4,
            'network_type': 'lora',
            'rank': 16,
            'alpha': 8.0,
        }
        meta = build_metadata(config=config, global_step=50, epoch=2)
        self.assertEqual(meta['ss_steps'], '50')
        self.assertEqual(meta['ss_epoch'], '2')
        self.assertEqual(meta['ss_max_train_steps'], '100')
        self.assertEqual(meta['ss_gradient_accumulation_steps'], '4')


if __name__ == '__main__':
    unittest.main()
