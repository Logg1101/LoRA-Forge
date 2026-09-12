import math
import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from core.config_schema import TrainingConfig
from engine.dataset import get_dataset_diagnostics
from engine.networks import save_checkpoint, load_training_state


class TestStepAccounting(unittest.TestCase):
    def test_config_training_plan_calculations(self):
        # 100 images, 1 repeat, batch_size=2 -> 50 batches per epoch
        # grad_accum=1 -> 50 steps/epoch, 10 epochs -> 500 total steps
        cfg1 = TrainingConfig(
            batch_size=2,
            dataset_repeats=1,
            epochs=10,
            grad_accum_steps=1,
            max_train_steps=0,
        )
        plan1 = cfg1.training_plan(image_count=100)
        self.assertEqual(plan1['steps_per_epoch'], 50)
        self.assertEqual(plan1['total_steps'], 500)

        # grad_accum=2 -> 25 steps/epoch, 10 epochs -> 250 total steps
        cfg2 = TrainingConfig(
            batch_size=2,
            dataset_repeats=1,
            epochs=10,
            grad_accum_steps=2,
            max_train_steps=0,
        )
        plan2 = cfg2.training_plan(image_count=100)
        self.assertEqual(plan2['steps_per_epoch'], 25)
        self.assertEqual(plan2['total_steps'], 250)

        # grad_accum=4 -> ceil(50/4) = 13 steps/epoch, 10 epochs -> 130 total steps
        cfg4 = TrainingConfig(
            batch_size=2,
            dataset_repeats=1,
            epochs=10,
            grad_accum_steps=4,
            max_train_steps=0,
        )
        plan4 = cfg4.training_plan(image_count=100)
        self.assertEqual(plan4['steps_per_epoch'], 13)
        self.assertEqual(plan4['total_steps'], 130)

        # max_train_steps override
        cfg_capped = TrainingConfig(
            batch_size=2,
            dataset_repeats=1,
            epochs=10,
            grad_accum_steps=4,
            max_train_steps=100,
        )
        plan_capped = cfg_capped.training_plan(image_count=100)
        self.assertEqual(plan_capped['steps_per_epoch'], 13)
        self.assertEqual(plan_capped['total_steps'], 100)

    def _run_simulated_training_loop(self, grad_accum_steps: int, max_train_steps: int, epochs: int, num_batches: int = 50):
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision='no',
        )

        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dataset = TensorDataset(torch.randn(num_batches * 2, 10), torch.randn(num_batches * 2, 2))
        dataloader = DataLoader(dataset, batch_size=2)

        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        global_step = 0
        actual_optimizer_steps = 0
        batches_processed = 0

        orig_step = optimizer.step
        def tracked_step(*args, **kwargs):
            nonlocal actual_optimizer_steps
            if accelerator.sync_gradients:
                actual_optimizer_steps += 1
            return orig_step(*args, **kwargs)
        optimizer.step = tracked_step

        for epoch in range(epochs):
            for batch in dataloader:
                batches_processed += 1
                inputs, targets = batch

                with accelerator.accumulate(model):
                    outputs = model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if max_train_steps > 0 and global_step >= max_train_steps:
                        break

            if max_train_steps > 0 and global_step >= max_train_steps:
                break

        return {
            'global_step': global_step,
            'actual_optimizer_steps': actual_optimizer_steps,
            'batches_processed': batches_processed,
        }

    def test_accumulation_1_max_steps_100(self):
        res = self._run_simulated_training_loop(
            grad_accum_steps=1,
            max_train_steps=100,
            epochs=5,
            num_batches=50,
        )
        self.assertEqual(res['global_step'], 100)
        self.assertEqual(res['actual_optimizer_steps'], 100)
        self.assertEqual(res['batches_processed'], 100)

    def test_accumulation_2_max_steps_100(self):
        res = self._run_simulated_training_loop(
            grad_accum_steps=2,
            max_train_steps=100,
            epochs=5,
            num_batches=50,
        )
        self.assertEqual(res['global_step'], 100)
        self.assertEqual(res['actual_optimizer_steps'], 100)
        self.assertEqual(res['batches_processed'], 200)

    def test_accumulation_4_max_steps_100_clean_epochs(self):
        # 40 batches/epoch is divisible by 4. 10 optimizer steps/epoch * 10 epochs = 100 steps
        res = self._run_simulated_training_loop(
            grad_accum_steps=4,
            max_train_steps=100,
            epochs=10,
            num_batches=40,
        )
        self.assertEqual(res['global_step'], 100)
        self.assertEqual(res['actual_optimizer_steps'], 100)
        self.assertEqual(res['batches_processed'], 400)

    def test_accumulation_4_max_steps_100_with_epoch_remainder(self):
        # 50 batches/epoch. 12 full syncs (48 batches) + 1 dataloader-end sync (2 batches) = 13 steps/epoch
        # 7 epochs * 13 steps = 91 steps (350 batches)
        # Epoch 8 needs 9 steps (36 batches) -> 350 + 36 = 386 batches
        res = self._run_simulated_training_loop(
            grad_accum_steps=4,
            max_train_steps=100,
            epochs=10,
            num_batches=50,
        )
        self.assertEqual(res['global_step'], 100)
        self.assertEqual(res['actual_optimizer_steps'], 100)
        self.assertEqual(res['batches_processed'], 386)

    def test_epoch_based_training_step_counts(self):
        res = self._run_simulated_training_loop(
            grad_accum_steps=4,
            max_train_steps=0,
            epochs=2,
            num_batches=50,
        )
        self.assertEqual(res['batches_processed'], 100)
        self.assertEqual(res['global_step'], 26)
        self.assertEqual(res['actual_optimizer_steps'], 26)

    def test_indivisible_batch_13_accum_4(self):
        """13 batches per epoch with accumulation 4 (1 epoch -> 4 optimizer updates, 3 epochs -> 12 updates)."""
        # 1 epoch: 12 batches (3 syncs) + 1 remainder batch (1 sync) = 4 optimizer steps
        res1 = self._run_simulated_training_loop(
            grad_accum_steps=4,
            max_train_steps=0,
            epochs=1,
            num_batches=13,
        )
        self.assertEqual(res1['batches_processed'], 13)
        self.assertEqual(res1['global_step'], 4)
        self.assertEqual(res1['actual_optimizer_steps'], 4)

        # 3 epochs: 39 batches -> 12 optimizer steps
        res3 = self._run_simulated_training_loop(
            grad_accum_steps=4,
            max_train_steps=0,
            epochs=3,
            num_batches=13,
        )
        self.assertEqual(res3['batches_processed'], 39)
        self.assertEqual(res3['global_step'], 12)
        self.assertEqual(res3['actual_optimizer_steps'], 12)

    def test_indivisible_batch_5_accum_2(self):
        """5 batches per epoch with accumulation 2 (3 epochs -> 9 updates, max_steps=7 -> terminates at 7)."""
        # 3 epochs without max_steps: 3 steps/epoch * 3 epochs = 9 optimizer steps (15 batches)
        res_full = self._run_simulated_training_loop(
            grad_accum_steps=2,
            max_train_steps=0,
            epochs=3,
            num_batches=5,
        )
        self.assertEqual(res_full['batches_processed'], 15)
        self.assertEqual(res_full['global_step'], 9)
        self.assertEqual(res_full['actual_optimizer_steps'], 9)

        # 3 epochs with max_train_steps=7: Epoch 1 (3 steps) + Epoch 2 (3 steps) + Epoch 3 (1 step after 2 batches) = 7 steps (12 batches)
        res_capped = self._run_simulated_training_loop(
            grad_accum_steps=2,
            max_train_steps=7,
            epochs=3,
            num_batches=5,
        )
        self.assertEqual(res_capped['batches_processed'], 12)
        self.assertEqual(res_capped['global_step'], 7)
        self.assertEqual(res_capped['actual_optimizer_steps'], 7)

    def test_indivisible_batch_7_accum_3(self):
        """7 batches per epoch with accumulation 3 (2 epochs -> 6 updates, max_steps=10 -> terminates at 10)."""
        # 2 epochs: ceil(7/3)=3 steps/epoch * 2 epochs = 6 optimizer steps (14 batches)
        res_full = self._run_simulated_training_loop(
            grad_accum_steps=3,
            max_train_steps=0,
            epochs=2,
            num_batches=7,
        )
        self.assertEqual(res_full['batches_processed'], 14)
        self.assertEqual(res_full['global_step'], 6)
        self.assertEqual(res_full['actual_optimizer_steps'], 6)

        # 5 epochs with max_train_steps=10:
        # Epoch 1: 7 batches -> 3 steps (cum 3)
        # Epoch 2: 7 batches -> 3 steps (cum 6)
        # Epoch 3: 7 batches -> 3 steps (cum 9)
        # Epoch 4: 3 batches -> 1 step (cum 10 -> terminates!) Total batches = 7+7+7+3 = 24 batches.
        res_capped = self._run_simulated_training_loop(
            grad_accum_steps=3,
            max_train_steps=10,
            epochs=5,
            num_batches=7,
        )
        self.assertEqual(res_capped['batches_processed'], 24)
        self.assertEqual(res_capped['global_step'], 10)
        self.assertEqual(res_capped['actual_optimizer_steps'], 10)

    def test_checkpoint_step_restoration(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / 'test_step_100.safetensors'
            model = nn.Linear(4, 4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            save_checkpoint(
                network=model,
                save_path=ckpt_path,
                global_step=100,
                epoch=4,
                optimizer=optimizer,
                save_state=True,
            )

            self.assertTrue(ckpt_path.exists())
            state_file = ckpt_path.with_suffix('.state.pt')
            self.assertTrue(state_file.exists())

            new_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            restored = load_training_state(ckpt_path, optimizer=new_opt)
            self.assertEqual(restored.get('global_step'), 100)
            self.assertEqual(restored.get('epoch'), 4)


if __name__ == '__main__':
    unittest.main()
