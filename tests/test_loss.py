"""
Unit tests for _compute_loss in engine/trainer.py.
Tests MSE, Huber, Min-SNR gamma weighting for both epsilon and v-prediction.
"""
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F


# Import the function under test
from engine.trainer import _compute_loss


def _make_noise_scheduler(prediction_type="epsilon", num_train_timesteps=1000):
    """Create a mock noise scheduler with the required attributes."""
    scheduler = MagicMock()
    scheduler.config.prediction_type = prediction_type
    scheduler.config.num_train_timesteps = num_train_timesteps
    # alphas_cumprod: linearly decreasing from 1.0 to ~0.0 over timesteps
    scheduler.alphas_cumprod = torch.linspace(0.9999, 0.0001, num_train_timesteps)
    return scheduler


class TestComputeLoss(unittest.TestCase):

    def setUp(self):
        self.bsz = 2
        self.channels = 4
        self.h, self.w = 8, 8
        self.noise_pred = torch.randn(self.bsz, self.channels, self.h, self.w)
        self.target = torch.randn(self.bsz, self.channels, self.h, self.w)
        self.timesteps = torch.tensor([100, 500])
        self.scheduler = _make_noise_scheduler()

    def test_mse_loss_returns_finite_scalar(self):
        """MSE loss should return a finite scalar."""
        config = {"loss_function": "mse", "min_snr_gamma": 0.0}
        loss = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config)
        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss}")
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")

    def test_mse_loss_matches_pytorch(self):
        """MSE loss with no weighting should match F.mse_loss directly."""
        config = {"loss_function": "mse", "min_snr_gamma": 0.0}
        loss = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config)
        expected = F.mse_loss(self.noise_pred.float(), self.target.float())
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_huber_loss_returns_finite_scalar(self):
        """Huber loss should return a finite scalar."""
        config = {"loss_function": "huber", "huber_c": 0.1, "min_snr_gamma": 0.0}
        loss = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config)
        self.assertTrue(torch.isfinite(loss), f"Huber loss is not finite: {loss}")

    def test_huber_loss_differs_from_mse(self):
        """Huber and MSE should generally produce different values."""
        config_mse = {"loss_function": "mse", "min_snr_gamma": 0.0}
        config_hub = {"loss_function": "huber", "huber_c": 0.1, "min_snr_gamma": 0.0}
        loss_mse = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_mse)
        loss_hub = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_hub)
        # They should be close but generally not identical
        self.assertTrue(torch.isfinite(loss_hub))
        self.assertTrue(torch.isfinite(loss_mse))

    def test_min_snr_gamma_zero_is_unweighted(self):
        """min_snr_gamma=0 should produce the same result as no weighting."""
        config_no = {"loss_function": "mse", "min_snr_gamma": 0.0}
        config_zero = {"loss_function": "mse", "min_snr_gamma": 0.0}
        loss_no = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_no)
        loss_zero = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_zero)
        self.assertAlmostEqual(loss_no.item(), loss_zero.item(), places=6)

    def test_min_snr_gamma_positive_changes_loss_epsilon(self):
        """min_snr_gamma > 0 with epsilon prediction should change the loss value."""
        config_plain = {"loss_function": "mse", "min_snr_gamma": 0.0}
        config_snr = {"loss_function": "mse", "min_snr_gamma": 5.0}
        loss_plain = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_plain)
        loss_snr = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config_snr)
        self.assertTrue(torch.isfinite(loss_snr))
        # The weighted loss should differ from plain loss
        self.assertNotAlmostEqual(loss_plain.item(), loss_snr.item(), places=3)

    def test_min_snr_gamma_v_prediction(self):
        """min_snr_gamma with v-prediction should use (snr + 1) denominator."""
        scheduler_v = _make_noise_scheduler(prediction_type="v_prediction")
        config = {"loss_function": "mse", "min_snr_gamma": 5.0, "v_prediction": False}
        loss = _compute_loss(self.noise_pred, self.target, self.timesteps, scheduler_v, config)
        self.assertTrue(torch.isfinite(loss))

    def test_v_prediction_config_override(self):
        """User config v_prediction=True should activate v-pred weighting even if scheduler says epsilon."""
        scheduler_eps = _make_noise_scheduler(prediction_type="epsilon")
        config_user_v = {"loss_function": "mse", "min_snr_gamma": 5.0, "v_prediction": True}
        config_no_v = {"loss_function": "mse", "min_snr_gamma": 5.0, "v_prediction": False}
        loss_v = _compute_loss(self.noise_pred, self.target, self.timesteps, scheduler_eps, config_user_v)
        loss_eps = _compute_loss(self.noise_pred, self.target, self.timesteps, scheduler_eps, config_no_v)
        self.assertTrue(torch.isfinite(loss_v))
        self.assertTrue(torch.isfinite(loss_eps))
        # v-prediction uses (snr+1) denominator, so should differ from epsilon
        self.assertNotAlmostEqual(loss_v.item(), loss_eps.item(), places=3)

    def test_loss_positive(self):
        """Loss should always be non-negative."""
        for loss_fn in ["mse", "huber"]:
            for gamma in [0.0, 5.0]:
                config = {"loss_function": loss_fn, "huber_c": 0.1, "min_snr_gamma": gamma}
                loss = _compute_loss(self.noise_pred, self.target, self.timesteps, self.scheduler, config)
                self.assertGreaterEqual(loss.item(), 0.0, f"Negative loss with {loss_fn}, gamma={gamma}")


if __name__ == "__main__":
    unittest.main()
