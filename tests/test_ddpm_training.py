import unittest
import torch
from diffusers import DDPMScheduler, EulerDiscreteScheduler
from engine.trainer import _ddpm_add_noise, _ddpm_get_velocity
from engine.networks import build_metadata


class TestDDPMTraining(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
        }
        self.ddpm = DDPMScheduler.from_config(self.cfg)
        self.latents = torch.randn(2, 4, 32, 32, dtype=torch.float32)
        self.noise = torch.randn(2, 4, 32, 32, dtype=torch.float32)
        self.timesteps = torch.tensor([100, 750], dtype=torch.long)

    def test_ddpm_add_noise_matches_ddpm_scheduler(self):
        """Verify _ddpm_add_noise matches DDPMScheduler.add_noise to machine precision."""
        expected = self.ddpm.add_noise(self.latents, self.noise, self.timesteps)
        actual = _ddpm_add_noise(self.latents, self.noise, self.timesteps, self.ddpm.alphas_cumprod)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_ddpm_get_velocity_matches_ddpm_scheduler(self):
        """Verify _ddpm_get_velocity matches DDPMScheduler.get_velocity to machine precision."""
        expected = self.ddpm.get_velocity(self.latents, self.noise, self.timesteps)
        actual = _ddpm_get_velocity(self.latents, self.noise, self.timesteps, self.ddpm.alphas_cumprod)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_euler_scheduler_add_noise_is_not_ddpm(self):
        """Demonstrate that EulerDiscreteScheduler.add_noise does NOT attenuate clean signal."""
        euler = EulerDiscreteScheduler.from_config(self.cfg)
        euler_noisy = euler.add_noise(self.latents, self.noise, self.timesteps)
        ddpm_noisy = _ddpm_add_noise(self.latents, self.noise, self.timesteps, self.ddpm.alphas_cumprod)
        self.assertFalse(torch.allclose(euler_noisy, ddpm_noisy, atol=1e-2))

    def test_metadata_sdxl_base_model_version(self):
        """Verify metadata sets ss_base_model_version=sdxl_base_v1-0 and v_prediction."""
        meta_sdxl = build_metadata(config={"model_type": "SDXL", "v_prediction": False})
        self.assertEqual(meta_sdxl["ss_base_model_version"], "sdxl_base_v1-0")
        self.assertEqual(meta_sdxl["ss_v_prediction"], "False")
        self.assertEqual(meta_sdxl["modelspec.prediction_type"], "epsilon")

        meta_v = build_metadata(config={"model_type": "SDXL", "v_prediction": True})
        self.assertEqual(meta_v["ss_v_prediction"], "True")
        self.assertEqual(meta_v["modelspec.prediction_type"], "v")


if __name__ == "__main__":
    unittest.main()
