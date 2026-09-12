"""
Unit tests for scheduler registry in engine/sampler.py.
"""
import unittest
from engine.sampler import SCHEDULER_REGISTRY, EulerDiscreteScheduler


class TestSamplerRegistry(unittest.TestCase):

    def setUp(self):
        self.default_scheduler_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "steps_offset": 1,
        }

    def test_all_registry_entries_instantiate(self):
        """Every sampler registered in SCHEDULER_REGISTRY should instantiate without error."""
        for name, factory in SCHEDULER_REGISTRY.items():
            scheduler = factory(self.default_scheduler_config)
            self.assertIsNotNone(scheduler, f"Scheduler '{name}' failed to instantiate")

    def test_unknown_sampler_falls_back_to_euler(self):
        """Unknown sampler names should fall back to Euler."""
        unknown_choice = "nonexistent_sampler"
        factory = SCHEDULER_REGISTRY.get(unknown_choice)
        if factory is None:
            factory = SCHEDULER_REGISTRY["euler"]
        scheduler = factory(self.default_scheduler_config)
        self.assertIsInstance(scheduler, EulerDiscreteScheduler)

    def test_default_config_produces_valid_scheduler(self):
        """The default euler scheduler with default config produces a valid scheduler instance."""
        factory = SCHEDULER_REGISTRY["euler"]
        scheduler = factory(self.default_scheduler_config)
        self.assertTrue(hasattr(scheduler, "step"))
        self.assertTrue(hasattr(scheduler, "set_timesteps"))


if __name__ == "__main__":
    unittest.main()
