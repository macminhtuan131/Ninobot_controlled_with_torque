"""Behavioral reward tests and an actual SB3 train/save/load smoke test."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from nino_rl.core import RobotState, TrackingState, NavReference
from nino_rl.control_v2 import BASELINE_ACTION, FRAME_SIZE, compute_reward
from nino_rl.training_contract import training_contract, validate_resume

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/ppo.yaml").read_text())


class RewardTests(unittest.TestCase):
    def terms(self, state=None, current=None, reference=None, **kwargs):
        previous = TrackingState(0, 0, 0, 30, 30)
        cfg = {**CONFIG["reward_v2"], "torque_scale_nm": .5}
        _, terms = compute_reward(previous, current or previous, state or RobotState(),
            BASELINE_ACTION, BASELINE_ACTION, [0., 0.], .1,
            {"impact_integral": 0.}, cfg, reference=reference, **kwargs)
        return terms

    def test_matching_command_beats_stopping_and_overspeed_is_costlier(self):
        reference = NavReference(desired_linear_velocity=.4, valid=True)
        costs = [self.terms(RobotState(ground_linear_velocity=v), reference=reference)["velocity"]
                 for v in [0., .2, .4, .6]]
        self.assertEqual(costs[2], 0.)
        self.assertLess(costs[0], costs[1])
        self.assertLess(costs[3], costs[1])
        # Backward references get the same along-command asymmetry.
        reverse = NavReference(desired_linear_velocity=-.4, valid=True)
        self.assertAlmostEqual(costs[3], self.terms(
            RobotState(ground_linear_velocity=-.6), reference=reverse)["velocity"])

    def test_truth_velocity_prevents_spinning_encoders_from_hiding_error(self):
        ref = NavReference(desired_linear_velocity=.4, valid=True)
        stopped = self.terms(RobotState(), reference=ref)["velocity"]
        spinning = self.terms(RobotState(linear_velocity=.4, left_wheel_velocity=6.4,
            right_wheel_velocity=6.4), reference=ref)["velocity"]
        self.assertEqual(stopped, spinning)
        self.assertLess(stopped, 0.)

    def test_turn_in_place_and_zero_reference(self):
        ref = NavReference(desired_angular_velocity=.5, valid=True)
        matched = self.terms(RobotState(ground_yaw_rate=.5), reference=ref)
        stopped = self.terms(reference=ref)
        self.assertEqual(matched["yaw_tracking"], 0.)
        self.assertLess(stopped["yaw_tracking"], 0.)
        zero = NavReference(valid=True)
        self.assertLess(self.terms(RobotState(ground_linear_velocity=.2),
                                  reference=zero)["velocity"], 0.)
        invalid = NavReference(desired_linear_velocity=.5, valid=False)
        self.assertEqual(self.terms(reference=invalid)["velocity"], 0.)

    def test_pi_torque_is_penalized_even_with_zero_residual(self):
        state = RobotState(applied_left_torque=2., applied_right_torque=2.)
        terms = self.terms(state, previous_state=RobotState())
        self.assertLess(terms["effort"], 0.)
        self.assertLess(terms["torque_rate"], 0.)
        self.assertEqual(terms["residual_effort"], 0.)
        self.assertEqual(self.terms(state, previous_state=state)["torque_rate"], 0.)

    def test_braking_cost_localized_at_endpoint(self):
        moving = RobotState(ground_linear_velocity=.4, ground_yaw_rate=.4)
        near = TrackingState(29.7, 0, 0, .3, .3)
        self.assertLess(self.terms(moving, near)["goal_braking"], -.1)
        self.assertAlmostEqual(self.terms(moving)["goal_braking"], 0.)
        self.assertEqual(self.terms(RobotState(), near)["goal_braking"], 0.)

    def test_unequal_step_sizes_cannot_farm_undiscounted_progress(self):
        # Old clipped progress gave +2 for two +0.1 steps and one -0.2 step.
        remaining = [30., 29.9, 29.8, 30.]
        total = 0.
        for before, after in zip(remaining, remaining[1:]):
            _, terms = compute_reward(TrackingState(30-before, 0, 0, before, before),
                TrackingState(30-after, 0, 0, after, after), RobotState(),
                BASELINE_ACTION, BASELINE_ACTION, [0., 0.], .1,
                {"impact_integral": 0.}, {**CONFIG["reward_v2"], "torque_scale_nm": .5})
            total += terms["progress"]
        self.assertAlmostEqual(total, 0.)

    def test_rate_cost_matches_linear_ramp_when_step_is_subdivided(self):
        cfg = {**CONFIG["reward_v2"], "torque_scale_nm": .5}
        tracking = TrackingState(0, 0, 0, 30, 30)
        def ramp(before, after, dt):
            _, terms = compute_reward(tracking, tracking,
                RobotState(applied_left_torque=after, applied_right_torque=after),
                [after, 0., 0.], [before, 0., 0.], [0., 0.], dt,
                {"impact_integral": 0.}, cfg,
                previous_state=RobotState(applied_left_torque=before, applied_right_torque=before))
            return terms["torque_rate"] + terms["smoothness"]
        self.assertAlmostEqual(ramp(0, 1, .1), ramp(0, .5, .05) + ramp(.5, 1, .05))

    def test_large_sensor_values_keep_new_tracking_penalties_bounded(self):
        terms = self.terms(RobotState(ground_linear_velocity=1e6, ground_yaw_rate=1e6),
                           reference=NavReference(valid=True))
        self.assertGreaterEqual(terms["velocity"], -CONFIG["reward_v2"]["overspeed_weight"])
        self.assertGreaterEqual(terms["yaw_tracking"], -CONFIG["reward_v2"]["yaw_tracking_weight"])
        self.assertTrue(all(np.isfinite(list(terms.values()))))

    def test_resume_accepts_phase_change_but_rejects_silent_reward_changes(self):
        model = SimpleNamespace(nino_training_contract=training_contract(CONFIG))
        changed = deepcopy(CONFIG)
        changed["curriculum"]["fixed_phase"] = 2
        validate_resume(model, changed)
        changed["reward_v2"]["velocity_weight"] *= 2
        with self.assertRaisesRegex(ValueError, "NEW run"):
            validate_resume(model, changed)
        with self.assertRaisesRegex(ValueError, "NEW run"):
            validate_resume(SimpleNamespace(), CONFIG)


try:
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import PPO
    from nino_rl.policies import HistoryFeatures, policy_spec
    HAS_RL = True
except ImportError:
    HAS_RL = False


@unittest.skipUnless(HAS_RL, "Install torch, gymnasium and stable-baselines3")
class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_history_order_is_visible_without_future_or_hidden_state(self):
        th.manual_seed(42)
        extractor = HistoryFeatures(gym.spaces.Box(-5., 5., shape=(300,), dtype=np.float32))
        frames = th.arange(5, dtype=th.float32).reshape(1, 5, 1).repeat(1, 1, FRAME_SIZE)
        changed = frames.clone()
        changed[:, 0], changed[:, 1] = frames[:, 1].clone(), frames[:, 0].clone()
        a, b = extractor(frames.flatten(1)), extractor(changed.flatten(1))
        th.testing.assert_close(a[:, :120], b[:, :120])
        self.assertFalse(th.allclose(a[:, 120:], b[:, 120:]))
        a.sum().backward()
        self.assertTrue(all(th.isfinite(p.grad).all() for p in extractor.parameters()))

    def test_real_ppo_update_save_load_and_resume(self):
        class ContractEnv(gym.Env):
            # Transport/API smoke only, deliberately not a robot simulator.
            observation_space = gym.spaces.Box(-5., 5., shape=(300,), dtype=np.float32)
            action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return np.zeros(300, np.float32), {}

            def step(self, action):
                self.steps += 1
                obs = np.zeros((5, 60), np.float32)
                obs[:, 52:55] = action
                return obs.reshape(300), float(-np.square(action).sum()), self.steps == 8, False, {}

        config = deepcopy(CONFIG)
        config["ppo"].update(n_steps=32, batch_size=16, n_epochs=2)
        cls, kwargs = policy_spec(config)
        model = PPO(cls, ContractEnv(), policy_kwargs=kwargs, n_steps=32, batch_size=16,
                    n_epochs=2, device="cpu", seed=42, target_kl=.015)
        observation = np.zeros(300, np.float32)
        action, _ = model.predict(observation, deterministic=True)
        # Small random head weights add a small state-dependent offset.
        np.testing.assert_allclose(action, [.4, 0., 0.], atol=.01)
        np.testing.assert_allclose(model.policy.log_std.detach().numpy(),
                                   config["ppo"]["log_std_init"])
        self.assertIsNot(model.policy.pi_features_extractor, model.policy.vf_features_extractor)
        before = model.policy.action_net.weight.detach().clone()
        model.learn(64)
        self.assertFalse(th.equal(before, model.policy.action_net.weight))
        self.assertTrue(all(th.isfinite(p).all() for p in model.policy.parameters()))
        model.nino_training_contract = training_contract(config)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.zip"
            model.save(path)
            restored = PPO.load(path, device="cpu")
            validate_resume(restored, config)
            np.testing.assert_allclose(restored.predict(observation, deterministic=True)[0],
                                       model.predict(observation, deterministic=True)[0], atol=1e-7)
            restored.set_env(ContractEnv())
            restored.learn(32, reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps, 96)


if __name__ == "__main__":
    unittest.main()
