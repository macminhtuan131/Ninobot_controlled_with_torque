"""Prevent silent PPO/reward changes when continuing a checkpoint."""
from copy import deepcopy


def training_contract(config):
    return {"revision": 21, **{key: deepcopy(value) for key, value in config.items()
            if key not in ("device", "seed", "curriculum", "terrain_curriculum", "reward")}}


def validate_resume(model, config):
    expected = training_contract(config)
    saved = deepcopy(getattr(model, "nino_training_contract", None))
    # Revisions 10/11 replace the exact-boundary BEST_EFFORT LiDAR barrier
    # with delivery freshness plus bounded timestamp-lag compensation.
    # Observation/action shapes, PPO state and the reward formula are
    # unchanged, so revisions 9-11 are safe to continue. Revisions 12-21
    # intentionally fine-tune the same policy on a looser arrival contract,
    # completion-scaled timeout, and non-fatal asynchronous LiDAR handling.
    # Migrate only these known deltas; all other changes remain strict.
    if isinstance(saved, dict) and saved.get("revision") in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
        saved["revision"] = 21
        saved.pop("goal_overshoot_tolerance_m", None)
        expected_policy = expected.get("policy_v2", {})
        saved_policy = saved.get("policy_v2")
        if (
            isinstance(saved_policy, dict)
            and "lidar_max_lag_seconds" not in saved_policy
            and expected_policy.get("lidar_max_lag_seconds") == 0.25
        ):
            saved_policy["lidar_max_lag_seconds"] = 0.25
        if (
            isinstance(saved_policy, dict)
            and "lockstep_min_completion_fraction" not in saved_policy
            and expected_policy.get("lockstep_min_completion_fraction") == 0.80
        ):
            saved_policy["lockstep_min_completion_fraction"] = 0.80
        for key, old_values, new in (
            ("goal_tolerance_m", (0.01, 0.15), 0.003),
            ("goal_max_speed_m_s", (0.05,), 0.10),
        ):
            if saved.get(key) in old_values and expected.get(key) == new:
                saved[key] = new
        for key, value in (
            ("goal_capture_on_crossing", True),
            ("goal_require_stopped", False),
        ):
            if key not in saved and expected.get(key) is value:
                saved[key] = value
        saved_reward = saved.get("reward_v2")
        expected_reward = expected.get("reward_v2", {})
        if (
            isinstance(saved_reward, dict)
            and "timeout_completion_scaling" not in saved_reward
            and expected_reward.get("timeout_completion_scaling") is True
        ):
            saved_reward["timeout_completion_scaling"] = True
        if isinstance(saved_reward, dict):
            for key, value in (
                ("success_position_penalty", 30.0),
                ("success_position_sigma_m", 0.25),
                ("success_heading_penalty", 20.0),
                ("success_heading_sigma_rad", 0.20944),
            ):
                if key not in saved_reward and expected_reward.get(key) == value:
                    saved_reward[key] = value
            for key, old, new in (
                ("lateral_weight", 0.8, 4.0),
                ("heading_weight", 0.2, 2.0),
            ):
                if saved_reward.get(key) == old and expected_reward.get(key) == new:
                    saved_reward[key] = new
            for key, value in (
                ("progress_lateral_sigma_m", 0.25),
                ("progress_heading_sigma_rad", 0.35),
                ("lateral_sigma_m", 0.10),
                ("heading_sigma_rad", 0.174533),
            ):
                if key not in saved_reward and expected_reward.get(key) == value:
                    saved_reward[key] = value
    if saved != expected:
        raise ValueError(
            "Checkpoint reward/policy/PPO contract differs or predates this patch. "
            "Start a NEW run without --resume, or use the matching run's --config. "
            "Curriculum --phase changes are allowed; old v2 models still support inference.")
    model.nino_training_contract = expected
