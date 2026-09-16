"""Prevent silent PPO/reward changes when continuing a checkpoint."""
from copy import deepcopy


def training_contract(config):
    return {"revision": 3, **{key: deepcopy(config[key]) for key in (
        "control_hz", "max_episode_seconds", "max_wheel_torque_nm",
        "policy_v2", "reward_v2", "ppo")}}


def validate_resume(model, config):
    if getattr(model, "nino_training_contract", None) != training_contract(config):
        raise ValueError(
            "Checkpoint reward/policy/PPO contract differs or predates this patch. "
            "Start a NEW run without --resume, or use the matching run's --config. "
            "Curriculum --phase changes are allowed; old v2 models still support inference.")
