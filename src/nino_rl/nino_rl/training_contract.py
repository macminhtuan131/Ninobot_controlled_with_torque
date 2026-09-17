"""Prevent silent PPO/reward changes when continuing a checkpoint."""
from copy import deepcopy


def training_contract(config):
    return {"revision": 4, **{key: deepcopy(value) for key, value in config.items()
            if key not in ("device", "seed", "curriculum", "terrain_curriculum", "reward")}}


def validate_resume(model, config):
    if getattr(model, "nino_training_contract", None) != training_contract(config):
        raise ValueError(
            "Checkpoint reward/policy/PPO contract differs or predates this patch. "
            "Start a NEW run without --resume, or use the matching run's --config. "
            "Curriculum --phase changes are allowed; old v2 models still support inference.")
