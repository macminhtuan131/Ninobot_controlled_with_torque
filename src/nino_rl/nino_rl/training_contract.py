"""Prevent silent PPO/reward changes when continuing a checkpoint."""
from copy import deepcopy


def training_contract(config):
    return {"revision": 26,
            "phase_schedule": {key: deepcopy(config.get("curriculum", {}).get(key))
                               for key in ("enabled", "phase_steps", "phase_order")},
            **{key: deepcopy(value) for key, value in config.items()
            if key not in ("device", "seed", "curriculum", "terrain_curriculum", "reward")}}


def validate_resume(model, config):
    expected = training_contract(config)
    saved = deepcopy(getattr(model, "nino_training_contract", None))
    # Revision 26 records the absolute phase schedule and five-success gate.
    if saved != expected:
        raise ValueError(
            "Checkpoint reward/policy/PPO contract differs or predates this patch. "
            "Start a NEW run without --resume, or use the matching run's --config. "
            "Curriculum --phase changes are allowed; old v2 models still support inference.")
    model.nino_training_contract = expected
