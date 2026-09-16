"""Small CPU-friendly history policy with the existing 300-in/3-out contract."""
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from nino_rl.control_v2 import FRAME_SIZE


class HistoryFeatures(BaseFeaturesExtractor):
    """Current state, latest difference, and learned ordered history features.

    All frames are past/current, oldest first. There is no persistent hidden
    state to accidentally leak across episode resets or deployment restarts.
    The difference is not a calibrated derivative when sensor timing varies.
    """
    def __init__(self, observation_space, history_features=64):
        size = observation_space.shape[0]
        if len(observation_space.shape) != 1 or size % FRAME_SIZE or size < FRAME_SIZE:
            raise ValueError("HistoryFeatures needs stacked 60-value frames")
        self.frames = size // FRAME_SIZE
        super().__init__(observation_space, 2 * FRAME_SIZE + history_features)
        self.temporal = nn.Sequential(
            nn.Conv1d(FRAME_SIZE, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * self.frames, history_features), nn.ELU(),
        )

    def forward(self, observations):
        frames = observations.reshape(-1, self.frames, FRAME_SIZE)
        latest = frames[:, -1]
        previous = frames[:, -2] if self.frames > 1 else latest
        return th.cat((latest, latest - previous,
                       self.temporal(frames.transpose(1, 2))), dim=1)


class HistoryActorCriticPolicy(ActorCriticPolicy):
    """SB3 Gaussian policy, initialized near a moving zero-residual baseline.

    Keep SB3's rollout log-probabilities and action clipping intact. Merely
    adding tanh to sampled actions without correcting their densities would
    invalidate PPO probability ratios. This is not a squashed-Gaussian policy.
    """
    def __init__(self, *args, initial_speed_scale=0.70, **kwargs):
        if not 0.0 < initial_speed_scale < 1.0:
            raise ValueError("initial_speed_scale must be strictly between 0 and 1")
        self.initial_speed_scale = initial_speed_scale
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.action_space.shape != (3,) or self.use_sde:
            raise ValueError("History policy requires three actions and standard Gaussian PPO")
        with th.no_grad():
            self.action_net.bias.zero_()
            self.action_net.bias[0] = 2.0 * self.initial_speed_scale - 1.0

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["initial_speed_scale"] = self.initial_speed_scale
        return data


def policy_spec(config):
    ppo = config["ppo"]
    activation = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[ppo["activation"].lower()]
    return HistoryActorCriticPolicy, {
        "features_extractor_class": HistoryFeatures,
        "features_extractor_kwargs": {"history_features": int(ppo["history_features"])},
        "share_features_extractor": False,
        "activation_fn": activation,
        "net_arch": {"pi": list(ppo["actor_layers"]), "vf": list(ppo["critic_layers"])},
        "log_std_init": float(ppo["log_std_init"]),
        "initial_speed_scale": float(ppo["initial_speed_scale"]),
    }
