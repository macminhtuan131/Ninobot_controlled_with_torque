"""Deterministic PPO evaluation, with automatic per-episode trajectory scoring."""
from nino_rl.evaluation import run


def main():
    run(baseline=False)


if __name__ == "__main__":
    main()
