"""Straight-line PI baseline with the exact same scoring pipeline as PPO."""
from nino_rl.evaluation import run


def main():
    run(baseline=True)


if __name__ == "__main__":
    main()
