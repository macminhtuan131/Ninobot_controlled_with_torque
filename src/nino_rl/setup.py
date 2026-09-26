from glob import glob
import os

from setuptools import find_packages, setup


package_name = "nino_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            ["package.xml", "requirements.txt", "README.md", "README_VI.md", "ALGORITHM_V2.md",
             "REWARD_POLICY_UPDATE.md", "RL_IMPROVEMENTS.md"],
        ),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "models", "completed_train"),
         glob("models/completed_train/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nino Robot Maintainer",
    maintainer_email="maintainer@example.com",
    description="PPO wheel-torque policy for Nino rough-terrain path tracking.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "check_cuda = nino_rl.check_cuda:main",
            "evaluate = nino_rl.evaluate:main",
            "evaluate_baseline = nino_rl.evaluate_baseline:main",
            "trajectory_metrics = nino_rl.trajectory_metrics:main",
            "compare_evaluations = nino_rl.evaluation:compare_main",
            "policy_node = nino_rl.policy_node:main",
            "preflight = nino_rl.preflight:main",
            "train = nino_rl.train:main",
            "wait_for_sim = nino_rl.wait_for_sim:main",
        ],
    },
)
