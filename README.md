# Nino: straight-line residual PPO and trajectory evaluation

This branch controls the **caster-supported differential-drive Nino AMR** in
ROS 2 Jazzy / Gazebo Harmonic. The model has two powered wheels, two passive
casters, IMU, wheel encoders and **2D** LiDAR. It is not an unsupported two-wheel
inverted pendulum. Nav2 is disabled for this experiment. The pipeline is a
direct straight `/cmd_vel` reference → wheel-speed PI + bounded PPO residuals
→ effort controller. PPO also scales the forward velocity reference.
The PPO policy supplies the steering correction normally associated with a
geometric tracker: differential wheel residuals must keep cross-track and
heading error near zero before forward progress receives full credit, including
while crossing the configured cable bump.

The updated workflow uses NVIDIA CUDA by default and refuses an automatic CPU
fallback. Gazebo physics and ROS still consume CPU; this is one Gazebo world,
not GPU-parallel Isaac Lab simulation. Keep only one train/evaluate/policy
process connected to that world at a time.

Nino simulation processes automatically use ROS domain 77 with localhost-only
discovery, preventing another machine or simulator from injecting a conflicting
`/clock`. To use manual `ros2 topic` or `ros2 service` diagnostics, first run
`export ROS_DOMAIN_ID=${NINO_ROS_DOMAIN_ID:-77}` and
`export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` in that shell.

- [Reward, policy and engineering decisions](src/nino_rl/RL_IMPROVEMENTS.md)
- [Vietnamese quick guide](src/nino_rl/README_VI.md)
- [Bài viết nghiên cứu tổng quan bằng tiếng Việt](docs/BAO_CAO_NGHIEN_CUU_RL_NINO.md)
- [Simulation and hardware reference](docs/HARDWARE_REFERENCE.md)

The current training contract is revision 21. Its validator can migrate only
the explicitly supported v2 revisions 9–20 when their saved configuration
matches; otherwise start a new run. Old 54-input/2-action models are
incompatible. No pretrained weights or measured performance gains are included.

## 1. Prepare Ubuntu and the NVIDIA GPU

Use Ubuntu **24.04**, its system Python **3.12**, and an NVIDIA-capable machine.
Run these commands on the machine that will actually train, not a separate
laptop without the GPU. Skip driver installation if `nvidia-smi` already works.

```bash
sudo apt update
sudo apt install git curl locales software-properties-common \
  python3-venv python3-pip build-essential ubuntu-drivers-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

After reboot, `nvidia-smi` must list the intended GPU (for example RTX 4080).
Resolve driver/Secure Boot issues before installing RL dependencies. Installing
`nvidia-utils` alone does not install a working kernel driver. See
[Ubuntu's NVIDIA driver guide](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/).

## 2. Install ROS 2 Jazzy, Gazebo, Nav2 and RViz

If `/opt/ros/jazzy/setup.bash` already exists, skip the ROS repository setup.
Otherwise enable Universe and install the official ROS apt-source package:

```bash
sudo add-apt-repository universe
NINO_ROS_APT_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/nino-ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${NINO_ROS_APT_VERSION}/ros2-apt-source_${NINO_ROS_APT_VERSION}.noble_all.deb"
sudo dpkg -i /tmp/nino-ros2-apt-source.deb
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop ros-dev-tools
```

Install the robot dependencies:

```bash
sudo apt install python3-rosdep python3-colcon-common-extensions \
  ros-jazzy-ros-gz ros-jazzy-ros-gz-interfaces ros-jazzy-gz-ros2-control \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-controller-manager \
  ros-jazzy-effort-controllers ros-jazzy-joint-state-broadcaster \
  ros-jazzy-ros2controlcli ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-localization ros-jazzy-rviz2 \
  ros-jazzy-tf2-ros
source /opt/ros/jazzy/setup.bash
```

Use the [official Jazzy installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
if repository packaging changes. `ros-jazzy-ros-gz` supplies the compatible
Gazebo integration; do not install Gazebo Classic for this project.

## 3. Get the source and apply the patch

For a fresh checkout:

```bash
cd ~
git clone --branch add_rl https://github.com/macminhtuan131/Ninobot_controlled_with_torque.git ninorobot
cd ~/ninorobot
```

For an existing checkout, enter its directory and check `git status`. Preserve
local edits before changing branches. The supplied patch was made against
`d015b8239648c64974e5badd041aa87f3a5a0fdc` on `add_rl`.

```bash
git rev-parse HEAD
git apply --check ~/Downloads/ninobot-rl-improvements.patch
git apply ~/Downloads/ninobot-rl-improvements.patch
```

Use the actual download path. If the check reports conflicts, stop and reconcile
the branch/local changes; do not force the patch or discard your work.
Skip applying the patch if these changes are already in your checkout.

Initialize rosdep once (skip `init` if already initialized), then resolve the
packages required by this training workspace:

```bash
sudo rosdep init
rosdep update
rosdep install --from-paths src/nino_description src/nino_control src/nino_rl \
  src/linorobot2/linorobot2_navigation --ignore-src --rosdistro jazzy -r -y
```

## 4. Create the Python environment and install CUDA PyTorch

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r src/nino_rl/requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The CUDA 12.6 wheel above is an explicit example from the
[official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).
Use a compatible NVIDIA driver. A `+cpu` wheel or `torch.version.cuda == None`
is wrong for this workflow. The system CUDA toolkit is not required just to use
these wheels. `--system-site-packages` is necessary for ROS Python modules;
`python3-venv` prevents the `ensurepip is not available` error.

## 5. Build with the venv interpreter

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install --packages-up-to nino_rl
source install/setup.bash
ros2 run nino_rl check_cuda
```

Expect `CUDA khả dụng: True`, the correct GPU, and a successful CUDA matrix
multiplication. Always use **`python -m colcon` inside the venv** so installed
Python entry points can import PyTorch. Do not move the venv after building.

For every new terminal, run these four lines first:

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
```

## 6. Start simulation and check it

Set the endpoint and forward controller in `src/nino_rl/config/ppo.yaml`:

```yaml
goal_tolerance_m: 0.003
navigation:
  start_pose: [0.0, 0.0, 0.0]
  goal_pose: [6.0, 0.0, 0.0]
  straight_speed_m_s: 0.75
  minimum_approach_speed_m_s: 0.03
  goal_slowdown_distance_m: 1.00
```

The robot is successful when it reaches the 3 mm endpoint region or crosses
the goal plane. The success reward is reduced smoothly by final position and
heading error, so inaccurate arrivals still score less without becoming an
overshoot failure. The direct command always has
`angular.z = 0`; the residual policy may apply differential wheel correction to
counter drift while following the straight reference. The curriculum cable is
at `x=4 m`, leaving 2 m for recovery and drift measurement before the 6 m goal.
With `headless:=false`, Gazebo displays the goal as a bright green disc, pole,
and flag. The marker is visual-only and cannot collide with the robot or LiDAR.

After each successful episode, one extra side feature is added near the
cable zone, cycling through pothole-like rough patches, obstacles, and short
cables until 20 are present. A ±0.60 m center corridor always remains clear.

Terminal A:

```bash
ros2 launch nino_rl training_sim.launch.py headless:=true
```

For visual debugging, stop that launch and restart with `headless:=false`.
Do not launch both copies. This training launch
starts the effort controller, sensors and Gazebo reset services. It does not
start Nav2, AMCL, a map server, or a planner.
It refuses to start if the same Gazebo world is already running; this prevents
multiple `/clock` and IMU publishers from corrupting lockstep training.
Wait for the simulator and controllers. Terminal B:

```bash
ros2 control list_controllers
ros2 param get /effort_drive accept_torque
ros2 topic hz /imu/data
```

Both `joint_state_broadcaster` and `wheel_effort_controller` must be active;
`accept_torque` must be `True`. Stop the topic-rate command with Ctrl-C.
The nominal IMU frequency is 50 Hz in simulation time. The training command
performs the mandatory 12-point preflight, including actuation, before learning.
Use `training_sim.launch.py`, not the standalone description launch, for RL.

## 7. Smoke-test, then train phase 1

Keep Terminal A running. Terminal B:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 4096 --check-env
```

This is a plumbing test, not enough training to learn a useful policy. Once it
works, start the real run:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 500000 \
  --checkpoint-every 25000
```

The run prints its directory, e.g. `rl_runs/20260917-123456-123456/`. It contains
`ppo.yaml`, software/device metadata, `monitor.csv`, TensorBoard logs,
`checkpoints/nino_ppo_*_steps.zip` and `nino_ppo_final.zip` when finished.
Rollouts are 2048 steps, so SB3 can exceed the requested step count to complete
a rollout. A 500000-step run needs at least 50000 simulated seconds at 10 Hz,
plus reset/update overhead; actual wall time depends on Gazebo throughput.

Terminal C:

```bash
tensorboard --logdir rl_runs --port 6006
```

Open `http://localhost:6006`. Inspect success, path RMSE/P95, completion,
heading RMSE, impact RMS, torque, reward terms and PPO KL/entropy together.
Gazebo uses lockstep training: every action advances exactly 50 two-ms physics
steps (0.1 simulated seconds), then waits for fresh sensor and torque feedback while
paused. Optimizer wall time therefore cannot consume the mission deadline.
CUDA memory usage alone is not evidence of successful learning.

A Ctrl-C or runtime transport failure saves `nino_ppo_interrupted.zip` if a
model exists; its unfinished rollout is discarded on resume. Fix the transport
problem before continuing. Resume only compatible v2 runs with their saved config:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 500000 \
  --config rl_runs/YOUR_RUN/ppo.yaml \
  --resume rl_runs/YOUR_RUN/checkpoints/nino_ppo_25000_steps.zip
```

`--timesteps` is additional training. A saved but never-updated model is still
untrained. Training seeds are set by `seed` in the YAML; use separate configs
with different seeds for repeatability checks.

## 8. Evaluate baseline and PPO automatically

Stop the trainer; keep the same training simulation running. Run these commands
**sequentially**, using the real model/config paths printed by training:

```bash
ros2 run nino_rl evaluate_baseline --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml --output rl_runs/baseline-p1
ros2 run nino_rl evaluate --device cuda --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml --model rl_runs/YOUR_RUN/nino_ppo_final.zip \
  --output rl_runs/ppo-p1
```

Each command prints a timestamped report directory. It contains `summary.json`,
`episodes.csv`, a config/metadata snapshot, and for every completed episode:
`episode-001/trajectory.csv`, `actual.csv`, `reference.csv`, `metrics.json`, and
`trajectory.png`. `trajectory.csv` contains `time_s,x_m,y_m,yaw_rad,frame_id`
and can be loaded directly by pandas, a spreadsheet, or MATLAB. The PNG overlays
the fixed straight reference and the measured robot trajectory.
Automatic metrics
include time-weighted path/cross-track RMSE, P95/max path error, heading RMSE,
endpoint error, completion, backtracking, success, timing, slip, torque and IMU
impact. Incomplete evaluations are marked and cannot be compared as full runs.

```bash
ros2 run nino_rl compare_evaluations \
  --baseline rl_runs/baseline-p1/BASELINE_STAMP/summary.json \
  --candidate rl_runs/ppo-p1/PPO_STAMP/summary.json \
  --output rl_runs/comparison-p1.json
```

Compare **success first**, then errors/comfort and successful completion time.
A stopped or failed robot can have low RMSE. Reports separate all episodes from
successful episodes. The comparator checks phase, seeds, count, perturbation
mode and task config. Matching seeds reproduce the terrain draws; they do not
make asynchronous ROS/Gazebo execution bitwise deterministic.

`--randomized` on **both** evaluators tests full-strength residual/sensor
perturbations. These are not physical friction/mass changes.
The baseline receives zero residual torque, including under randomized testing.

## 9. Run the cable phases after evaluation

Every phase has exactly one cable at 4 m. The requested order is hard to easy;
only cable diameter and absolute angle define phase difficulty. The sign of a
nonzero angle is randomized so the policy does not favor one wheel.

| Phase | Difficulty | Diameter | Absolute angle |
|---|---|---:|---:|
| 1 | Hardest | 15 mm | 45 degrees |
| 2 | Very hard | 13 mm | 36 degrees |
| 3 | Hard | 11 mm | 27 degrees |
| 4 | Medium | 9 mm | 18 degrees |
| 5 | Easy | 7 mm | 9 degrees |
| 6 | Easiest | 5 mm | 0 degrees |

Domain randomization is disabled by default so the phase comparison is based
only on size and angle. Use `--randomized` during evaluation only when you
specifically want a robustness test.

Suggested initial gate: at least 19/20 held-out successes, no rollover/collision,
acceptable P95 path error (e.g. <0.25 m for this hallway), and no material comfort
regression versus baseline. These are proposed acceptance criteria, not measured
results. Use additional seeds for a final test, distinct from development seeds.

```bash
ros2 run nino_rl train --device cuda --phase 2 --timesteps 500000 \
  --config rl_runs/PHASE1_RUN/ppo.yaml \
  --resume rl_runs/PHASE1_RUN/nino_ppo_final.zip
```

Evaluate phase 2 using the same phase/seeds for baseline and PPO. Repeat for
phases 3–6, resuming the preceding phase. Keep several checkpoints; the final
one is not automatically best. Evaluate them on the same development seeds,
then test the chosen model on new seeds. Do not run an evaluation callback
against the same live world while the trainer is collecting a rollout.

## 10. Compare against your own ideal path or timed trajectory

A geometric reference CSV uses `x_m,y_m,frame_id`. An actual trace uses
`time_s,x_m,y_m,yaw_rad,frame_id`; yaw is optional. All coordinates must use the
same frame. The automatically exported files already follow this format.

```bash
ros2 run nino_rl trajectory_metrics \
  --actual rl_runs/ppo-p1/PPO_STAMP/episode-001/actual.csv \
  --reference rl_runs/ppo-p1/PPO_STAMP/episode-001/reference.csv \
  --mode path --output rl_runs/path-score.json --plot rl_runs/path-overlay.png
```

To compare with a scheduled ideal trajectory, give the reference a strictly
increasing `time_s` column as well, then use `--mode timed --max-gap 0.5`.
`position_rmse_m` compares interpolated positions at common simulation times;
there is no timestamp shift, rigid alignment, or extrapolation. Clock origins
must match. Exported time is relative to the first odometry sample; the episode
metrics record `clock_origin_sim_s` for converting absolute simulator timestamps. Long gaps, duplicate timestamps and frame mismatches are rejected.
Coverage is reported so an early-ending run cannot hide unobserved reference
time. Choose the gap limit according to your sampling rate, not to conceal loss.

A geometric straight path has no desired timestamps. Its default automatic score is therefore
**path RMSE**, not timed position RMSE. Poses are wheel odometry transformed by
localization, not external ground truth. For physical accuracy studies, export
motion-capture or correctly transformed simulator ground-truth poses instead.
Nearest-segment progress is ambiguous on self-crossing paths; use timed scoring
for such experiments. Do not concatenate several episode clocks into one CSV.

The scoring tools also work without ROS:

```bash
PYTHONPATH=src/nino_rl python -m nino_rl.trajectory_metrics --help
```

## Troubleshooting and scope

| Symptom | Action |
|---|---|
| `ensurepip` missing | Install `python3-venv`, recreate the incomplete venv |
| Torch `+cpu` / CUDA false | Install the CUDA wheel in the same venv; check driver; run `check_cuda` |
| Cannot import `rclpy` | Source Jazzy and use system Python 3.12 + `--system-site-packages` |
| `ros2 run` cannot import Torch | Rebuild Python packages with active venv and `python -m colcon` |
| Zero torque preflight | Use training launch, active effort controller, `accept_torque=True` |
| IMU coverage timeout | Check `/clock`, `/imu/data` stamps and load; the bounded wait handles delivery races but does not fabricate samples |
| Resume contract error | Use matching post-update run/config, or start a new run |
| No `/cmd_vel` subscriber | Rebuild `nino_control`, restart the simulation, and rerun preflight |

This patch does not introduce SWAE, a 3D terrain map, an ESKF, Isaac Lab,
asymmetric privileged critics, or unvalidated slope balancing. The existing
optional terrain-preview input remains invalid/zero without a real producer.
See the design note for the selection rationale and exact reward. Hardware
transfer needs separate validation; the included simulator cannot establish it.
