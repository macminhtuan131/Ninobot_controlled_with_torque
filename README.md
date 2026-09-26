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
- [Timeout diagnosis and revision 27 validation](docs/RL_TRAINING_DIAGNOSIS_20260922.md)
- [Goal-first reward, policy and terrain curriculum (revision 30)](docs/RL_GOAL_FIRST_REVISION_28.md)
- [Vietnamese quick guide](src/nino_rl/README_VI.md)
- [Bài viết nghiên cứu tổng quan bằng tiếng Việt](docs/BAO_CAO_NGHIEN_CUU_RL_NINO.md)
- [Simulation and hardware reference](docs/HARDWARE_REFERENCE.md)

The current training contract is revision 30. Because its goal-distance reward,
restored full-size terrain, per-action exploration, physics-completion timing,
sensor-invisible goal marker, reward balance,
traversable-challenge flags, terrain geometry, downward preview, and rolling
hazard curriculum changed, older checkpoints cannot be resumed; start a new
run. New checkpoints retain rolling curriculum state, so
resume no longer resets hazard difficulty. Old models may still be used for
inference with their matching config. Old 54-input/2-action models are
incompatible. The `Completed_train` branch includes [trained weights and recorded training
results](src/nino_rl/models/completed_train/README.md). The final checkpoint has
1,001,634 cumulative steps; held-out performance gains and hardware transfer
have not been established.

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
goal_tolerance_m: 0.10
navigation:
  start_pose: [0.0, 0.0, 0.0]
  goal_pose: [6.0, 0.0, 0.0]
  straight_speed_m_s: 0.75
  minimum_approach_speed_m_s: 0.03
  goal_slowdown_distance_m: 0.50
```

The robot is successful only while its center is inside the 10 cm endpoint
circle and its heading is within 12 degrees of the path direction. Merely
crossing the goal plane no longer counts, so overshoots and lateral misses are
failures. The direct command always has
`angular.z = 0`; the residual policy may apply differential wheel correction to
counter drift while following the straight reference. The curriculum cable is
at `x=4 m`, leaving 2 m for recovery and drift measurement before the 6 m goal.
With `headless:=false`, Gazebo displays the goal as a bright green disc, pole,
and flag. The marker is visual-only and cannot collide with the robot or LiDAR.

Training starts with one adaptive hazard. One more is added after five
consecutive successful episodes. Any failure breaks the streak, and adding a
hazard clears the streak. The cap is eight adaptive hazards plus the single
phase cable. Rolling results and the
current hazard count are stored in every regular, final, and interrupted
checkpoint. The established pothole-like patches, low stepped bumps, short
transverse cables, and road-groove surrogates are randomized per episode inside
the four-metre training zone. Every center is uniformly sampled between the
left and right wheel-center lines (approximately ±17.1 cm). The 35 cm post geometry is classified as a blocking
route-planning obstacle and is never sampled or rewarded as a wheel-control
challenge. Adaptive geometry remains at full height in every phase: in
particular, the pothole basin keeps its 30 mm relief. The phase cable changes
diameter and skew; its tilt sign is randomized every episode, including the
easiest phase's ±5-degree cable.

The road groove uses two gently sloped, 20 mm-high shoulders around a 100 mm
floor-level transverse channel. This creates a physical relative drop for the
wheels. Gazebo cannot subtract a runtime hole from the hall floor, so it is a
local raised-road surrogate rather than below-world geometry.

Speed is a learned continuous action: PPO scales the 0.75 m/s straight
reference from 0 to 100% on every 0.1 s policy step. A 20 Hz downward-looking
LiDAR fan provides a fresh, deployable preview of low cable/terrain relief, in
addition to the forward safety LiDAR. Successful arrival before the 15 s target
earns a proportional bonus, while impact, slip, torque, timeout, and path terms
prevent "always full speed" from being the only useful strategy. TensorBoard
records mean/min/max speed scale and mean ground speed for every episode.

The other two actions are common and differential residual effort. They map
bijectively onto bounded left/right wheel torque, so PPO can increase one wheel,
decrease the other, or change both independently while the 500 Hz PI loop keeps
the requested wheel velocity stable. Acceleration is not a competing actuator
mode: it is the physical result of bounded torque, velocity targets, and the
controller's wheel-acceleration/slew limits.

Traversable challenges use one-shot privileged reward flags that are not added
to the actor observation. Flags require a powered wheel's swept footprint to
intersect the region, rather than merely passing a bump under the chassis.
This is a geometric crossing estimate, not a physical contact-force sensor.
Entering a challenge can earn at most +2 over the
whole episode, clearing its local forward edge at most +8, and a successful
goal earns up to another +90 in proportion to the fraction cleared. These
totals are divided across the episode's challenge count, so adding hazards does
not inflate the maximum return. Oscillation cannot collect a flag twice, and a
collision, rollover, timeout, off-path, or wrong-direction step earns no new
challenge bonus. Episode reports and TensorBoard include chosen/cleared counts
and fractions under `challenge_*` fields.

At phase 1, the randomized pothole is a 0.60 m round, 30 mm-deep relative basin with smooth
approximately 12.5-degree entry/exit ramps and a roughly 3 mm leading edge. It
is sized to admit the 16 mm caster wheels while still requiring useful drive
effort. Gazebo cannot subtract a randomly spawned shape from the existing flat
floor, so this is an annular raised-basin surrogate rather than a literal hole
below the hall floor; a true excavated hole would require replacing the floor
with a pre-cut mesh or heightmap.

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
`ppo.yaml`, software/device metadata, `monitor.csv`, `episodes.jsonl`, TensorBoard logs,
`checkpoints/nino_ppo_*_steps.zip` and `nino_ppo_final.zip` when finished.
Rollouts are 2048 steps, so SB3 can exceed the requested step count to complete
a rollout. Training is already active while those steps are collected; the
first PPO optimizer table does not appear until the first rollout completes.
Watch the live `EPISODE END` lines for collection progress. A 500000-step run
needs at least 50000 simulated seconds at 10 Hz,
plus reset/update overhead; actual wall time depends on Gazebo throughput.

Terminal C:

```bash
tensorboard --logdir rl_runs --port 6006
```

Judge learning using `episode/success`, `episode/timeout_failure`, and
`episode/endpoint_distance_m` together. `episode_reward/*` shows whole-episode
reward totals, while `reward_terms/*` shows per-step averages. The corrected
timing should keep `episode/max_clock_error_seconds` near zero and
`episode/max_motion_sensor_lag_seconds` below 0.04 s. Individual records are
saved in `episodes.jsonl`, including the termination reason and curriculum phase.
Revision 28 needs a fresh run; older models learned under different timing and
reward semantics. Restart the simulator after rebuilding so both lidar masks
exclude the decorative goal beacon.

Progress now measures reduction in actual endpoint distance, including lateral
misses and overshoot. The speed reference stays low after passing the goal;
more than 0.30 m longitudinal overshoot ends the forward-only task with
`goal_missed` and a failure penalty. An in-circle heading error retains a small
forward reference so PPO still has differential steering authority. Initial
Gaussian standard deviations are `[0.20, 0.12, 0.05]` for speed/common torque/
steering; each remains trainable. Watch `policy/std_*` and
`episode/goal_missed_failure` alongside the other outcomes.

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

## 9. Run all six cable phases automatically

Every phase has exactly one cable at 4 m. Training progresses from easy to hard;
only cable diameter and absolute angle define phase difficulty. The sign of a
nonzero angle is randomized so the policy does not favor one wheel.

Run one continuous training job with the default configuration:

```bash
ros2 run nino_rl train --device cuda --timesteps 600000 --check-env
```

Omit `--phase`: specifying it deliberately locks the run to one phase.
Steps 0–99,999 use phase 6, then phases 5, 4, 3, and 2 each receive
100,000 steps; phase 1 starts at step 500,000. Terrain changes on the first
episode reset after a boundary, so an active crossing is never interrupted.
The phase remains 1 after the schedule finishes. Resume preserves absolute
step progress and the hazard success streak. API checks do not advance the
schedule. PPO may finish its final rollout beyond the requested step budget.

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
ros2 run nino_rl train --device cuda --timesteps 300000 \
  --config rl_runs/YOUR_RUN/ppo.yaml \
  --resume rl_runs/YOUR_RUN/checkpoints/nino_ppo_300000_steps.zip
```

Evaluate each phase using the same phase/seeds for baseline and PPO. The phase
schedule is step-based and does not wait for evaluation success. Keep several checkpoints; the final
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
terrain-preview input is now driven by the simulated downward LiDAR. Hardware
deployment requires an equivalent calibrated producer; simulation alone cannot
establish transfer performance. See the design note for the selection rationale
and exact reward.
