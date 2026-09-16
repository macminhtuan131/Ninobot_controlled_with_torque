# Reward and policy v2

**Updated reward and policy:** [REWARD_POLICY_UPDATE.md](REWARD_POLICY_UPDATE.md)
supersedes the reward coefficients, network architecture, deadline handling and
resume instructions below. The 300-input/3-action transport contract is unchanged.

This is a new controller contract. Do not resume or deploy a v1 checkpoint.
The default input is 5 x 60 = 300 values; the action is 3 values. Active math
is in `nino_rl/control_v2.py`. The old `core.compute_reward` and YAML `reward`
section are retained for historical regression tests only; training calls
`control_v2.compute_reward` and reads `reward_v2`.

## What changed

- PPO still uses separate 256/256 ELU actor and critic networks.
- Action `[speed, forward, yaw]` is normalized to [-1,1]. Speed maps to [0,1].
  Residuals are `0.5 * clip(forward-yaw)` and `0.5 * clip(forward+yaw)` Nm.
- `/nino_rl/control_command` carries `[speed_scale, left_Nm, right_Nm]` in one
  Float64MultiArray. `effort_drive` scales the *actual* post-safety Nav2
  `/cmd_vel` before computing the PI wheel-speed targets, then adds residuals.
  It preserves the existing final torque, speed and slew limits. Positive
  in-place turn scales have a floor of 0.25; explicit zero still stops them.
- Speed scale slews at 2/s, with immediate zero reference on stop/watchdog.
  Actual braking remains constrained by wheel acceleration/torque limits.
  PI integral is bled while slowing. A stale v2 heartbeat stops the scaled
  reference; it never silently reverts to full-speed Nav2. The odometry reset
  service releases v2 ownership. Legacy torque commands cannot override it.
- Actor receives no simulator ground-truth slip values. Ground truth remains
  required only for simulated reward and evaluation metrics.
- IMU body axes match base_link in the current URDF. Rotate measured specific
  force into world z, then subtract 9.80665 m/s². Set
  `policy_v2.imu_includes_gravity: false` ONLY for drivers already removing
  gravity. Sensor position is the existing IMU position, not the center of mass.
- Timestamped 50 Hz IMU samples are integrated between policy steps. Missing
  coverage, paused simulation clocks and stale Nav2/ground-truth streams abort
  training instead of silently generating fabricated rewards. This does not
  guarantee capture of impacts faster than the sensor bandwidth.
- PPO pauses Gazebo during optimizer updates, then resumes at the next rollout;
  optimizer wall time therefore does not consume the episode time budget.
- Episode TensorBoard scalars use means over episodes at each logging interval,
  rather than overwriting each other with only the final episode's value.

## Observation (one 60-value frame)

Old observation indices 0..49 (50 values), old indices 52..53 (time fraction and
Nav2 validity), previous 3D action, world vertical acceleration / 10, and four
terrain-preview values. The old previous-action slots retain the two decoded,
normalized wheel residuals. The two ground-truth-slip slots are omitted.
Frames are stacked oldest to newest, clipped to [-5,5], reset by repeating the
first frame. Training, evaluation and deployment use the same implementation.

Optional `/nino_rl/terrain_preview` is a Float64MultiArray:

`[distance_ahead_m, left_track_height_m, right_track_height_m]`

The producer must publish a fresh, sensor-derived preview along the planned
path (or from a surveyed map available during deployment), not Gazebo cable
spawn coordinates. Distance must be nonnegative and all values finite. Values
are normalized by 5m / 0.1m / 0.1m; the fourth observation value is validity.
Data expires after 0.5 wall seconds. Missing preview is zeros with validity=0.
Set `require_terrain_preview: true` to stop when missing. No preview producer or
new physical sensor is included in this change. The existing horizontal 2D
LiDAR cannot guarantee detection of low bumps: default mode learns reactive
control, not guaranteed anticipatory braking. Use the same preview setup for
training and deployment. No contact-force or wheel-liftoff detector is included.

## Active reward

Let h = actual simulation step duration / 0.1s and C(x)=min(x²,9).
Distances are metres; angles radians. Coefficients are initial tuning values.

| Term | Definition |
|---|---|
| progress | 20 clip(previous_remaining - remaining, -0.1, 0.1) |
| lateral | -0.8 h C(lateral_error / 0.30) |
| heading | -0.2 h C(heading_error / 0.35) |
| impact | -0.05 impact_scale integral(min((a_world_z/2)^4,81) dt) / 0.1 |
| body_rate | -0.05 h [C(gyro_x) + C(gyro_y)] |
| attitude | -0.5 h [C(max(abs(roll)-0.20,0)/0.15) + C(max(abs(pitch)-0.30,0)/0.15)] |
| slip | -0.1 h [C(slip_left/0.30) + C(slip_right/0.30)] |
| smoothness | -0.02 sum((action - previous_action)²) / h |
| effort | -0.01 h mean((commanded_residual / max_residual)²) |
| time | -0.01 h |
| stall | -0.5 h after <5cm net progress in 3s while Nav2 requests forward motion and goal is not reached |
| terminal | +100 success; -100 collision/rollover/wrong direction; -75 off path; -50 timeout |

`impact_scale = min(1, 0.25 + (phase-1)/5)` ramps impact penalty across phases.
Failure takes precedence over success and timeout. Goal tolerance/stopped-arrival
criteria remain in `core.goal_reached`. Collision currently uses the existing
LiDAR proximity test; it is not a physical contact classifier. The wheel-ground
contact of traversable bumps is not itself counted as a collision.

All IMU samples in a policy window contribute using zero-order hold. Coverage
must exceed 80%; no sample is held for more than 0.1s. Duplicate timestamps are
ignored and backwards time clears history. Log uncapped peak and RMS vertical
acceleration separately. This is VDV-inspired, NOT an ISO 2631 measurement.

Path identities are frozen per episode in map coordinates while the current
map->odom transform is applied to both previous/current projection. This avoids
reward from planner path replacement or localization-frame shifts.

## Apply, rebuild and run

The incremental patch targets commit 6e9a729 plus the previous
`nino_rl_cpu_reward_and_startup.patch`. Keep local changes and do not use
`git reset --hard`. `git apply --check` must succeed before applying. Do not
apply both incremental and full patches to the same checkout.

Stop old training and simulator processes before rebuilding. Existing saved
checkpoints are untouched, but v1 models cannot be resumed as v2.

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install --packages-up-to nino_rl
source install/setup.bash
ros2 launch nino_rl training_sim.launch.py headless:=true rviz:=false
```

Wait for simulator readiness and Nav2 active. In another sourced terminal:

```bash
ros2 run nino_rl train --phase 1 --timesteps 4096 --checkpoint-every 2048 --check-env
```

Let the smoke test finish and produce `nino_ppo_final.zip`. Then train a new
longer run or resume that v2 model, e.g.:

```bash
ros2 run nino_rl train --phase 1 --timesteps 100000 --checkpoint-every 10000
```

Device selection remains `--device cpu|cuda|auto`. The effective configuration,
including phase/device overrides, is saved under each run directory.

Evaluate sequentially in the same simulator, never with train simultaneously:

```bash
ros2 run nino_rl evaluate --model /absolute/path/to/nino_ppo_final.zip --phase 1 --episodes 30 --seed 10000
ros2 run nino_rl evaluate_baseline --phase 1 --episodes 30 --seed 10000
```

Inspect success, completion time, RMS/max path deviation, uncapped peak/RMS
vertical acceleration and slip. Both evaluations default to randomization off;
pass `--randomized` to BOTH for matched tests. Existing randomization affects
residual traction/noise and observations; it is not a full physical contact
friction/mass randomizer. Use different seeds for final reporting than tuning.

Manual curriculum gate: consider advancing only after >=90% held-out success
and acceptable path/impact metrics on the current phase. No automatic claim of
optimality is made. Phase 1 flat; 2 fixed small cable; 3 random position; 4 random
angle; 5 random diameter; 6 multiple cables. These reuse the actual cable world;
no ramp generator was added. Resume a v2 model with `--phase 2`, etc.; `--timesteps`
is ADDITIONAL steps when resuming.

## Validation

Portable tests (no ROS needed):

```bash
PYTHONPATH=src/nino_rl:src/nino_control python -m unittest discover -s src/nino_rl/test -p test_control_v2.py -v
```

Tests cover gravity compensation, inter-step impact capture, missing samples,
signed progress, terminal precedence, actor independence from truth slip,
history reset, checkpoint compatibility, action mapping and real controller
methods exercised with a fake ROS clock/transport (scaled PI targets, zero
commands, stale heartbeat, command ownership). ROS/Gazebo end-to-end validation
must still run on the robot workspace. No end-to-end training improvement has
been measured in the editing environment.
