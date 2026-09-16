# Nino Nav2-guided wheel-torque RL

**Current implementation: [algorithm v2](ALGORITHM_V2.md).** Active policy has
three actions (speed scale, common torque, differential torque) and 300 stacked
inputs. Use that guide for reward, curriculum and commands. The 54-input,
two-action descriptions below document v1 and are retained as historical notes;
old checkpoints cannot be resumed/deployed by v2.

This package trains a differential-drive Nino robot to follow a Nav2 hallway
plan by learning bounded left/right residual wheel torque. Nav2 retains map
loading, AMCL localization, global planning, obstacle-aware local control, and
the desired motion reference and baseline wheel-speed control. RL observes the
controller's `/cmd_vel_nav`; the actuator adapter adds the learned residual to
the bounded Nav2 baseline torque.

## Architecture

```text
long_hall map -> map_server -> AMCL ----------------------> map -> odom TF
                                      Nav2 goal
                                           |
                                           v
/scan + odom/TF ----------------> Nav2 planner/controller
                                     | /plan + /cmd_vel_nav reference
                                     v
/imu + /joint_states + /odom -> normalized 54-D observation -> PPO residual
                                                               |
                                                        [-1, 1]^2
                                                               |
                                           residual torque scale + saturation
                                                               |
                                               /wheel_torque_commands
                                                               |
                                         JointGroupEffortController (2 wheels)
```

The spawned model always comes from
`nino_description/urdf/nino.urdf.xacro`. It contains no Gazebo diff-drive
actuator. Its drive joints are `left_wheel_joint` and `right_wheel_joint`, both
with effort command interfaces. The expected tree is:

```text
map -> odom -> base_footprint -> base_link
                              +-> imu_link
                              +-> laser
                              +-> left/right wheel links
```

## Merged and modified packages

- `src/linorobot2`: upstream Jazzy Linorobot2 source, including its original
  description, bringup, Gazebo, base, navigation, maps, and RViz assets.
- `linorobot2_navigation`: the original `map.*`, `playground.*`, and
  `turtlebot3_world.*` files are byte-for-byte preserved. `long_hall.*` is a
  new map matching the Nino hallway; it does not replace an old map.
- `nino_description`: Nino Xacro, Gazebo Harmonic spawn, sensors, effort
  interfaces, and a ground-truth odometry stream used only to calculate
  simulated slip.
- `nino_control`: safe actuator adapter. Training combines bounded Nav2
  baseline torque with bounded RL residual torque; baseline evaluation disables
  the residual input.
- `nino_rl`: Nav2 goal/reset interface, curriculum terrain manager,
  observations, reward, termination, preflight, PPO training and evaluation.

## Topics and actions

| Name | Purpose |
|---|---|
| `/scan` | obstacle input for AMCL and both Nav2 costmaps |
| `/imu/data` | orientation, angular velocity, linear acceleration |
| `/joint_states` | wheel encoder position and velocity |
| `/odom` | wheel odometry and `odom -> base_footprint` |
| `/tf`, `/tf_static` | localization and robot/sensor transforms |
| `/map`, `/amcl_pose` | static map and localization result |
| `/plan` | Nav2 global/local tracking path supplied to RL |
| `/cmd_vel_nav` | Nav2 controller's desired local linear/angular reference for RL |
| `/cmd_vel` | collision-filtered Nav2 command used to produce baseline torque |
| `/wheel_torque_commands` | scaled RL residual, `[left_Nm, right_Nm]` |
| `/wheel_torque_applied` | saturated/rate-limited effort actually applied |
| `/ground_truth/odom` | simulation-only slip metric; never localization/control |
| `/navigate_to_pose` | configurable hallway task goal |

`navigation.nav_cmd_topic` selects the RL reference topic. The default uses
the obstacle-aware local controller output before the velocity smoother and
collision monitor. The monitor can stop publishing after a sustained zero
command, so its output is not a reliable navigation heartbeat. Baseline
actuation still uses the collision-filtered `/cmd_vel`.

## Observation and action

The policy observation is a normalized, clipped 54-element vector:

| Indices | Values | Reason |
|---|---|---|
| 0–17 | nine local-frame Nav2 look-ahead points | path curvature at 0.5–15 m, following Lee and Joe (2026) |
| 18–19 | `cos`/`sin` heading error | continuous heading representation without a ±π jump |
| 20–23 | robot linear/yaw velocity, left/right wheel velocity | body and drivetrain dynamics |
| 24–27 | normalized IMU quaternion | full attitude without Euler singularity |
| 28–33 | IMU gyro XYZ and acceleration XYZ | detects vibration, impacts, yaw motion and cable traversal |
| 34–38 | five minimum lidar sectors | compact obstacle proximity context |
| 39–40 | previous left/right action | actuator memory and smoother torque |
| 41–42 | Nav2 desired linear/angular velocity | local controller reference requested by this task |
| 43–44 | unit direction to local target | direct local steering reference |
| 45–47 | cross-track, local-waypoint distance, final-goal distance | progress and path-corridor context |
| 48–49 | roll and pitch | explicit rollover/stability state |
| 50–51 | left/right longitudinal slip | wheel speed versus Gazebo body speed |
| 52 | normalized time remaining for current waypoint | time-budget awareness |
| 53 | fresh-navigation validity flag | prevents blind actuation |

The 18 look-ahead coordinates, heading encoding, body velocity and yaw rate
come from the supplied 2026 path-tracking paper. The supplied vertically
challenging-mobility paper motivates goal-heading/velocity state, progressive
terrain, progress, rollover and timeout terms. Full IMU quaternion, gyro,
acceleration, roll/pitch and slip are additional task-specific observations:
they expose disturbances caused by cables and permit explicit stability and
slip objectives.

The continuous action is `a in [-1, 1]^2`. It maps to
`delta_tau = clip(a * max_wheel_torque_nm)` and is added to the Nav2 baseline
wheel torque while Nav2 commands motion. A second independent safety layer in
`effort_drive` clamps total torque, rate-limits changes, enforces wheel-speed
limits, and applies a 0.25 s watchdog.

## Reward and termination

The implementation follows:

```text
R = R_progress + R_heading/alignment + R_waypoint + R_goal
    - R_cross_track - R_impact - R_body_rate - R_attitude
    - R_slip - R_effort - R_action_rate - R_oscillation
    - R_stuck - R_timeout - R_terminal_failure
```

Progress uses reduction in remaining Nav2 path length, which remains stable
when Nav2 replans. A waypoint bonus is adjusted by its cumulative time margin;
the adjustment is deliberately smaller than the goal/progress terms. Goal
success also receives an early-finish bonus. Quadratic cross-track and heading
terms discourage drift, while roll/pitch, IMU vibration, wheel slip, effort and
first/second-difference action terms promote stable cable traversal. The
VDV-inspired impact term uses `(|a|-g)^4`, but normalizes and clips the value
before exponentiation so an IMU spike cannot dominate an entire PPO rollout.
Separate roll/pitch thresholds avoid penalizing normal small body motion.

An episode ends on final-goal success, flip, collision, sustained wrong
direction, sustained corridor exit, sustained invalid localization/navigation,
or timeout. Corridor and navigation faults must persist for configurable hold
durations, so a small temporary deviation does not terminate the episode.

Each episode records completion and waypoint times, success/failure reason,
mean/RMS/maximum cross-track error, roll/pitch and maximum tilt, RMS/maximum
wheel slip, RMS/maximum applied torque, IMU acceleration/vibration, and whether
the final and intermediate time budgets were met.

## Curriculum

Only one progressive phase is active per episode:

1. flat hallway, no cable;
2. one fixed small perpendicular cable;
3. one cable with randomized position;
4. randomized position and angle from -45 to +45 degrees;
5. randomized position, angle and diameter;
6. two to six randomized cables.

The legacy 29-cable model remains in `long_hall.sdf`, but the episode terrain
manager removes it before training and spawns only the current phase. The SDF
world itself is not overwritten.

The final goal defaults to map position `(30, 0, 0)`, two metres before the
hall's end wall. Intermediate reward checkpoints occur every five metres
along the path; they are not all at the hallway end. Nav2 does not detect a
hallway endpoint automatically: set `navigation.goal_pose` for another map.

## Install, build and launch

Install the ROS runtime dependencies (Ubuntu 24.04 / ROS 2 Jazzy):

```bash
sudo apt update
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-localization
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.7"
python -m pip install -r src/nino_rl/requirements.txt
python -m colcon build --symlink-install
source install/setup.bash
```

Keep the venv active while building. Using `python -m colcon` is required so
the installed `nino_rl` executables use the interpreter that contains PyTorch,
Gymnasium, and Stable-Baselines3.

Start only one simulation/Nav2 launch at a time. AMCL automatically receives
the configured starting pose and publishes `map -> odom`; the trainer also
reinitializes localization after every episode reset. A startup TF wait is
normal. The launch file now gates Nav2 until wheel controllers, all sensors,
`/odom`, and `odom -> base_footprint` are live. Persistent missing interfaces
therefore remain visible in the `nino_sim_readiness` log instead of racing Nav2
startup; preflight still performs the complete live-system check.

Start the RL topology:

```bash
ros2 launch nino_rl training_sim.launch.py headless:=true rviz:=false
```

Before training, run the live gate directly if desired:

```bash
ros2 run nino_rl preflight
```

`train` always runs all twelve checks itself and exits without training if any
check fails:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run nino_rl train --timesteps 500000 --device cpu --check-env
```

Evaluate an RL model with the RL topology still running:

```bash
ros2 run nino_rl evaluate --model rl_runs/<run>/nino_ppo_final.zip \
  --episodes 10 --device cpu
```

## Baseline versus RL

For a fair normal-controller run, stop the RL topology, then launch:

```bash
ros2 launch nino_rl baseline_nav.launch.py headless:=false rviz:=true
ros2 run nino_rl evaluate_baseline --episodes 10 --phase 6
```

Baseline mode accepts Nav2 `/cmd_vel` and rejects `/wheel_torque_commands`.
Training/evaluation accepts both and learns a bounded residual correction to
the Nav2 baseline. Compare the generated CSV/JSON in
`rl_runs/baseline` with `rl_runs/evaluation`; both contain the same path,
timing, attitude and slip metrics.
