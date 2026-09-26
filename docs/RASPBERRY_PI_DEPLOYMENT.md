# Run the trained policy on a Raspberry Pi

The trained policy is `src/nino_rl/models/completed_train/nino_ppo_final.zip`.
Keep it with `ppo.yaml` from the same directory. The ZIP is a Stable-Baselines3
checkpoint, so the Pi runs it through the ROS `nino_rl` package; it cannot be
flashed to a motor controller or run by itself. A checkout of the
`Completed_train` branch already contains both files.

## 1. Prepare the Pi

Use a Pi running a 64-bit Ubuntu 24.04 installation with ROS 2 Jazzy and
Python 3.12, matching this repository's ROS environment. Install the ROS
packages needed by the robot's controller and sensors, and provide a working
`ros2_control` hardware system for the motors. The repository's
`nino_description` launch files use Gazebo hardware and do **not** start real
motors. The [hardware reference](HARDWARE_REFERENCE.md#moving-from-gazebo-to-xiaomi-cybergear-hardware)
describes the required two-wheel effort interface.

Clone the project on the Pi, or copy the full workspace to it:

```bash
git clone --branch Completed_train https://github.com/macminhtuan131/Ninobot_controlled_with_torque.git ~/ninorobot
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r src/nino_rl/requirements.txt
python -m colcon build --packages-select nino_control nino_rl
source install/setup.bash
```

Install a CPU PyTorch build compatible with the Pi's 64-bit Python before the
`pip install -r` step if your package index does not supply one. CUDA is not
needed on the Pi. Build with the same venv active so the `ros2 run` entry point
can import PyTorch and Stable-Baselines3. If your hardware workspace has its
own ROS packages, build and source those as well. `nino_description` and the
Gazebo training launch are not needed to run inference on a physical robot.
The shared ROS interface currently imports `ros_gz_interfaces` at startup,
however, so its ROS Python package must be installed even on the Pi.

Check that the policy and matching config were installed:

```bash
ros2 pkg prefix --share nino_rl
ls "$(ros2 pkg prefix --share nino_rl)/models/completed_train/"
python -c 'import torch, stable_baselines3, rclpy; print(torch.__version__, stable_baselines3.__version__)'
```

## 2. Connect the real robot interface

The motor hardware must expose the `left_wheel_joint/effort` and
`right_wheel_joint/effort` command interfaces, report wheel joint position and
velocity in `/joint_states`, and run an active `wheel_effort_controller` plus
`joint_state_broadcaster`. The `nino_control` `effort_drive` node closes the
wheel-speed PI loop and sends torque to `/wheel_effort_controller/commands`.
Start the real hardware controller manager using **your hardware's launch
file**; do not use `nino_description sim.launch.py` on the Pi.

The policy also needs these live ROS topics:

| Topic | Message | Purpose |
|---|---|---|
| `/odom` | `nav_msgs/msg/Odometry` | Robot pose and speed in `odom` |
| `/imu/data` | `sensor_msgs/msg/Imu` | Body motion; config expects gravity in acceleration |
| `/joint_states` | `sensor_msgs/msg/JointState` | Measured left/right wheel state |
| `/scan` | `sensor_msgs/msg/LaserScan` | Forward obstacle stop |
| `/terrain_scan` **or** `/nino_rl/terrain_preview` | `LaserScan` or `std_msgs/msg/Float64MultiArray` | Downward terrain preview |
| `/wheel_torque_applied` | `std_msgs/msg/Float64MultiArray` | Applied left/right wheel torque feedback from `effort_drive` |

For `/nino_rl/terrain_preview`, send `[forward_distance_m,
left_relief_m, right_relief_m]` more often than every 0.5 seconds. If using
`/terrain_scan`, the built-in conversion assumes a sensor height of 0.2325 m
and pitch of 0.45 rad; calibrate or adapt it for the physical mounting before
using the policy. The training config requires fresh preview data and will
command zero torque when it is absent. The policy also stops for missing core
sensors, obstacles, excessive tilt, path departure, or timeout.

With the hardware controller manager running, start the existing PI/effort
adapter in another terminal. The supplied YAML defaults to simulation time, so
override it for real time:

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run nino_control effort_drive --ros-args \
  --params-file "$(ros2 pkg prefix --share nino_control)/config/effort_drive.yaml" \
  -p use_sim_time:=false -p accept_cmd_vel:=true -p accept_torque:=true
```

If another node already publishes `/odom` or its TF, configure `effort_drive`
to publish `/odom/unfiltered` and set `publish_odom_tf:=false`; arrange for
the localization node to provide `/odom` and the `odom` to `base_footprint`
transform. There must be one authoritative pose source.

Verify the ROS side before enabling the policy:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 param get /effort_drive accept_torque
ros2 topic hz /odom
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic hz /scan
ros2 topic hz /terrain_scan
```

If you publish `/nino_rl/terrain_preview` directly, check that topic instead
of `/terrain_scan`. Confirm motor direction, joint order, effort units, encoder
signs, emergency stop, and torque limits with the physical wheels restrained.
The simulation's limits are not a substitute for the real motor limits.

## 3. Start inference

The installed policy defaults to the bundled ZIP, its matching config, CPU
inference, and the [6 m straight path](../src/nino_rl/config/path.yaml) starting
at `(0, 0)` in the `odom` frame. Set the robot's odometry origin and heading to
match that path, or supply a straight path YAML in the same waypoint format.
The saved policy was trained for this straight-route task, not arbitrary Nav2
paths. Do not change the saved `ppo.yaml` policy/observation parameters without
checking model compatibility.

In a third terminal, with the ROS and workspace setup sourced:

```bash
ros2 run nino_rl policy_node --device cpu \
  --path /absolute/path/to/your/path.yaml
```

Omit `--path` only when the bundled path and odometry origin are correct. For
a separately copied checkpoint, use `--model /path/nino_ppo_final.zip` and
`--config /path/ppo.yaml` together. Do not add `--use-sim-time` on the real Pi.
Only one policy, evaluator, teleop controller, or other `/cmd_vel` publisher
should command the robot at a time.

This checkpoint has training success of 630/1,571 episodes in its final resumed
segment. No held-out evaluation or physical robot run is recorded for it. Run
the [simulation evaluation](../src/nino_rl/models/completed_train/README.md)
before a physical test, then validate cautiously with the wheels restrained
and a working emergency stop. Physical behavior is an open validation item.
