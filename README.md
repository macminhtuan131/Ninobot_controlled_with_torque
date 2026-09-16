# Nino Robot — ROS 2 Jazzy, Gazebo Sim, sensors, and torque control

This ROS 2 workspace contains the Nino differential-drive AMR description,
Gazebo Sim Harmonic world, wheel-torque control stack, IMU, wheel encoders, and
2D lidar. The robot spawns at world pose `0 0 0` and can be driven through
`/cmd_vel` or direct torque commands.

The `add_rl` branch also includes CPU-first PPO training, evaluation,
and a Nav2-aware wheel-torque policy. See the complete Vietnamese guide:
[src/nino_rl/README_VI.md](src/nino_rl/README_VI.md).

The current Nav2-guided architecture, 54-value observation, six-phase terrain
curriculum, mandatory preflight, baseline comparison, and preserved-map notes
are documented in [src/nino_rl/README.md](src/nino_rl/README.md).

The simulator uses this control path:

```text
/cmd_vel ──> effort_drive PI loop ──> /wheel_effort_controller/commands
                                             │
                                             v
                              JointGroupEffortController
                                             │
                                             v
                          left/right joint effort interfaces
                                             │
                                             v
                                  gz_ros2_control + Gazebo
```

## Workspace layout

```text
ninorobot/
├── README.md
└── src/
    ├── nino_description/
    │   ├── launch/sim.launch.py
    │   ├── meshes/
    │   ├── urdf/nino.urdf.xacro
    │   └── worlds/
    │       ├── long_hall.sdf
    │       └── flat_world.sdf
    └── nino_control/
        ├── config/controllers.yaml
        ├── config/effort_drive.yaml
        ├── nino_control/effort_drive.py
        ├── nino_control/kinematics.py
        └── test/test_kinematics.py
```

`nino_description` owns geometry and simulation bring-up. `nino_control` owns
controller configuration, torque control, differential-drive kinematics, and
wheel odometry.

## Install dependencies

ROS 2 Jazzy binary packages target Ubuntu 24.04. Ubuntu 22.04 requires a Jazzy
source build or an Ubuntu 24.04 ROS container.

```bash
sudo apt update
sudo apt install ros-jazzy-ros-gz ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-teleop-twist-keyboard \
  ros-jazzy-gz-ros2-control ros-jazzy-controller-manager \
  ros-jazzy-effort-controllers \
  ros-jazzy-joint-state-broadcaster ros-jazzy-ros2controlcli ros-jazzy-rviz2 \
  python3-colcon-common-extensions
```

You can also let `rosdep` resolve package dependencies. Initialize it once,
then install everything declared by this workspace:

```bash
sudo rosdep init
rosdep update
cd /home/tue/ninorobot
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
```

If `rosdep init` reports that its sources list already exists, skip that first
command and continue with `rosdep update`.

## Build and run

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch nino_description sim.launch.py
```

For RL training, build the Python packages with the project venv interpreter
as described in [the Nav2-guided RL guide](src/nino_rl/README.md); otherwise
the installed training command cannot import PyTorch or Stable-Baselines3.

To open the sensor view and print all three sensors in the launch terminal:

```bash
ros2 launch nino_description sim.launch.py rviz:=true sensor_monitor:=true
```

RViz displays the lidar scan, robot, encoder odometry, and TF tree. The terminal
monitor prints IMU vectors, left/right encoder position and velocity, lidar
sample count, nearest range, and the received rate for every source.

For a server-only run:

```bash
ros2 launch nino_description sim.launch.py headless:=true
```

The robot spawns at `x=0`, `y=0`, `z=0`, and `yaw=0`. RL episode reset also
sets this physical Gazebo pose explicitly before resetting wheel odometry and
controller state. Do not start a second copy of the launch file while one is
already running. `Ctrl-C` requests a clean Gazebo server stop and should finish
without a false process-crash error.

## Verify ros2_control

With the simulation running:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 topic hz /joint_states
```

The expected active controllers are `joint_state_broadcaster` and
`wheel_effort_controller`. The claimed command interfaces are:

```text
left_wheel_joint/effort
right_wheel_joint/effort
```

The controller's input is a `std_msgs/msg/Float64MultiArray` ordered as
`[left_wheel_joint, right_wheel_joint]`.

## View IMU, encoder, and lidar data

The simulated sensors use standard ROS 2 messages and Linorobot2 topic names:

| Sensor | Topic | Type | Frame/rate |
|---|---|---|---|
| IMU | `/imu/data` | `sensor_msgs/msg/Imu` | `imu_link`, 50 Hz |
| Wheel encoders | `/joint_states` | `sensor_msgs/msg/JointState` | left/right joints, 500 Hz |
| 2D lidar | `/scan` | `sensor_msgs/msg/LaserScan` | `laser`, 10 Hz |

View each complete message:

```bash
ros2 topic echo /imu/data
ros2 topic echo /joint_states
ros2 topic echo /scan
```

Or view all three as a compact live summary:

```bash
ros2 run nino_control sensor_monitor --ros-args -p use_sim_time:=true
```

Check that data is flowing at the expected rates:

```bash
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic hz /scan
```

Plot the IMU angular velocity in real time:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_plot rqt_plot -e \
  /imu/data/angular_velocity/x:y:z
```

`QLayout::removeWidget: Cannot remove a null widget` is a harmless startup
warning from `rqt_plot` 1.4.5 on Jazzy. If the plot window opens, it can be
ignored. If no curves appear, first verify the sensor with
`ros2 topic echo /imu/data --once` and `ros2 topic hz /imu/data`.

To capture an exact 30-second interval, run:

```bash
timeout --signal=INT 30s ros2 bag record -o imu_30s /imu/data
```

In `/joint_states`, `position` is the encoder angle in radians and `velocity`
is radians per second. Match values to `left_wheel_joint` and
`right_wheel_joint` using the same array index in `name`, `position`, and
`velocity`.

## Drive with `/cmd_vel`

The default `effort_drive` node converts desired base velocity into wheel-speed
targets, closes a PI loop using measured `/joint_states`, and sends bounded
torque to `JointGroupEffortController`.

Keyboard control:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

Constant forward command:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.0}}"
```

The `/cmd_vel` watchdog applies zero target velocity after 0.5 seconds without
a new command.

## Control wheel torque directly

The safe direct-torque input is `/wheel_torque_commands`. Values are N·m in
left/right order, are clamped to the configured limit, and expire after 0.25 s.

Apply `0.8 N·m` to both wheels:

```bash
ros2 topic pub --rate 20 /wheel_torque_commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.8]}"
```

The two entries are independent. For example, apply `0.8 N·m` to the left
wheel and `0.4 N·m` to the right wheel:

```bash
ros2 topic pub --rate 20 /wheel_torque_commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.4]}"
```

Use a negative value to reverse a wheel. The order is always
`[left_wheel_joint, right_wheel_joint]`.

Stop the publisher with `Ctrl-C`. The timeout returns both commands to zero.
Monitor the torque actually sent by the adapter:

```bash
ros2 topic echo /wheel_torque_applied
```

This topic reports the safe applied command, which can be lower than the
requested torque while the torque ramp or wheel-speed guard is active.

To inspect `JointGroupEffortController` without the safety adapter, launch with:

```bash
ros2 launch nino_description sim.launch.py start_effort_drive:=false
```

Then publish to the controller directly:

```bash
ros2 topic pub --rate 20 /wheel_effort_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.8, 0.8]}"
```

The raw interface bypasses the watchdog, torque ramp, and software speed guard;
use it only with the robot restrained or for a short controller check. It holds
its last received effort. Always send zero before stopping the raw publisher:

```bash
ros2 topic pub --once /wheel_effort_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0]}"
```

Limit the simulated wheel torque below the URDF maximum when launching:

```bash
ros2 launch nino_description sim.launch.py max_wheel_torque:=3.0
```

The hard URDF and ros2_control interface limit is `±12 N·m`. The default
`/cmd_vel` speed loop is separately limited to `±2 N·m`; direct torque mode can
use the full configured hardware limit. PI gains, acceleration limits, watchdog
times, and odometry settings are kept in
`src/nino_control/config/effort_drive.yaml`.

Direct effort is slew-limited to `10 N·m/s`. A `12 rad/s` software wheel-speed
guard removes torque that would accelerate an overspeed wheel, while allowing
opposite braking torque. The URDF's `24 rad/s` emergency limit remains enabled
as a final independent backstop.

## Nav2 and Linorobot2 compatibility

No Nav2 nodes are included or launched yet. The default simulation already
provides the mobile-base interfaces Nav2 needs:

- `/cmd_vel` — `geometry_msgs/msg/Twist`
- `/odom` — `nav_msgs/msg/Odometry`
- `odom -> base_footprint` TF
- `base_footprint -> base_link` and wheel TF from `robot_state_publisher`

Linorobot2's Jazzy EKF expects raw wheel odometry on `/odom/unfiltered` and
publishes filtered `/odom` plus the odometry TF. Use:

```bash
ros2 launch nino_description sim.launch.py linorobot2_mode:=true
```

In this mode Nino publishes `/odom/unfiltered` and does not publish
`odom -> base_footprint`, avoiding duplicate `/odom` publishers or TF sources
when Linorobot2's `ekf_filter_node` runs. The IMU and lidar remain available on
`/imu/data` and `/scan`. Use the `jazzy` branch of
[Linorobot2](https://github.com/linorobot/linorobot2/tree/jazzy), select its 2WD
base configuration, and keep its standard `cmd_vel`, `odom`,
`base_footprint`, `base_link`, `imu_link`, and `laser` names.

Start Nino in the Linorobot2-compatible mode:

```bash
ros2 launch nino_description sim.launch.py linorobot2_mode:=true rviz:=true sensor_monitor:=true
```

Then run Linorobot2's Jazzy EKF (from a terminal where Linorobot2 is built and
sourced):

```bash
ros2 run robot_localization ekf_node --ros-args \
  --params-file "$(ros2 pkg prefix --share linorobot2_base)/config/ekf.yaml" \
  -p use_sim_time:=true -r odometry/filtered:=/odom
```

Do not run Linorobot2's full Gazebo launch at the same time as Nino's simulation;
that launch creates another robot and another Gazebo instance. Run its EKF and
navigation/SLAM nodes against Nino's standard topics instead.

## Moving from Gazebo to Xiaomi CyberGear hardware

`gz_ros2_control/GazeboSimSystem` is simulation-only. On the physical robot,
replace that hardware plugin with a ros2_control `SystemInterface` that:

1. Converts `left_wheel_joint/effort` and `right_wheel_joint/effort` from N·m
   into CyberGear CAN torque commands.
2. Reports measured joint position, velocity, and effort in SI units.
3. Enforces the motor, gearbox, electrical, thermal, and emergency-stop limits
   for the actual installation.

The controller and high-level topic contract can remain unchanged, so
`/cmd_vel`, direct torque commands, odometry, Linorobot2, and later Nav2 do not
need to know whether the joint backend is Gazebo or CyberGear CAN hardware.

## Physics and collision model

The supplied CAD masses, centers of mass, and inertia tensors are retained.
Left/right inertial and contact properties are symmetric; the effective wheel
separation uses the measured tyre contact centers. The chassis collision uses
the same `base_link.STL`, origin, and scale as its visual, so their surfaces are
exactly aligned. Rolling surfaces use exact measured cylinders for stable
contact. The caster bracket collision uses its CAD mesh in the same frame as
the visual mesh, keeping the open fork clear.
The modeled moving mass is `4.6888672492 kg`; its aggregate center of mass in
`base_footprint` is approximately `[0.056018, 0.0, 0.127546] m`. Every inertia
tensor is positive definite and satisfies the rigid-body triangle conditions.

The default closed hall is 34 m long and 4 m wide. Its 29 cable bumps
vary from 16–44 mm diameter and −22° to +24°. Every angled cable length is
`4.0 / cos(angle)`, so it reaches both inner wall faces. A marked
`1.30 x 1.00 m` area centered on `(0,0,0)` remains cable-free for deterministic
spawning and resets; the nearest cable centers are at `x=-1.10 m` and
`x=0.90 m`.
# Ninobot_controlled_with_torque
# Ninobot_controlled_with_torque
# Ninobot_controlled_with_torque
# Ninobot_controlled_with_torque
