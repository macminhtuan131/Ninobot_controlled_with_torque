# Bundled completed training run

Source: `rl_runs/20260924-064153-058321`, contract revision 30, final checkpoint
at **1,001,634 cumulative steps**. The latest completed local run was selected;
this does not imply it is the best checkpoint. The weights, exact training
configuration, software metadata, monitor log and episode metrics are bundled
and installed into `share/nino_rl/models/completed_train` by colcon.
`results.json` records provenance, SHA-256 checksums and aggregate metrics.
`episodes.csv` preserves the fields used for those aggregates.

| Training metric | Final resumed segment | Last 100 episodes |
|---|---:|---:|
| Episodes | 1,571 | 100 |
| Successes | 630 (40.10%) | 48 (48.00%) |
| Mean episode path RMSE | 0.148 m | 0.119 m |
| Mean episode path P95 | 0.287 m | 0.228 m |
| Mean endpoint distance | 0.331 m | 0.250 m |
| Mean episode duration (including failures) | 12.77 s | 13.39 s |

These are stochastic training collection statistics from the final resumed
segment, not metrics for every cumulative training step or a held-out evaluation
of the final policy. The segment includes 815 missed goals, 89 off-path exits,
31 timeouts and 6 wrong-direction failures. Hardware transfer and improvement
over the PI baseline have not been measured.

## Use in the project

For a Raspberry Pi and physical robot, follow the
[step-by-step deployment guide](../../../../docs/RASPBERRY_PI_DEPLOYMENT.md).
It covers installation, the required motor hardware interface and sensor
topics, verification, and how to start the policy on the Pi.

Follow the root README for the ROS/venv dependencies and colcon build, then
source `install/setup.bash`. The existing policy node now defaults to these
installed weights, their matching configuration and CPU inference:

```bash
ros2 run nino_rl policy_node
```

This starts control on the connected robot. The robot must already provide
`/odom`, `/imu/data`, `/joint_states`, `/scan`, fresh terrain preview through
`/terrain_scan` or `/nino_rl/terrain_preview`, and effort feedback on
`/wheel_torque_applied`. The existing `nino_control` effort drive must consume
`/cmd_vel` and `/nino_rl/control_command` with torque acceptance enabled and a
working hardware effort-controller interface. The residual policy does not
replace the wheel PI loop or supply a hardware driver.
Use the calibrated sensor conventions expected by the saved configuration,
including gravity in IMU acceleration and the downward terrain preview. See
`nino_rl/ros_interface.py` for topic schemas and preview preprocessing, and
`docs/HARDWARE_REFERENCE.md` in the repository for the hardware context.

The default path is a straight 6 m reference starting at (0, 0) in odom, with
zero initial heading. Supply a matching `--path /absolute/path/to/path.yaml`
for the actual odometry origin; this policy was trained for straight routes.
There is no Nav2 planner in this workflow. The bundled training results do not
meet the root README's proposed 19/20 held-out success gate; validate in
simulation before physical deployment.

For the existing running simulation (stop the trainer/evaluator first):

```bash
ros2 run nino_rl policy_node --use-sim-time
```

Use the same ROS domain/discovery settings as the simulator and ensure physics
is running; the inference node does not advance a paused lockstep world.
Custom weights remain supported with `--model`, `--config`, and `--device`.
Always pair other checkpoints with their own saved configuration.

For reproducible held-out testing, keep the training simulator running and run
these sequentially from the repository root, with the policy node stopped:

```bash
ros2 run nino_rl evaluate_baseline --phase 1 --episodes 20 --seed 10000 \
  --config src/nino_rl/models/completed_train/ppo.yaml --output rl_runs/bundled-baseline
ros2 run nino_rl evaluate --device cpu --phase 1 --episodes 20 --seed 10000 \
  --config src/nino_rl/models/completed_train/ppo.yaml \
  --model src/nino_rl/models/completed_train/nino_ppo_final.zip --output rl_runs/bundled-ppo
```

Compare the generated reports using the root README evaluation instructions.
No held-out or physical robot evaluation is claimed by this bundle.
