# Mixed terrain and result checking

The arena is `nino_description/worlds/long_hall.sdf`: a closed 34 m by 4 m
hall. Its preview contains the existing 29 cables plus three potholes and
three speed bumps. Training/evaluation resets replace this preview with the
selected curriculum; seeing the preview does not mean phase 1 uses obstacles.

## Geometry and curriculum

| Phase | Episode terrain |
| --- | --- |
| 1 | Flat floor |
| 2 | One fixed cable |
| 3 | One cable with random position |
| 4 | Random cable position and angle |
| 5 | Random cable position, angle and diameter |
| 6 | Four cables, three potholes and three speed bumps by default |

Edit `terrain_curriculum.mixed_obstacles` in `config/ppo.yaml` to change
counts or size ranges. Setting `enabled: false` restores the cable-only
phase 6. Old saved configs without this section also use cable-only phase 6;
copy the new section into a saved config if resuming a run with mixed terrain.
The reward, observation/action sizes and checkpoint contract are unchanged.

Mixed obstacles occupy separate longitudinal slots between X=4 and X=27 m.
Cables have 12–32 mm diameter and up to 10 degrees yaw. Potholes are rectangular
recesses, 10–30 mm deep, with vertical lips and a solid bottom. They are made
by tiling the collision floor around apertures, not placing a dark box on a
solid floor. Bumps are 10–30 mm high and 0.40–0.70 m long; 24 box slices
approximate a rounded cosine profile. Mixed-training bumps span 3.8 m;
preview bumps span 1.2 m to fit between the existing angled cables.
These are rigid geometric approximations, not deformable cables or asphalt.

The replacement floor covers X=[2,29]. Permanent end pads remain during
replacement, including the normal spawn at (0,0) and goal at (30,0).
Custom reset poses must fit an end pad with a 0.65 m clearance. A world
version marker prevents accidentally running the new code against the old
solid floor. Rebuild and restart Gazebo after installing this change.

## Build, inspect and train

From the repository root, use the ROS/venv setup from the root README:

```bash
colcon build --symlink-install --packages-select nino_description nino_rl
source install/setup.bash
ros2 launch nino_rl training_sim.launch.py headless:=false
```

This displays the preview course. In another sourced terminal, use a short
phase-6 run to check live contacts/reset behavior before a long run:

```bash
ros2 run nino_rl train --device cuda --phase 6 --timesteps 4096 --check-env \
  --config src/nino_rl/config/ppo.yaml
```

This smoke run is not a trained policy. For learning, first complete and
evaluate the earlier phases, then resume a compatible revision-4 checkpoint
with `--phase 6`, the updated terrain config, and an appropriate training
budget. The mandatory preflight still runs.

## Existing tools for checking results

| Tool | What it already provides |
| --- | --- |
| `evaluate_baseline` | Nav2/PI baseline episodes |
| `evaluate` | Deterministic PPO checkpoint evaluation |
| `compare_evaluations` | Baseline/PPO deltas with compatibility checks |
| `trajectory_metrics` | Score exported trajectories against geometric/timed references |
| TensorBoard | Training reward terms and episode performance curves |
| `preflight`, `check_cuda` | Startup and GPU diagnostics, not policy quality scores |

Stop the trainer, keep `training_sim.launch.py` running, and execute the
evaluations sequentially. Replace `YOUR_RUN` with an actual model directory.
Use the same updated config, phase, seed and episode count for both:

```bash
ros2 run nino_rl evaluate_baseline --phase 6 --episodes 20 --seed 10000 \
  --config src/nino_rl/config/ppo.yaml --output rl_runs/baseline-mixed
ros2 run nino_rl evaluate --device cuda --phase 6 --episodes 20 --seed 10000 \
  --config src/nino_rl/config/ppo.yaml \
  --model rl_runs/YOUR_RUN/nino_ppo_final.zip --output rl_runs/ppo-mixed
ros2 run nino_rl compare_evaluations \
  --baseline rl_runs/baseline-mixed/BASELINE_STAMP/summary.json \
  --candidate rl_runs/ppo-mixed/PPO_STAMP/summary.json \
  --output rl_runs/comparison-mixed.json
tensorboard --logdir rl_runs --port 6006
```

Replace both timestamp placeholders with the directories printed by evaluation.
Add `--randomized` to both evaluation commands for full-strength configured
control/sensor perturbations. Terrain itself is sampled in phase 6 even without
this flag. Matching seeds reproduce geometry, not asynchronous simulator timing.

Reports contain `summary.json`, `episodes.csv`, config/metadata snapshots and
actual/reference trajectory CSVs. This change additionally writes
`terrain_layouts.json` (positions, sizes and types for each reset), adds terrain
generator revision metadata, and rejects comparisons of different actual layout
hashes. New reports use schema 2; rerun both controllers rather than comparing
old reports to new ones. These checks do not fingerprint external Gazebo binaries
or hand-edited world files: use the same running simulator for both runs.

Read success/completion and termination reasons first, then path RMSE, heading
error, duration, IMU acceleration peaks/RMS, wheel slip and torque. Early failure
can misleadingly lower time or average error. The existing path metrics use
localized wheel odometry; they are not independent ground-truth measurements.

## Verification

Pure Python tests cover reproducible layouts across 100 seeds, separated mixed
obstacle footprints, floor coverage, true pothole depths, clear permanent pads,
terrain reset service requests and mismatched evaluation reports. Run:

```bash
PYTHONPATH=src/nino_rl:src/nino_control python -m pytest -q src/nino_rl/test
```

The full suite requires ROS Python packages. Without ROS, exclude
`test_navigation_frames.py`; the transport tests in `test_mixed_terrain.py`
use service fakes. Live Gazebo contact behavior and learning performance must
still be checked on the ROS/Gazebo machine; no training improvement is claimed
from geometry tests alone.
