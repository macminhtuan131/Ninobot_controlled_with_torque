# Timeout investigation and training revision 27

## Evidence from the supplied TensorBoard run

Run: `rl_runs/20260922-201214-649628`.
At the inspected 67,584-step logging point, the episode averages were:

| Metric | Value |
|---|---:|
| Success | 0% |
| Episode duration | 20.0 s |
| Final route completion | 77.3% |
| Distance from goal | 1.369 m |
| Mean reported ground speed | 0.455 m/s |
| Mean absolute lateral error | 0.064 m |
| Challenges cleared | 100% |

The inspected last 100 monitor records all had length 200; 66 had positive
return. Positive failed returns alone do not prove that failure is the best
discounted policy. They show why return must be read alongside success.
The speed/elapsed/progress mismatch warranted investigating timing before
increasing speed rewards or relaxing the deadline.

## Confirmed stepping bug

`advance_world()` returned as soon as the Gazebo service acknowledged the
request. The response acknowledges queued work, not completed physics.
A stationary live diagnostic submitted twenty 25-step chunks:

| Observation | Credited time | Observed clock advance |
|---|---:|---:|
| Old code, immediately after the twentieth return | 1.000 s | 0.052 s |
| Old code, after allowing the queue to finish | 1.000 s | 1.000 s |
| Fixed code, immediately after the twentieth return | 1.000 s | 1.000 s |

This allowed action selection, reward sampling and the logical deadline to
run ahead of physics. Wall-fresh sensor callbacks could still describe the
beginning of a queued action. The fix waits for the full clock target after
each chunk and requires motion sensor timestamps within 40 ms of the action
end. Episode clock disagreement above 20 ms aborts instead of becoming
mislabelled training data. Partial clock progress is no longer credited as a
complete chunk, and physics requests are never blindly retried.

## Goal marker visibility

The green goal pole had no collision shape but had ordinary visual visibility.
GPU LiDAR operates through the renderer and applies a visibility mask;
see the [Gazebo Ogre2 GPU rays implementation](https://github.com/gazebosim/gz-rendering/blob/gz-rendering8/ogre2/src/Ogre2GpuRays.cc).
A baseline episode reported collision at 0.207 m from the goal, before the
required 0.10 m arrival circle. The goal decoration now uses visibility bit
0x04, excluded by both lidar sensors while remaining visible in the GUI.
Ordinary obstacles retain their existing visibility and collision behavior.

## Reward changes

| Term | Revision 26 | Revision 27 |
|---|---:|---:|
| Aligned progress per metre | +20 | +10 |
| Challenge entry budget per episode | +10 | +2 |
| Challenge clearance budget per episode | +30 | +8 |
| Challenge bonus paid on successful arrival | +60 | +90 |
| Base success reward | +100 | +100 |
| Timeout penalty | -100 to -50 | -100 |

The total successful challenge bonus stays +100. Entry and clearance still
give immediate encouragement, while most of the bonus requires reaching the
goal. On the default 6 m route, progress and immediate challenge shaping can
contribute at most +70 undiscounted, below even the smallest failure penalty
(-75 for off-path). A regression test exercises the optimistic full-progress,
all-challenges-cleared failure cases and checks that earlier success beats
later success. This budget bound is specific to the default route and does
not establish convergence or policy invariance under discounting.

PPO architecture, learning rate, per-wheel residual action mapping, 20-second
deadline, strict arrival test, 6→5→4→3→2→1 schedule and five-success terrain
promotion remain as configured. There is no evidence here that KL stopping
or GPU capacity caused the low success rate.

## Reading the next run

- `episode/success`, `episode/timeout_failure`, other termination rates and
  `episode/endpoint_distance_m` measure whether the task is being learned.
- `episode_reward/*` contains complete episode reward totals.
  `reward_terms/*` retains per-step means for comparison with old logs.
- `episode/max_clock_error_seconds` should stay near zero, and
  `episode/max_motion_sensor_lag_seconds` must remain at most 0.04 s.
- `episodes.jsonl` records every complete episode, its training step, phase,
  termination reason and reward breakdown, including unfinished PPO rollouts.

Use a fresh run under revision 27; old checkpoints learned under different
action timing and rewards. The interrupted old checkpoint is preserved in
its original run directory. A short live smoke run validates transport and
optimization, not a trained success rate. A longer held-out evaluation is
still needed before claiming better control performance.

## Completed validation

- Rebuilt `nino_control`, `nino_description` and `nino_rl`; restarted Gazebo.
- 133 regression tests passed. The existing optional Matplotlib Axes3D warning
  does not affect the training or these tests.
- The strengthened 12-point live preflight passed.
- Three unsteered baseline episodes completed without transport errors; one
  reached the strict goal circle in 11.7 s. Steering failures remain legitimate
  task failures that the residual policy must learn to correct.
- A controlled visibility test measured minimum lidar range 0.2058 m with the
  masked goal decoration. An otherwise identical ordinary visible pole at the
  same position produced 0.0857 m, below the collision threshold. The temporary
  control object was removed and the robot reset afterward.
- Fresh CUDA smoke run:
  `rl_runs/validation/20260922-222526-220371`, 2,048 steps, one PPO update
  (10 optimization epochs), final checkpoint successfully saved.
- Its 12 completed rollout episodes had seven off-path failures and five
  timeouts, all negative returns. These were collected before the first update;
  zero successes here is not a post-training evaluation.
- Maximum episode clock error: 4.55e-12 s. Maximum motion feedback lag: 0.022 s.
  Each episode's reward components matched its total return within 1.4e-12.
  New termination, timing and episode reward tags were verified in TensorBoard.

The restarted simulator is left running, with the robot stopped. No long
600k-step run was started automatically after validation.
