# Nino residual PPO package

The current implementation uses a 300-value observation history and three
continuous actions: Nav2 speed scale, common torque residual and differential
torque residual. The robot has passive caster supports; it is not a two-wheel
inverted pendulum. CUDA training is the default.

1. Follow [the complete root README](../../README.md) from Ubuntu/driver setup
   through ROS/Gazebo, the Python venv, build, preflight and training.
2. Read [RL_IMPROVEMENTS.md](RL_IMPROVEMENTS.md) for the exact active reward,
   policy, engineering decisions and trajectory metrics.
3. Use `evaluate_baseline` and `evaluate` sequentially with matching phase,
   configuration and seeds. Each exports actual/reference CSVs and automatic
   path RMSE, P95, heading, completion, comfort and timing scores.
4. Use `trajectory_metrics` for custom geometric or timestamped references and
   `compare_evaluations` for compatible baseline/PPO summaries.

Training contract revision 4 requires a fresh run after this patch. Later phase
changes can resume revision-4 checkpoints. The older `ALGORITHM_V2.md` and
`REWARD_POLICY_UPDATE.md` describe historical revisions; current defaults and
commands are in the root README and RL_IMPROVEMENTS.md.
