# Nino RL — hướng dẫn hiện tại

Hướng dẫn đầy đủ **từ máy chưa cài gì đến train và đánh giá** nằm ở
[README chính](../../README.md). Tài liệu [RL_IMPROVEMENTS.md](RL_IMPROVEMENTS.md)
giải thích reward, policy và các ý tưởng đã chọn/lược bỏ.

Robot trong URDF có **hai bánh chủ động và hai caster đỡ**, không phải robot hai
bánh tự cân bằng không có điểm tựa. Giữ kiến trúc Nav2 + PI + PPO residual;
không áp dụng công thức ZMP dành cho con lắc ngược hoặc giả định có LiDAR 3D.

## Quy trình

1. Ubuntu 24.04, Python hệ thống 3.12; cài driver NVIDIA để `nvidia-smi` thấy GPU.
2. Cài ROS 2 Jazzy, Gazebo Harmonic, Nav2, RViz và rosdep theo README chính.
3. Clone nhánh `add_rl`, kiểm tra rồi áp dụng patch. Không ghi đè thay đổi riêng.
4. Cài `python3-venv`; tạo `.venv` với `--system-site-packages`, cài **PyTorch CUDA**.
5. Build bằng `python -m colcon build --symlink-install --packages-up-to nino_rl`
   khi venv đang active; source `install/setup.bash`.
6. Chạy `ros2 run nino_rl check_cuda`: phải ra `True` và đúng GPU NVIDIA.
7. Terminal A chạy `ros2 launch nino_rl training_sim.launch.py headless:=true`.
8. Terminal B chạy smoke test rồi train phase 1 bằng CUDA.
9. Dừng trainer, giữ simulation; đánh giá baseline và PPO **lần lượt**, cùng
   phase, config, số episode và seed. Xem success trước rồi mới xem RMSE.
10. Chỉ chuyển phase khi đánh giá đạt yêu cầu; phase 2 tiếp tục từ checkpoint
    phase 1, tương tự đến phase 6. Không khởi tạo ngẫu nhiên ở phase 6.

Mỗi terminal mới cần:

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
```

Train smoke / train thật:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 4096 --check-env
ros2 run nino_rl train --device cuda --phase 1 --timesteps 500000
```

Đánh giá (thay `YOUR_RUN` bằng thư mục thật được in lúc train):

```bash
ros2 run nino_rl evaluate_baseline --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml
ros2 run nino_rl evaluate --device cuda --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml --model rl_runs/YOUR_RUN/nino_ppo_final.zip
```

Mỗi episode tự xuất `actual.csv`, `reference.csv`, `metrics.json`; tổng hợp ở
`episodes.csv` và `summary.json`. Có path/cross-track RMSE, P95, heading RMSE,
lỗi endpoint, mức hoàn thành, đi lùi, thời gian, slip, torque và độ xóc IMU.
`compare_evaluations` so hai summary và từ chối điều kiện test không khớp.

Path của Nav2 không có lịch thời gian nên mặc định tính **path RMSE**. Nếu bạn
có quỹ đạo lý tưởng có thời gian, dùng `trajectory_metrics --mode timed` để tính
**position RMSE theo thời gian**. CSV phải cùng `frame_id` và cùng mốc thời gian;
không tự dịch quỹ đạo để che sai số. Xe đứng yên trên đường có thể RMSE bằng 0
nhưng completion bằng 0 — vì vậy không chọn model chỉ theo RMSE.

Policy giữ 5×60 = 300 input, 3 action (speed scale, torque tiến, torque quay),
PPO + history encoder; reward mới thêm phạt sát giới hạn torque. IMU chờ có giới
hạn để xử lý lệch thời điểm giữa `/clock` và dữ liệu sensor, vẫn báo lỗi khi thật
sự mất mẫu. Perturbation tăng dần theo phase. Gazebo vẫn chạy vật lý bằng CPU;
GPU chạy phần mạng neural, không tự biến thành Isaac Lab.

Sau patch phải **train mới**, không resume checkpoint cũ khác contract. Nếu
`.zip` thuộc lần chạy bị lỗi trước khi PPO cập nhật thì nó chưa phải policy đã
học. Phần sửa lỗi thường gặp và lệnh so sánh/đồ thị chi tiết ở README chính.
