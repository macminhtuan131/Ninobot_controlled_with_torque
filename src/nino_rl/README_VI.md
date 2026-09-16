# RL điều khiển mô-men hai bánh cho Nino

**Cập nhật reward/policy:** [công thức và hướng dẫn train](REWARD_POLICY_UPDATE.md).
Giữ 300 đầu vào/3 action; phải train mới, không resume checkpoint trước bản vá này.

**Bản đang chạy: [thuật toán v2](ALGORITHM_V2.md).** Policy có 3 action và
300 đầu vào (5 frame × 60). Reward đang dùng ở `nino_rl/control_v2.py`, cấu hình
`reward_v2`. Các mô tả 54 đầu vào/2 action phía dưới là ghi chép v1; không dùng
checkpoint v1 để resume/deploy v2. Hướng dẫn build, train và đánh giá bản mới
nằm trong tài liệu v2 ở trên.

Package `nino_rl` cung cấp ba chương trình hoàn chỉnh:

- `train`: huấn luyện PPO trong Gazebo trên dãy gờ cáp với curriculum;
- `evaluate`: chạy test xác định và ghi chỉ số ra CSV/JSON;
- `policy_node`: chạy policy đã học, nhận đường `/plan` từ Nav2 và xuất mô-men trái/phải.

Mục tiêu là đi qua địa hình gồ ghề mà vẫn bám đường, hạn chế lật/kẹt, giảm thời gian, dừng êm đúng endpoint và không đi ra ngoài hành lang. Policy sử dụng đồng thời IMU, encoder, LiDAR, odometry và các waypoint nhìn trước.

> **An toàn:** model mới khởi tạo hoặc chưa được đánh giá có thể phát mô-men bất ngờ. Chỉ chạy trong Gazebo cho đến khi đạt tiêu chí test. Khi thử trên robot thật phải kê bánh/giới hạn mô-men, có nút dừng khẩn cấp và người giám sát.

## Phần lấy từ hai bài báo và phần đã điều chỉnh

### Từ *Reinforcement Learning for Wheeled Mobility on Vertically Challenging Terrain*

- dùng PPO cho action liên tục;
- curriculum từ đoạn dễ đến đoạn có nhiều gờ khó hơn;
- reward tiến về đích, phạt đứng yên dưới 1 cm/0,1 s, phạt roll/pitch quá 30 độ và phạt timeout theo quãng đường còn lại;
- đánh giá success, thời gian và roll/pitch.

Bài báo xuất vận tốc/góc lái và dùng SWAE cho bản đồ cao độ. Nino là robot vi sai 2WD, vì vậy implementation này **đổi action thành mô-men bánh trái/phải** và không giả vờ có bản đồ cao độ khi robot chỉ có LiDAR 2D.

### Từ *Reinforcement Learning-Based Control of a 4-Wheel Independent Steering Mobile Robot for Robust Path Tracking in Outdoor Environments*

- 9 điểm nhìn trước ở `0.5, 1, 2, 3.5, 5, 7.5, 10, 12.5, 15 m`;
- reward progress/alignment/lateral/heading/smoothness và các hệ số trong paper;
- mạng policy/value `[256, 256]`, ELU, `gamma=0.99`, learning rate `5e-4`;
- domain randomization cho độ bám, trễ motor 10–40 ms, noise/bias/dropout quan sát.

Bài báo dùng robot 4WIS, Isaac Lab và 4096 môi trường song song. Nino dùng Gazebo Harmonic với một mô phỏng vật lý, nên action 4WIS được đổi thành 2 mô-men bánh. RTX chạy phần cập nhật neural network; Gazebo/ROS vẫn chủ yếu chạy CPU. Con số 59 triệu bước/18 phút của paper **không phải** tốc độ có thể kỳ vọng từ một Gazebo đơn.

Trong implementation hiện tại, độ bám `[0.3, 1.0]` được mô phỏng bằng hệ số truyền mô-men ngẫu nhiên theo episode; không thay đổi hệ số contact của SDF khi simulator đang chạy. `long_hall.sdf` có 29 gờ cáp cao khoảng 1,6–4,4 cm; vùng `1,30 x 1,00 m` quanh spawn `(0,0,0)` được chừa phẳng, sau đó mật độ gờ tăng trên toàn hành lang. Curriculum tăng dần quãng đường chứa gờ. Paper dùng đường Catmull–Rom kín bán kính khoảng 15 m; hành lang Nino không đủ rộng, nên code sinh đường Catmull–Rom mở ngẫu nhiên với biên ngang tối đa 0,6 m và tăng biên theo curriculum.

### Observation và action của Nino

Observation có 54 giá trị hữu hạn, được chuẩn hóa. Callback `/imu/data` đọc và chuẩn hóa quaternion trước khi tạo observation:

| Thành phần | Số chiều |
|---|---:|
| 9 waypoint `(x,y)` trong hệ robot | 18 |
| `cos/sin` sai số hướng | 2 |
| vận tốc thẳng và yaw rate | 2 |
| encoder velocity trái/phải | 2 |
| IMU quaternion `(x,y,z,w)`, angular velocity XYZ, linear acceleration XYZ | 10 |
| khoảng cách nhỏ nhất trong 5 vùng LiDAR | 5 |
| action trước đó | 2 |
| vận tốc dài/góc tham chiếu từ Nav2 | 2 |
| hướng đơn vị tới waypoint cục bộ | 2 |
| sai số ngang, khoảng cách waypoint và đích cuối | 3 |
| roll/pitch tường minh | 2 |
| hệ số trượt bánh trái/phải | 2 |
| thời gian waypoint còn lại và cờ Nav2 hợp lệ | 2 |

Roll/pitch vẫn được tính từ quaternion để dùng cho điều kiện rollover và reward, nhưng policy nhận trực tiếp toàn bộ 10 giá trị IMU. Hướng mong muốn là tiếp tuyến của đường tại điểm gần robot nhất; `cos/sin` sai số hướng tránh gián đoạn tại ±π.

Reward IMU bổ sung gồm:

- `imu_stability`: thưởng **tiến độ ổn định**, lớn nhất khi roll/pitch nhỏ, tốc độ quay quanh X/Y nhỏ và gia tốc thay đổi ít;
- `imu_vibration`: phạt rung dựa trên gyro X/Y và độ thay đổi vector gia tốc giữa hai bước;
- `impact`: phạt bậc bốn theo `(|a|-g)` đã chuẩn hóa và chặn biên; giữ độ nhạy với cú xóc nhưng không để một spike IMU phá hỏng toàn bộ PPO rollout;
- `body_rate`: phạt tốc độ roll/pitch đã chuẩn hóa và chặn biên;
- `attitude`: chỉ phạt phần roll vượt `0,20 rad` hoặc pitch vượt `0,30 rad`;
- `action_rate` và `smoothness`: phạt cả vi phân bậc một và bậc hai của torque residual;
- `direction`: thưởng tiến độ theo `max(0, cos(e_heading))`, lớn nhất khi robot tiến đúng hướng mong muốn;
- `direction_correction`: thưởng khi sai số hướng giảm sau quyết định torque và phạt khi sai số tăng;
- `wrong_way`/`reverse`: phạt quay ngược hướng đường hoặc chạy lùi;
- `wrong_direction_failure`: phạt 100 điểm khi attempt bị hủy vì đi sai hướng liên tục;
- `rollover`: phạt mạnh riêng khi roll/pitch vượt ngưỡng 30 độ.
- `endpoint_motion`: trong 2 m cuối, phạt vận tốc thẳng/yaw-rate vượt ngưỡng để policy học giảm tốc và dừng êm;
- `success`: thưởng 100 điểm khi xe thật sự ổn định tại endpoint;
- `early_finish`: thưởng thêm tuyến tính theo thời gian về sớm, tối đa 100 điểm. Với deadline thưởng 90 s, về ở 72 s được thêm 20 điểm; từ 90 s trở đi không còn thưởng sớm.

Stable bonus và direction reward đều được nhân với quãng đường tiến lên nên robot không thể nhận thưởng chỉ bằng cách đứng yên. Sai hướng vẫn bị `heading` phạt riêng. Không phạt trực tiếp gyro-Z trong stability reward vì yaw-rate là cần thiết khi bám đường cong.

Một episode chỉ thành công khi đồng thời thỏa tất cả điều kiện mặc định:

- cách endpoint không quá `0,45 m`, lệch ngang không quá `0,25 m`, sai hướng không quá `12°`;
- vận tốc thẳng không quá `0,20 m/s`, yaw-rate không quá `0,30 rad/s`;
- roll và pitch đều không quá `10°`.

Episode dừng ngay khi thành công, đi sai hướng liên tục, collision, lệch đường hoặc rollover; episode bị truncate khi hết `120 s`. `target_finish_seconds: 90` là mốc tính thưởng về sớm, còn `max_episode_seconds: 120` là deadline cứng. Hai giá trị đều chỉnh được trong `config/ppo.yaml`.

### Một episode là một attempt điều chỉnh hướng

Chiều tiến mong muốn không bị gắn cứng vào trục `+X`: nó là thứ tự waypoint từ đầu đến cuối trong path huấn luyện hoặc `/plan` của Nav2. Mỗi chu kỳ 10 Hz thực hiện:

1. đọc quaternion, angular velocity và linear acceleration từ IMU;
2. đọc tốc độ encoder bánh trái/phải;
3. tính sai số hướng so với tiếp tuyến của path và tạo observation 54 chiều;
4. PPO chọn hai torque mới, sau đó reward đo xem quyết định đó làm xe thẳng lại hay lệch thêm.

Nếu gờ làm lệch xe trong thời gian ngắn, attempt vẫn tiếp tục để policy học phục hồi. Attempt chỉ bị hủy và reset khi sai số hướng từ `60°` hoặc vận tốc lùi từ `0,10 m/s` tồn tại liên tục ít nhất `0,5 s`. Khoảng `1 s` đầu episode được miễn kiểm tra để Gazebo ổn định. Các ngưỡng tương ứng là `wrong_direction_*` trong `config/ppo.yaml`.

Mỗi lần reset, training node lần lượt reset trạng thái model, gọi Gazebo `set_pose` để đặt model `nino` về chính xác `x=0, y=0, z=0, roll=0, pitch=0, yaw=0`, rồi reset odometry và bộ nhớ controller về zero. Không đặt gờ nào trong vùng spawn phẳng.

Action là `Box([-1,-1], [1,1])`, nhân với giới hạn mặc định `0,5 N.m`, rồi phát torque residual lên `/wheel_torque_commands` theo thứ tự `[trái, phải]`. `effort_drive` cộng residual này với torque nền của Nav2, sau đó vẫn giới hạn cứng tối đa `12 N.m`, slew-rate và timeout 0,25 s.

## 1. Cài đặt Ubuntu 24.04 + ROS 2 Jazzy, không cần NVIDIA

Các dependency ROS:

```bash
sudo apt update
sudo apt install python3-venv python3-colcon-common-extensions \
  ros-jazzy-ros-gz ros-jazzy-gz-ros2-control \
  ros-jazzy-ros-gz-interfaces ros-jazzy-tf2-ros
```

Tạo môi trường Python nhưng giữ quyền nhìn thấy `rclpy` của ROS:

```bash
cd /home/tue/ninorobot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.7"
python -m pip install -r src/nino_rl/requirements.txt
```

Kiểm tra PyTorch đang chạy CPU:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả đúng phải có `False` ở cuối. Không chạy `check_cuda`, vì lệnh đó cố ý báo lỗi trên máy không có NVIDIA.

Luôn build bằng `python -m colcon` khi `.venv` đang active. Nhờ vậy executable Python trong `install/` dùng đúng interpreter có PyTorch/SB3, đồng thời vẫn import được ROS Jazzy.

## 2. Huấn luyện

Đóng mọi phiên `sim.launch.py` cũ trước, sau đó mở hai terminal.

Terminal 1 — mô phỏng headless và service reset episode:

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 launch nino_rl training_sim.launch.py headless:=true
```

Launch đợi controller bánh xe, bốn luồng sensor, `/odom` và TF
`odom -> base_footprint` sẵn sàng rồi mới khởi động Nav2. Dòng
`nino_sim_readiness: Waiting for ...` trong vài giây đầu là bình thường; không
nhấn `Ctrl-C` trong lúc này.

Muốn **xem xe train trực tiếp**, dùng Gazebo GUI ở Terminal 1 (train sẽ chậm hơn headless):

```bash
ros2 launch nino_rl training_sim.launch.py headless:=false rviz:=true
```

Terminal 2 — train 500.000 bước bằng CPU:

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run nino_rl train --timesteps 500000 --output rl_runs --device cpu
```

Chạy thử pipeline ngắn trước khi train dài:

```bash
ros2 run nino_rl train --timesteps 4096 --checkpoint-every 2048 --check-env
```

Mỗi run tạo thư mục theo thời gian gồm:

```text
rl_runs/YYYYMMDD-HHMMSS/
├── checkpoints/nino_ppo_*.zip
├── monitor.csv
├── nino_ppo_final.zip
├── ppo.yaml
└── tensorboard/
```

Theo dõi reward/loss trong lúc train ở một terminal riêng:

```bash
cd /home/tue/ninorobot
source .venv/bin/activate
python -m tensorboard.main --logdir /home/tue/ninorobot/rl_runs
```

Mở `http://localhost:6006`. Cảnh báo `TensorFlow installation not found` không ảnh hưởng vì PPO ghi log bằng PyTorch. Theo dõi GPU trong **một terminal khác**:

```bash
watch -n 1 nvidia-smi
```

Trong TensorBoard, nhóm `reward_terms/*` cho biết từng phần thưởng/phạt; nhóm `episode/*` hiển thị success, đúng hạn, thời gian, khoảng cách endpoint, sai số ngang, max tilt và độ rung IMU.

Train tiếp từ checkpoint:

```bash
ros2 run nino_rl train \
  --resume rl_runs/YYYYMMDD-HHMMSS/checkpoints/nino_ppo_250000_steps.zip \
  --timesteps 250000
```

Các stage curriculum mặc định kết thúc ở `x=4, 10, 18, 30 m`. Chỉnh torque, curriculum, reward, PPO và randomization trong `config/ppo.yaml`. Sau khi chỉnh phải build lại hoặc truyền đường dẫn source:

```bash
ros2 run nino_rl train --config /home/tue/ninorobot/src/nino_rl/config/ppo.yaml
```

## 3. Test model

Giữ Terminal 1 đang chạy `training_sim.launch.py`, rồi chạy:

```bash
ros2 run nino_rl evaluate \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --episodes 10
```

Test độ bền với noise/trễ/độ bám ngẫu nhiên:

```bash
ros2 run nino_rl evaluate \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --episodes 25 --randomized
```

Output trong `rl_runs/evaluation/` gồm từng attempt ở CSV và summary JSON: success rate, tỷ lệ hủy do sai hướng, số lượng từng nguyên nhân kết thúc, tỷ lệ thành công trước 90 s, thời gian thành công, khoảng cách cuối tới endpoint, vận tốc cuối, sai số ngang, roll/pitch, max tilt, angular-rate XY và độ thay đổi gia tốc IMU trung bình. Nên chỉ chuyển sang robot thật khi test deterministic và randomized đều ổn định, không rollover/collision và sai số phù hợp giới hạn cơ khí của bạn.

## 4. Chạy policy trong Gazebo

Mở simulation có GUI/RViz:

```bash
ros2 launch nino_rl training_sim.launch.py headless:=false rviz:=true
```

Chạy policy với đường thẳng mặc định trong `config/path.yaml`:

```bash
ros2 run nino_rl policy_node \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --use-sim-time
```

Dừng bằng `Ctrl-C`; node luôn phát torque `0,0` khi shutdown, và `effort_drive` tự timeout nếu policy chết.

## 5. Kết nối Nav2/Linorobot2

`policy_node` nghe `nav_msgs/msg/Path` trên `/plan`. Nếu Nav2 phát path trong frame `map`, node dùng TF để đổi sang `odom`. Khi chưa có `/plan`, nó dùng `config/path.yaml`.

Sau khi Linorobot2 + Nav2 đã chạy và có `/odom`, `/imu/data`, `/joint_states`, `/scan`, TF `map -> odom`, chạy:

```bash
source /opt/ros/jazzy/setup.bash
source /home/tue/ninorobot/.venv/bin/activate
source /home/tue/ninorobot/install/setup.bash
ros2 run nino_rl policy_node \
  --model /duong/dan/nino_ppo_final.zip \
  --plan-topic /plan --device cpu
```

Không thêm `--use-sim-time` trên robot thật. Có thể remap nếu Nav2 của bạn phát global plan ở tên khác, ví dụ `--plan-topic /plan_smoothed`.

Nav2 controller server vẫn có thể phát `/cmd_vel`; lệnh torque từ policy có ưu tiên trong adapter khi được phát đều ở 10 Hz. Progress checker và goal checker của Nav2 vẫn theo `/odom`. Trước khi chạy thật, kiểm tra:

```bash
ros2 topic hz /plan
ros2 topic hz /odom
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic hz /scan
ros2 topic echo /wheel_torque_applied
```

## Xử lý lỗi nhanh

- `Missing /world/long_hall/control`: phải launch bằng `training_sim.launch.py`, không chỉ `sim.launch.py`.
- thiếu `/world/long_hall/set_pose`: build lại workspace và khởi động lại `training_sim.launch.py` để nạp bridge reset pose mới.
- thiếu `/reset_wheel_odometry`: build/source lại workspace sau thay đổi `nino_control`.
- báo lỗi yêu cầu CUDA: build/source lại sau khi cập nhật config hoặc thêm `--device cpu` vào lệnh train/evaluate.
- thiếu sensor khi reset: kiểm tra simulation chỉ chạy một phiên và bốn topic `/odom`, `/imu/data`, `/joint_states`, `/scan` đang có dữ liệu.
- model báo shape khác `(54,)`: model đó được train bằng observation version cũ/khác, không được dùng trực tiếp; hãy train lại.
- robot rung mạnh: giảm `max_wheel_torque_nm`, tăng phạt `smoothness_weight`, rồi train/evaluate lại; không chỉnh model trong lúc đang chạy thật.

## Nguồn

- Tong Xu, Chenhui Pan, Xuesu Xiao, *Reinforcement Learning for Wheeled Mobility on Vertically Challenging Terrain*, arXiv:2409.02383.
- Hyoseok Lee, Hyun-Min Joe, *Reinforcement Learning-Based Control of a 4-Wheel Independent Steering Mobile Robot for Robust Path Tracking in Outdoor Environments*, Sensors 2026, 26, 1761.
- [Cài PyTorch và kiểm tra CUDA](https://docs.pytorch.org/get-started/locally/)
- [Stable-Baselines3: custom environment](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [Gazebo Harmonic và ROS 2](https://gazebosim.org/docs/harmonic/ros2_integration/)
