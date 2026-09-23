# Nino: PPO residual đi thẳng và đánh giá quỹ đạo

Nhánh này điều khiển **Nino AMR dẫn động vi sai có bánh caster hỗ trợ** trong
ROS 2 Jazzy / Gazebo Harmonic. Mô hình có hai bánh chủ động, hai bánh caster thụ
động, IMU, encoder bánh xe và LiDAR **2D**. Đây không phải robot con lắc ngược
hai bánh không có điểm tựa. Nav2 được tắt trong thử nghiệm này. Pipeline là:
tham chiếu `/cmd_vel` đi thẳng trực tiếp → PI tốc độ bánh xe + PPO residual có
giới hạn → bộ điều khiển effort. PPO cũng điều chỉnh tỷ lệ tham chiếu vận tốc
tiến. Policy PPO cung cấp hiệu chỉnh lái vốn thường do bộ bám hình học đảm
nhiệm: residual vi sai của hai bánh phải giữ sai số lệch ngang và sai số hướng
gần 0 trước khi chuyển động tiến được nhận toàn bộ reward, kể cả khi vượt qua
gờ dây cáp đã cấu hình.

Quy trình mới mặc định sử dụng NVIDIA CUDA và không tự động chuyển sang CPU.
Vật lý Gazebo và ROS vẫn dùng CPU; đây là một world Gazebo, không phải mô phỏng
Isaac Lab song song bằng GPU. Mỗi lần chỉ để một tiến trình
train/evaluate/policy kết nối với world đó.

Các tiến trình mô phỏng Nino tự động dùng ROS domain 77 với phạm vi khám phá chỉ
trên localhost, ngăn máy hoặc simulator khác phát `/clock` xung đột. Để dùng
`ros2 topic` hoặc `ros2 service` thủ công, trước tiên chạy
`export ROS_DOMAIN_ID=${NINO_ROS_DOMAIN_ID:-77}` và
`export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` trong shell đó.

- [Reward, policy và các quyết định kỹ thuật](src/nino_rl/RL_IMPROVEMENTS.md)
- [Chẩn đoán timeout và xác thực revision 27](docs/RL_TRAINING_DIAGNOSIS_20260922.md)
- [Reward ưu tiên đích, policy và curriculum địa hình (revision 30)](docs/RL_GOAL_FIRST_REVISION_28.md)
- [Hướng dẫn nhanh bằng tiếng Việt](src/nino_rl/README_VI.md)
- [Bài viết nghiên cứu tổng quan bằng tiếng Việt](docs/BAO_CAO_NGHIEN_CUU_RL_NINO.md)
- [Tài liệu tham khảo mô phỏng và phần cứng](docs/HARDWARE_REFERENCE.md)

Training contract hiện tại là revision 30. Vì reward khoảng cách tới đích,
địa hình kích thước đầy đủ, khám phá theo từng action, cách tính thời gian hoàn
thành vật lý, marker đích không nhìn thấy bởi sensor, cân bằng reward, cờ thử
thách có thể vượt qua, hình học địa hình, preview hướng xuống và rolling hazard
curriculum đã thay đổi, không thể resume checkpoint cũ; hãy bắt đầu lần chạy
mới. Checkpoint mới giữ trạng thái rolling curriculum nên resume không còn đặt
lại độ khó hazard. Model cũ vẫn có thể inference với config tương ứng. Model cũ
54 input/2 action không tương thích. Dự án không cung cấp trọng số pretrained
hoặc kết quả cải thiện hiệu năng đã đo.

## 1. Chuẩn bị Ubuntu và GPU NVIDIA

Dùng Ubuntu **24.04**, Python hệ thống **3.12** và máy có GPU NVIDIA. Chạy các
lệnh sau trên chính máy sẽ huấn luyện, không phải laptop khác không có GPU. Bỏ
qua bước cài driver nếu `nvidia-smi` đã hoạt động.

```bash
sudo apt update
sudo apt install git curl locales software-properties-common \
  python3-venv python3-pip build-essential ubuntu-drivers-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Sau khi khởi động lại, `nvidia-smi` phải liệt kê đúng GPU cần dùng (ví dụ RTX
4080). Khắc phục lỗi driver/Secure Boot trước khi cài dependency RL. Chỉ cài
`nvidia-utils` không tạo ra kernel driver hoạt động. Xem
[hướng dẫn driver NVIDIA của Ubuntu](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/).

## 2. Cài ROS 2 Jazzy, Gazebo, Nav2 và RViz

Nếu `/opt/ros/jazzy/setup.bash` đã tồn tại, bỏ qua bước thiết lập repository
ROS. Nếu chưa có, bật Universe và cài gói apt-source chính thức của ROS:

```bash
sudo add-apt-repository universe
NINO_ROS_APT_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/nino-ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${NINO_ROS_APT_VERSION}/ros2-apt-source_${NINO_ROS_APT_VERSION}.noble_all.deb"
sudo dpkg -i /tmp/nino-ros2-apt-source.deb
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop ros-dev-tools
```

Cài các dependency của robot:

```bash
sudo apt install python3-rosdep python3-colcon-common-extensions \
  ros-jazzy-ros-gz ros-jazzy-ros-gz-interfaces ros-jazzy-gz-ros2-control \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-controller-manager \
  ros-jazzy-effort-controllers ros-jazzy-joint-state-broadcaster \
  ros-jazzy-ros2controlcli ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-localization ros-jazzy-rviz2 \
  ros-jazzy-tf2-ros
source /opt/ros/jazzy/setup.bash
```

Nếu cách đóng gói repository thay đổi, xem
[hướng dẫn cài Jazzy chính thức](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
`ros-jazzy-ros-gz` cung cấp tích hợp Gazebo tương thích; không cài Gazebo
Classic cho dự án này.

## 3. Lấy mã nguồn và áp dụng patch

Với checkout mới:

```bash
cd ~
git clone --branch add_rl https://github.com/macminhtuan131/Ninobot_controlled_with_torque.git ninorobot
cd ~/ninorobot
```

Với checkout đã có, vào thư mục dự án và kiểm tra `git status`. Bảo toàn các
thay đổi cục bộ trước khi chuyển branch. Patch được cung cấp được tạo dựa trên
commit `d015b8239648c64974e5badd041aa87f3a5a0fdc` của branch `add_rl`.

```bash
git rev-parse HEAD
git apply --check ~/Downloads/ninobot-rl-improvements.patch
git apply ~/Downloads/ninobot-rl-improvements.patch
```

Dùng đúng đường dẫn tải xuống thực tế. Nếu bước kiểm tra báo xung đột, hãy dừng
và xử lý khác biệt branch/thay đổi cục bộ; không ép áp dụng patch hoặc xóa công
việc của bạn. Bỏ qua patch nếu checkout đã có các thay đổi này.

Khởi tạo rosdep một lần (bỏ qua `init` nếu đã khởi tạo), sau đó cài các package
cần thiết cho workspace huấn luyện:

```bash
sudo rosdep init
rosdep update
rosdep install --from-paths src/nino_description src/nino_control src/nino_rl \
  src/linorobot2/linorobot2_navigation --ignore-src --rosdistro jazzy -r -y
```

## 4. Tạo môi trường Python và cài PyTorch CUDA

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r src/nino_rl/requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

CUDA 12.6 wheel ở trên là ví dụ cụ thể từ
[ma trận phiên bản PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).
Hãy dùng NVIDIA driver tương thích. Wheel `+cpu` hoặc
`torch.version.cuda == None` là sai với quy trình này. Không cần CUDA toolkit hệ
thống chỉ để dùng các wheel này. `--system-site-packages` cần thiết cho module
Python của ROS; `python3-venv` tránh lỗi `ensurepip is not available`.

## 5. Build bằng Python interpreter trong venv

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install --packages-up-to nino_rl
source install/setup.bash
ros2 run nino_rl check_cuda
```

Kết quả mong đợi là `CUDA khả dụng: True`, đúng GPU và phép nhân ma trận CUDA
thành công. Luôn dùng **`python -m colcon` bên trong venv** để entry point Python
đã cài có thể import PyTorch. Không di chuyển venv sau khi build.

Trong mỗi terminal mới, chạy bốn dòng sau trước tiên:

```bash
cd ~/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
```

## 6. Khởi động và kiểm tra mô phỏng

Đặt endpoint và bộ điều khiển tiến trong `src/nino_rl/config/ppo.yaml`:

```yaml
goal_tolerance_m: 0.10
navigation:
  start_pose: [0.0, 0.0, 0.0]
  goal_pose: [6.0, 0.0, 0.0]
  straight_speed_m_s: 0.75
  minimum_approach_speed_m_s: 0.03
  goal_slowdown_distance_m: 0.50
```

Robot chỉ thành công khi tâm của nó nằm trong vòng tròn endpoint bán kính 10 cm
và hướng lệch không quá 12 độ so với hướng đường đi. Chỉ vượt qua mặt phẳng đích
không còn được tính là thành công, nên đi quá đích hoặc lệch ngang đều thất bại.
Lệnh trực tiếp luôn có `angular.z = 0`; policy residual có thể dùng hiệu chỉnh vi
sai giữa hai bánh để chống trôi khi bám tham chiếu thẳng. Dây cáp curriculum ở
`x=4 m`, chừa 2 m để phục hồi và đo độ trôi trước đích 6 m. Với
`headless:=false`, Gazebo hiển thị đích bằng đĩa, cột và cờ xanh sáng. Marker chỉ
để hiển thị, không thể va chạm với robot hoặc LiDAR.

Huấn luyện bắt đầu với một hazard thích nghi. Sau năm episode thành công liên
tiếp sẽ thêm một hazard. Mỗi thất bại ngắt chuỗi và việc thêm hazard xóa chuỗi.
Giới hạn là tám hazard thích nghi cộng một dây cáp của phase. Kết quả rolling và
số hazard hiện tại được lưu trong mọi checkpoint thường, cuối và bị gián đoạn.
Các mảng giống ổ gà, gờ bậc thấp, dây cáp ngang ngắn và mô phỏng rãnh đường được
randomize mỗi episode trong vùng huấn luyện dài bốn mét. Tâm mỗi vật được lấy
mẫu đều giữa hai đường tâm bánh trái/phải (xấp xỉ ±17,1 cm). Hình học cột 35 cm
được xem là vật cản chặn đường cho lập kế hoạch và không bao giờ được lấy mẫu
hay nhận reward như thử thách điều khiển bánh. Hình học thích nghi giữ nguyên
chiều cao ở mọi phase; riêng lòng ổ gà giữ độ chênh 30 mm. Dây cáp phase thay
đổi đường kính và góc xiên; dấu của góc nghiêng được randomize mỗi episode, kể
cả dây ±5 độ ở phase dễ nhất.

Rãnh đường dùng hai vai dốc nhẹ cao 20 mm quanh kênh ngang rộng 100 mm ở cao độ
sàn. Cấu trúc này tạo độ tụt tương đối thật cho bánh xe. Gazebo không thể trừ
một hố runtime khỏi sàn nhà, nên đây là mô phỏng mặt đường nâng cục bộ chứ không
phải hình học nằm dưới world.

Tốc độ là một action liên tục được học: PPO điều chỉnh tham chiếu thẳng 0,75 m/s
từ 0 đến 100% ở mỗi bước policy 0,1 giây. Một quạt LiDAR nhìn xuống 20 Hz cung
cấp preview mới và có thể triển khai về gờ dây cáp/địa hình thấp, bổ sung cho
LiDAR an toàn phía trước. Đến đích trước mục tiêu 15 giây nhận bonus tỷ lệ;
trong khi các term va đập, trượt, torque, timeout và đường đi ngăn chiến lược
“luôn chạy hết tốc độ” trở thành lựa chọn hữu ích duy nhất. TensorBoard ghi tỷ
lệ tốc độ trung bình/tối thiểu/tối đa và tốc độ mặt đất trung bình mỗi episode.

Hai action còn lại là effort residual chung và vi sai. Chúng ánh xạ song ánh
sang torque bánh trái/phải có giới hạn, nên PPO có thể tăng một bánh, giảm bánh
kia hoặc thay đổi độc lập cả hai trong khi vòng PI 500 Hz giữ ổn định vận tốc
bánh yêu cầu. Gia tốc không phải chế độ actuator cạnh tranh; nó là kết quả vật
lý của torque giới hạn, mục tiêu vận tốc và giới hạn gia tốc/slew của bộ điều
khiển.

Thử thách có thể vượt qua dùng cờ reward privileged một lần và không thêm vào
observation của actor. Cờ yêu cầu vùng quét của một bánh chủ động giao với khu
vực, thay vì chỉ để gờ đi dưới gầm. Đây là ước lượng giao cắt hình học, không
phải sensor lực tiếp xúc vật lý. Đi vào thử thách nhận tối đa +2 trong toàn
episode, vượt cạnh trước cục bộ nhận tối đa +8, và đạt đích thành công nhận thêm
tối đa +90 tỷ lệ theo phần đã vượt. Các tổng này được chia theo số thử thách của
episode nên thêm hazard không làm tăng return tối đa. Dao động không thể thu
cùng cờ hai lần; bước có va chạm, lật, timeout, lệch đường hoặc sai hướng không
nhận bonus thử thách mới. Báo cáo episode và TensorBoard gồm số lượng/tỷ lệ đã
chọn và đã vượt trong các trường `challenge_*`.

Ở phase 1, ổ gà randomize là lòng tròn 0,60 m, sâu tương đối 30 mm, có dốc vào/
ra mượt khoảng 12,5 độ và mép trước khoảng 3 mm. Kích thước cho phép bánh caster
16 mm đi vào nhưng vẫn đòi hỏi effort truyền động hữu ích. Gazebo không thể trừ
một hình random khỏi sàn phẳng, nên đây là mô phỏng lòng chảo nâng dạng vành
khuyên, không phải hố thật dưới sàn; muốn có hố đào thật phải thay sàn bằng mesh
đục sẵn hoặc heightmap.

Terminal A:

```bash
ros2 launch nino_rl training_sim.launch.py headless:=true
```

Để debug trực quan, dừng launch đó rồi khởi động lại với `headless:=false`.
Không chạy đồng thời hai bản. Launch huấn luyện này khởi động effort controller,
sensor và dịch vụ reset Gazebo. Nó không khởi động Nav2, AMCL, map server hoặc
planner. Launch sẽ từ chối chạy nếu cùng world Gazebo đã hoạt động, tránh nhiều
publisher `/clock` và IMU làm hỏng huấn luyện lockstep. Đợi simulator và các
controller. Terminal B:

```bash
ros2 control list_controllers
ros2 param get /effort_drive accept_torque
ros2 topic hz /imu/data
```

Cả `joint_state_broadcaster` và `wheel_effort_controller` phải ở trạng thái
active; `accept_torque` phải là `True`. Dừng lệnh đo tần số topic bằng Ctrl-C.
Tần số IMU danh định là 50 Hz theo thời gian mô phỏng. Lệnh train thực hiện
preflight 12 điểm bắt buộc, gồm kiểm tra actuation, trước khi học. Dùng
`training_sim.launch.py`, không dùng description launch độc lập, cho RL.

## 7. Smoke test, sau đó train phase 1

Giữ Terminal A hoạt động. Terminal B:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 4096 --check-env
```

Đây chỉ là kiểm tra pipeline, chưa đủ để học policy hữu ích. Khi lệnh chạy tốt,
bắt đầu lần train thật:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 500000 \
  --checkpoint-every 25000
```

Lệnh in thư mục chạy, ví dụ `rl_runs/20260917-123456-123456/`. Thư mục chứa
`ppo.yaml`, metadata phần mềm/thiết bị, `monitor.csv`, `episodes.jsonl`, log
TensorBoard, `checkpoints/nino_ppo_*_steps.zip` và `nino_ppo_final.zip` khi hoàn
tất. Mỗi rollout dài 2048 bước nên SB3 có thể vượt số bước yêu cầu để hoàn thành
rollout. Huấn luyện đã hoạt động khi thu thập các bước này; bảng optimizer PPO
đầu tiên chỉ xuất hiện sau rollout đầu. Theo dõi dòng `EPISODE END` trực tiếp.
Lần chạy 500000 bước cần ít nhất 50000 giây mô phỏng ở 10 Hz, cộng overhead
reset/update; thời gian thực phụ thuộc thông lượng Gazebo.

Terminal C:

```bash
tensorboard --logdir rl_runs --port 6006
```

Đánh giá học bằng cách xem đồng thời `episode/success`,
`episode/timeout_failure` và `episode/endpoint_distance_m`.
`episode_reward/*` là tổng reward toàn episode, còn `reward_terms/*` là trung
bình mỗi bước. Timing đã sửa phải giữ `episode/max_clock_error_seconds` gần 0
và `episode/max_motion_sensor_lag_seconds` dưới 0,04 giây. Record riêng được lưu
trong `episodes.jsonl`, gồm lý do kết thúc và curriculum phase. Revision 28 cần
train mới; model cũ học với timing và semantics reward khác. Khởi động lại
simulator sau khi build để cả hai mask LiDAR loại marker đích trang trí.

Progress hiện đo mức giảm khoảng cách endpoint thực, kể cả lệch ngang và đi quá
đích. Tham chiếu tốc độ giữ thấp sau khi qua đích; overshoot dọc lớn hơn 0,30 m
kết thúc tác vụ chỉ-tiến với `goal_missed` và penalty thất bại. Khi ở trong vòng
đích nhưng sai hướng, hệ thống vẫn giữ tham chiếu tiến nhỏ để PPO còn quyền lái
vi sai. Độ lệch chuẩn Gaussian ban đầu là `[0.20, 0.12, 0.05]` cho tốc độ/torque
chung/lái; tất cả vẫn có thể học. Theo dõi `policy/std_*` và
`episode/goal_missed_failure` cùng các kết quả khác.

Mở `http://localhost:6006`. Xem đồng thời success, path RMSE/P95, completion,
heading RMSE, impact RMS, torque, reward term và PPO KL/entropy. Gazebo dùng
huấn luyện lockstep: mỗi action tiến đúng 50 bước vật lý 2 ms (0,1 giây mô
phỏng), sau đó đợi sensor và phản hồi torque mới trong khi pause. Vì vậy thời
gian optimizer không thể tiêu tốn deadline nhiệm vụ. Chỉ thấy CUDA dùng bộ nhớ
không chứng minh việc học thành công.

Ctrl-C hoặc lỗi transport runtime sẽ lưu `nino_ppo_interrupted.zip` nếu đã có
model; rollout chưa xong bị bỏ khi resume. Sửa lỗi transport trước khi tiếp tục.
Chỉ resume lần chạy v2 tương thích với config đã lưu:

```bash
ros2 run nino_rl train --device cuda --phase 1 --timesteps 500000 \
  --config rl_runs/YOUR_RUN/ppo.yaml \
  --resume rl_runs/YOUR_RUN/checkpoints/nino_ppo_25000_steps.zip
```

`--timesteps` là số bước train bổ sung. Model đã lưu nhưng chưa từng update vẫn
chưa được huấn luyện. Seed train được đặt bởi `seed` trong YAML; dùng config
riêng với seed khác để kiểm tra khả năng lặp lại.

## 8. Tự động đánh giá baseline và PPO

Dừng trainer nhưng giữ nguyên mô phỏng huấn luyện. Chạy các lệnh sau **lần
lượt**, dùng đường dẫn model/config thật được in khi train:

```bash
ros2 run nino_rl evaluate_baseline --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml --output rl_runs/baseline-p1
ros2 run nino_rl evaluate --device cuda --phase 1 --episodes 20 --seed 10000 \
  --config rl_runs/YOUR_RUN/ppo.yaml --model rl_runs/YOUR_RUN/nino_ppo_final.zip \
  --output rl_runs/ppo-p1
```

Mỗi lệnh in thư mục báo cáo có timestamp. Nó chứa `summary.json`,
`episodes.csv`, snapshot config/metadata và, với mỗi episode hoàn tất:
`episode-001/trajectory.csv`, `actual.csv`, `reference.csv`, `metrics.json` và
`trajectory.png`. `trajectory.csv` gồm
`time_s,x_m,y_m,yaw_rad,frame_id`, có thể đọc trực tiếp bằng pandas, bảng tính
hoặc MATLAB. PNG chồng tham chiếu đường thẳng cố định với quỹ đạo robot đo được.
Metric tự động gồm path/cross-track RMSE có trọng số thời gian, P95/sai số đường
lớn nhất, heading RMSE, sai số endpoint, completion, đi lùi, success, timing,
slip, torque và tác động IMU. Đánh giá chưa hoàn tất được đánh dấu và không thể
so sánh như lần chạy đầy đủ.

```bash
ros2 run nino_rl compare_evaluations \
  --baseline rl_runs/baseline-p1/BASELINE_STAMP/summary.json \
  --candidate rl_runs/ppo-p1/PPO_STAMP/summary.json \
  --output rl_runs/comparison-p1.json
```

So sánh **success trước**, rồi đến sai số/độ êm và thời gian hoàn thành thành
công. Robot đứng yên hoặc thất bại có thể có RMSE thấp. Báo cáo tách tất cả
episode khỏi episode thành công. Bộ so sánh kiểm tra phase, seed, số lượng, chế
độ perturbation và config tác vụ. Seed giống nhau tái tạo các lần lấy mẫu địa
hình; chúng không làm thực thi ROS/Gazebo bất đồng bộ trở nên giống bit tuyệt
đối.

`--randomized` trên **cả hai** evaluator kiểm tra perturbation residual/sensor
toàn cường độ. Đây không phải thay đổi vật lý về ma sát/khối lượng. Baseline
nhận torque residual bằng 0, kể cả trong kiểm thử randomized.

## 9. Tự động chạy cả sáu phase dây cáp

Mỗi phase có đúng một dây cáp tại 4 m. Huấn luyện đi từ dễ đến khó; chỉ đường
kính cáp và trị tuyệt đối của góc quyết định độ khó phase. Dấu của góc khác 0
được randomize để policy không thiên vị bánh nào.

Chạy một job huấn luyện liên tục với cấu hình mặc định:

```bash
ros2 run nino_rl train --device cuda --timesteps 600000 --check-env
```

Bỏ `--phase`: chỉ định nó sẽ cố ý khóa lần chạy ở một phase. Bước 0–99.999 dùng
phase 6; các phase 5, 4, 3 và 2 sau đó nhận 100.000 bước mỗi phase; phase 1 bắt
đầu ở bước 500.000. Địa hình đổi ở lần reset episode đầu tiên sau ranh giới nên
không bao giờ ngắt một lần vượt đang diễn ra. Sau khi lịch kết thúc, phase giữ ở
1. Resume bảo toàn tiến độ bước tuyệt đối và chuỗi thành công hazard. Kiểm tra
API không làm lịch tiến lên. PPO có thể kết thúc rollout cuối sau budget bước.

| Phase | Độ khó | Đường kính | Trị tuyệt đối của góc |
|---|---|---:|---:|
| 1 | Khó nhất | 15 mm | 45 độ |
| 2 | Rất khó | 13 mm | 36 độ |
| 3 | Khó | 11 mm | 27 độ |
| 4 | Trung bình | 9 mm | 18 độ |
| 5 | Dễ | 7 mm | 9 độ |
| 6 | Dễ nhất | 5 mm | 0 độ |

Domain randomization mặc định tắt để so sánh phase chỉ dựa trên kích thước và
góc. Chỉ dùng `--randomized` khi đánh giá nếu cần kiểm tra robustness.

Ngưỡng ban đầu đề xuất: ít nhất 19/20 lần thành công trên tập giữ lại, không lật
hoặc va chạm, sai số đường P95 chấp nhận được (ví dụ <0,25 m cho hành lang này)
và độ êm không giảm đáng kể so với baseline. Đây là tiêu chí chấp nhận đề xuất,
không phải kết quả đo. Dùng các seed bổ sung, khác seed phát triển, cho kiểm thử
cuối.

```bash
ros2 run nino_rl train --device cuda --timesteps 300000 \
  --config rl_runs/YOUR_RUN/ppo.yaml \
  --resume rl_runs/YOUR_RUN/checkpoints/nino_ppo_300000_steps.zip
```

Đánh giá mỗi phase với cùng phase/seed cho baseline và PPO. Lịch phase dựa trên
số bước và không đợi đánh giá thành công. Giữ nhiều checkpoint; checkpoint cuối
không tự động là tốt nhất. Đánh giá chúng trên cùng seed phát triển, sau đó kiểm
thử model được chọn bằng seed mới. Không chạy callback đánh giá trên cùng world
đang hoạt động khi trainer thu thập rollout.

## 10. So sánh với đường lý tưởng hoặc quỹ đạo có thời gian của bạn

CSV tham chiếu hình học dùng `x_m,y_m,frame_id`. Trace thực dùng
`time_s,x_m,y_m,yaw_rad,frame_id`; yaw không bắt buộc. Mọi tọa độ phải cùng
frame. Các file tự động xuất đã đúng định dạng này.

```bash
ros2 run nino_rl trajectory_metrics \
  --actual rl_runs/ppo-p1/PPO_STAMP/episode-001/actual.csv \
  --reference rl_runs/ppo-p1/PPO_STAMP/episode-001/reference.csv \
  --mode path --output rl_runs/path-score.json --plot rl_runs/path-overlay.png
```

Để so với quỹ đạo lý tưởng theo lịch, thêm cột `time_s` tăng nghiêm ngặt cho
tham chiếu rồi dùng `--mode timed --max-gap 0.5`. `position_rmse_m` so sánh vị
trí nội suy tại cùng thời gian mô phỏng; không dịch timestamp, căn chỉnh rigid
hoặc ngoại suy. Gốc đồng hồ phải khớp. Thời gian xuất là tương đối với mẫu
odometry đầu; metric episode ghi `clock_origin_sim_s` để đổi timestamp simulator
tuyệt đối. Khoảng trống dài, timestamp trùng và frame không khớp đều bị từ chối.
Coverage được báo cáo để lần chạy kết thúc sớm không che thời gian tham chiếu
chưa quan sát. Chọn giới hạn gap theo sampling rate, không dùng nó để che mất
dữ liệu.

Đường thẳng hình học không có timestamp mong muốn, nên điểm tự động mặc định là
**path RMSE**, không phải timed position RMSE. Pose là wheel odometry được
localization biến đổi, không phải ground truth ngoài. Với nghiên cứu độ chính
xác vật lý, hãy xuất motion-capture hoặc pose ground-truth simulator đã biến
đổi đúng. Tiến độ theo đoạn gần nhất không rõ ràng trên đường tự cắt; dùng timed
scoring cho trường hợp đó. Không nối nhiều đồng hồ episode vào một CSV.

Các công cụ scoring cũng chạy được khi không có ROS:

```bash
PYTHONPATH=src/nino_rl python -m nino_rl.trajectory_metrics --help
```

## Khắc phục sự cố và phạm vi

| Triệu chứng | Cách xử lý |
|---|---|
| Thiếu `ensurepip` | Cài `python3-venv`, tạo lại venv chưa hoàn chỉnh |
| Torch `+cpu` / CUDA false | Cài CUDA wheel trong cùng venv; kiểm tra driver; chạy `check_cuda` |
| Không import được `rclpy` | Source Jazzy và dùng Python hệ thống 3.12 + `--system-site-packages` |
| `ros2 run` không import được Torch | Build lại package Python khi venv active bằng `python -m colcon` |
| Preflight torque bằng 0 | Dùng training launch, effort controller active và `accept_torque=True` |
| Timeout độ phủ IMU | Kiểm tra timestamp `/clock`, `/imu/data` và tải; thời gian đợi giới hạn xử lý race khi truyền nhưng không tạo mẫu giả |
| Lỗi contract khi resume | Dùng run/config tương ứng sau cập nhật hoặc bắt đầu run mới |
| Không có subscriber `/cmd_vel` | Build lại `nino_control`, khởi động lại mô phỏng và chạy lại preflight |

Patch này không bổ sung SWAE, bản đồ địa hình 3D, ESKF, Isaac Lab, asymmetric
privileged critic hoặc cân bằng dốc chưa được kiểm chứng. Input preview địa hình
hiện có giờ được điều khiển bởi LiDAR mô phỏng nhìn xuống. Triển khai phần cứng
cần producer tương đương đã hiệu chuẩn; chỉ mô phỏng không thể xác lập hiệu năng
transfer. Xem design note để biết lý do lựa chọn và reward chính xác.
