# Fine-tune FLUX 3 Action trên Astribot "serving brewed coffee"

Full fine-tune toàn bộ DiT 7B của `flux-3-action-base` (giữ nguyên kiến trúc) trên dataset LeRobot v3.0
`serving_brewed_coffee_lrb_annotated`: 394 episode, 30 Hz, 3 camera 360×640, state/action 16 chiều
(2 tay × 7 khớp + 2 gripper). Chạy vừa **1 GPU 96 GB**, hoặc 2 GPU nếu có GPU trống.

## Cách làm cho vừa 1 GPU

Full fine-tune 7B với AdamW cần khoảng 112 GB (trọng số fp32, gradient và 2 moment), chưa kể 2 EMA,
nên không vừa 96 GB. Script dùng **FSDP2 CPU offload**: bản master fp32, gradient, trạng thái AdamW và EMA
nằm trong RAM (pinned). Từng block được đưa lên GPU để forward/backward ở bf16, còn bước AdamW chạy
trên CPU. Phép toán không đổi so với recipe gốc (master fp32, compute bf16, reduce fp32), kiến trúc
không đổi, không dùng LoRA hay lượng tử hoá.

Đo thực tế trên GPU 0 (RTX PRO 6000 Blackwell Max-Q) với model đúng kích thước:

| | 1 GPU |
|---|---|
| Tốc độ | khoảng 0.52 cửa sổ/s, tức ~61 s cho một update 32 cửa sổ |
| 3000 update (mặc định) | khoảng 2.2 ngày (2 GPU: khoảng 1.1 ngày) |
| GPU | peak khoảng 62 GB (8 cửa sổ/micro-batch) |
| RAM host | khoảng 190 GB (máy có 440 GB, dùng chung với người khác) |
| Mỗi checkpoint | 129 GB, ghi mất khoảng 85 s; mặc định giữ checkpoint mới nhất và trọng số có val/loss thấp nhất (`best/`, 78 GB) |

## Recipe (configs/astribot/train.json)

Theo recipe history của SO-101 (cũng 30 Hz, joint delta), mở rộng cho robot này:

- **Đầu vào**: 8 tick quan sát (2 snapshot ảnh tại t−7 và t), 8 state đo được, 8 lệnh trước đó, câu lệnh
  `serving brewed coffee` (drop 10% khi train).
- **Camera** (layout `droid`, canvas 544×736): `cam_head` ở trên với độ phân giải đầy đủ, `cam_left_wrist`
  và `cam_right_wrist` ở dưới với nửa độ phân giải. Thứ tự này cũng phải dùng y hệt khi chạy trên robot.
- **Đầu ra**: 42 action (1.4 s), robot thực thi 32 rồi lập kế hoạch lại. 14 khớp tay được học dạng
  delta theo từng frame, 2 gripper giữ giá trị tuyệt đối (0–100). Chuẩn hoá q01/q99 tính từ dataset
  (`outputs/astribot/index/statistics.json`).
- **Loss**: flow matching chung cho video và action, gồm video ×1.0 và action ×0.5, gripper có trọng số ×2.
- **Optimizer**: AdamW (0.9, 0.99), weight decay 0.05, clip gradient 1.0, LR trunk 3e-5 / head 1.5e-4,
  trunk đóng băng 200 update đầu (chỉ head mới học), warmup 600, giảm dần về 0 ở 500 update cuối,
  2 EMA (σ_rel 0.10 và 0.05).
- **Validation**: 20 episode cuối (không dùng để train). Mỗi 100 update tính `val/loss` và lấy mẫu action
  trên 8 cửa sổ giống hệt lúc inference (Euler 4 bước, shift 6.93, guidance 3), ra
  `val/sample_action_mse_*`, có MSE theo radian cho từng khớp.

## Chuẩn bị (một lần)

1. **Hugging Face** (weights là repo gated):
   tạo tài khoản ở https://huggingface.co, mở https://huggingface.co/black-forest-labs/flux-3-action-base
   rồi bấm đồng ý license, sau đó tạo token *Read* ở https://huggingface.co/settings/tokens và đăng nhập:
   ```bash
   cd ~/projects/flux-action && .venv/bin/hf auth login
   ```
2. **Weights & Biases**: tạo tài khoản ở https://wandb.ai, lấy API key ở https://wandb.ai/authorize, rồi:
   ```bash
   .venv/bin/wandb login
   ```
   Training không gọi wandb trực tiếp. Launcher bật một tiến trình phụ (`wandb_sync.py`, log ở
   `<output_dir>/wandb_sync.log`) cứ 2 phút đọc `metrics.jsonl` rồi đẩy những gì wandb còn thiếu. Khi gateway
   công ty bắt đăng nhập lại hoặc mất mạng, chỉ biểu đồ wandb tạm đứng; mạng có lại thì nó tự bù đủ.
   Cũng có thể đẩy tay bất cứ lúc nào:
   `.venv/bin/python scripts/astribot/wandb_sync.py --output-dir <output_dir> --project astribot_making_coffee --once`
3. Tải weights (khoảng 25 GB) và index dataset:
   ```bash
   bash scripts/astribot/prepare.sh
   ```

## Train

```bash
cd ~/projects/flux-action
DETACH=1 bash scripts/astribot/train_astri_coffee.sh                    # 1 GPU (GPU 0)
DETACH=1 CUDA_VISIBLE_DEVICES=0,3 bash scripts/astribot/train_astri_coffee_2gpu.sh   # 2 GPU
tail -f outputs/astribot/flux3_astri_coffee_full_ft/train.log
```

- `DETACH=1`: chạy nền bằng `setsid nohup`, nên **mất mạng hay đứt SSH thì vẫn train tiếp**. Không dùng
  `DETACH=1` thì nên chạy trong `tmux`.
- **Resume**: chạy lại đúng lệnh cũ. Script tự lấy checkpoint đầy đủ mới nhất, khôi phục model, optimizer,
  EMA, lịch LR, vị trí dữ liệu và RNG (tiếp tục chính xác), và ghi tiếp vào cùng run W&B. Nếu tiến trình
  crash, launcher tự resume (tối đa `MAX_RESTARTS=20` lần).
- **Dừng an toàn**: `touch outputs/astribot/flux3_astri_coffee_full_ft/STOP`. Run sẽ checkpoint sau update
  hiện tại rồi thoát, launcher không tự khởi động lại.
- **Đổi giữa 1 và 2 GPU**: giữ `GLOBAL_BATCH` như cũ (mặc định 32) thì resume được qua lại.

### Tham số hay chỉnh (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MAX_STEPS` | 3000 | số optimizer update |
| `GLOBAL_BATCH` | 32 | cửa sổ mỗi update (= số GPU × `WINDOWS_PER_GPU` × accumulation) |
| `WINDOWS_PER_GPU` | 8 | micro-batch mỗi GPU (giảm nếu OOM) |
| `LR`, `HEADS_LR_MULT` | 3e-5, 5 | LR trunk và hệ số nhân cho head |
| `FROZEN_STEPS`, `TRUNK_WARMUP_STEPS`, `HEADS_WARMUP_STEPS`, `COOLDOWN_STEPS` | 200, 600, 200, 500 | lịch LR |
| `SAVE_STEPS`, `KEEP_CHECKPOINTS` | 100, 1 | checkpoint |
| `KEEP_BEST` | loss | giữ thêm trọng số (model + EMA, không có optimizer) của checkpoint có `val/<KEEP_BEST>` thấp nhất trong `<output_dir>/best`; `null` để tắt |
| `EVAL_STEPS`, `VAL_SAMPLE_WINDOWS` | 100, 8 | validation |
| `RUN_NAME` | flux3_astri_coffee_full_ft | tên thư mục output và tên run W&B; đổi tên để mở run mới |
| `EXTRA_OVERRIDES` | | trường bất kỳ của config, ví dụ `"policy.augment=true policy.caption_dropout=0.2"` |

Ví dụ: `RUN_NAME=lr5e-5 LR=5e-5 MAX_STEPS=5000 DETACH=1 bash scripts/astribot/train_astri_coffee.sh`.
Trường nào có trong hợp đồng model/dữ liệu (camera, chunk, normalization, loss...) mà bị đổi thì trainer
từ chối resume. Khi đó cần đổi `RUN_NAME` để mở run mới.

## W&B: nên xem gì

- `train/loss`, `train/video_mse`, `train/action_mse`, `train/grad_norm`, `train/lr_trunk`, `train/lr_heads`,
  `train/windows_per_s`, `train/eta_hours`, `train/peak_mem_gb`.
- `val/loss`, `val/action_mse`: loss trên dữ liệu held-out, ở noise cố định nên so được giữa các lần eval.
- `val/sample_action_mse_normalized`: chỉ số chính để chọn checkpoint (sai số của action lấy mẫu như lúc
  inference); `val/sample_action_mse_raw_per_channel/*` cho biết sai số theo radian² của từng khớp, gripper
  theo đơn vị 0–100.

## Sau khi train: export và đánh giá

```bash
bash scripts/astribot/export_eval.sh                          # checkpoint mới nhất, EMA 0.10
STEP=2000 PROFILE=model bash scripts/astribot/export_eval.sh  # model | ema_0p10 | ema_0p05
STEP=best PROFILE=model bash scripts/astribot/export_eval.sh  # checkpoint có val/loss thấp nhất
```

Script tạo package inference bf16 tại `<output_dir>/export/...`, đọc được bằng
`FluxActionPolicy.from_pretrained` / `flux-action infer` (đã kèm sẵn setting sampler), và báo cáo
`eval_val.json` trên 20 episode held-out. Khi chạy trên robot, gọi `policy.select_action(batch)` ở **mỗi**
tick 30 Hz, đưa vào `images.head`, `images.left_wrist`, `images.right_wrist` (uint8, 360×640) và `state`
(16 giá trị cùng thứ tự và đơn vị với dataset), rồi gọi `policy.reset()` khi bắt đầu episode mới.

## Chạy trên robot Astribot S1 (ROS 2)

Toàn bộ phần chạy robot nằm trong `scripts/astribot/` của repo này, không cần `vmo_diffusion_policy_vla`.
ROS 2 Humble chỉ có `rclpy` cho Python 3.10, còn model cần Python 3.12, nên có 2 tiến trình nói chuyện qua
`127.0.0.1:5555` (chỉ trong máy, không pickle):

```text
Robot (ROS 1) ── bridge vm_astribot/run_bridge.sh ── ROS 2, ROS_DOMAIN_ID=25
      camera + joint state ▼                         ▲ lệnh khớp (chỉ khi --execute)
robot_client.py   Python 3.10 hệ thống (/usr/bin/python3): vòng 30 Hz, lịch sử 8 tick, limiter,
                  tracking guard, tư thế sẵn sàng, tự dừng khi mất cảm biến, ghi log episode,
                  quay video, phím h để về HOME
      cửa sổ 8 tick ▼                                ▲ chunk 42 action tuyệt đối
policy_server.py  .venv Python 3.12, GPU: FLUX 3 Action 7B bf16 + video VAE + Qwen3-VL
```

| File | Vai trò |
|---|---|
| `serve_policy.sh` → `policy_server.py` | giữ model trên GPU; nhận `seed` theo từng request |
| `run_robot.sh` → `robot_client.py` | client ROS 2 (source `ros2_env.sh`: ROS 2, Fast-DDS UDP, `astribot_msgs`) |
| `robot_runtime.py` | phần không phụ thuộc ROS: hợp đồng dữ liệu, lịch sử, limiter, tracking, kiểm tra định dạng |
| `robot_ros2.py` | node ROS 2: đọc camera/khớp, luồng lệnh tay 250 Hz, tư thế sẵn sàng/HOME, lệnh giữ, timer quay video |
| `robot_video.py` | quay video thời gian thực: từng camera + `grid.mp4` có bảng trạng thái (port `AstribotVideoRecorder` của vmo) |
| `move_home.sh` → `move_home.py` | đưa robot về HOME (hoặc tư thế sẵn sàng) thật mượt, có `--dry-run` (port `move_astribot_home.sh` của vmo) |
| `robot_log.py`, `replay_log.py` | ghi log episode và `console.log`; phát lại log qua model, so cảnh với dữ liệu train |
| `fake_robot.py`, `test_fake_robot.sh` | robot giả phát lại 1 episode held-out (domain 77, chỉ localhost) |
| `ros2_astribot.yaml`, `fastdds_udp.xml` | topic và cấu hình DDS (giống `vmo_diffusion_policy_vla`) |
| `astribot_wire.py` | giao thức socket (cùng định dạng với client cũ của vmo, client cũ vẫn dùng được) |

Python hệ thống 3.10 đã có sẵn numpy, OpenCV, PyYAML (đi kèm ROS Humble), không cần cài thêm gì.

### Hợp đồng dữ liệu (đã kiểm tra trên dataset)

| Mục | Giá trị |
|---|---|
| State / action | 16 số `arm_left_j0..6, arm_right_j0..6, gripper_left, gripper_right`; tay tính bằng radian, action là vị trí tuyệt đối (≈ state của tick sau) |
| Gripper | state = vị trí ngón đo được (~0 mở, 35–42 khi kẹp vật); action = lệnh 0 (mở) … 100 (đóng) |
| Camera | `images.head`, `images.left_wrist`, `images.right_wrist`: RGB uint8 360×640 (16:9). Robot gửi JPEG, head 1280×720 và wrist 640×360, client thu nhỏ bằng `INTER_AREA` |
| Lịch sử | tick `t` ghi state và ảnh của `t` cùng lệnh đã gửi trước `t`; tick đầu lặp 8 lần. Sau tư thế sẵn sàng, "lệnh trước đó" là chính tư thế sẵn sàng (lệnh đã gửi thật). Dry-run không gửi lệnh nên dùng state đo được thay lệnh (như `select_action`); gripper đang kẹp vật thì giá trị này (~37) khác thang lệnh (100), nên chỉ dry-run khi gripper đang mở |
| Đầu ra | 42 action, robot chạy 32 rồi dự đoán lại; server cộng dồn delta của tay lên lệnh cuối |

Client tự kiểm tra lúc khởi động: tỉ lệ 16:9 của từng camera, state đo được so với khoảng q01..q99 lúc train
(lệch quá 1 khoảng thì từ chối `--execute`, dấu hiệu sai đơn vị hoặc sai thứ tự khớp), metadata của server
(16 chiều, 360×640, đúng thứ tự tên khớp). `tests/test_astribot_runtime.py` kiểm tra lịch sử khớp bit-exact với
`ObservationHistory` của flux.

### Kết quả trên tập held-out (20 episode, `step-2300-model-bf16`, 4 bước × guidance 3)

| Pha (140 cửa sổ) | Chuyển động tay dự đoán / thật (32 bước) | RMSE model / đứng yên | Model tốt hơn đứng yên |
|---|---|---|---|
| đang làm việc (80) | 0.246 / 0.301 rad | 0.047 / 0.078 rad | 78% |
| vừa bắt đầu chuyển động (20) | 0.193 / 0.511 rad | 0.121 / 0.129 rad | 55% |
| 16 frame trước khi người điều khiển bắt đầu (20) | 0.014 / 0.280 rad | 0.054 / 0.050 rad | 5% |
| frame đầu, robot đứng yên (20) | 0.014 / 0.025 rad | 0.014 / 0.005 rad | 0% |

Khi đang làm việc model tốt (sai số giảm 40%). Nhưng **model không tự bắt đầu**: mọi episode train đều đứng yên
~1.6 s (trung vị 49 frame) trước khi người điều khiển bắt đầu, không có dấu hiệu nào trong ảnh, nên với lịch sử
đứng yên model dự đoán gần như đứng yên. Mô phỏng vòng kín với ảnh tĩnh tái hiện đúng lần chạy robot ngày
2026-10-21 (0.010–0.013 rad mỗi chu kỳ, robot gần như đứng im 33 chu kỳ): với **seed cố định** cùng đầu vào cho
cùng đầu ra nên model mắc kẹt; với seed đổi mỗi chu kỳ có chu kỳ bật lên 0.14–0.27 rad. Vì vậy client mặc định
`--seed-mode cycle` (seed = `--seed` + số thứ tự chu kỳ).

Lần chạy đó cảnh cũng khác dữ liệu train: robot đứng xa bàn hơn (bàn nhỏ hơn trong camera đầu, mép bàn gần đáy
ảnh cổ tay, bình rót ở mép phải thay vì trước gripper phải). Với ảnh robot model còn dè dặt hơn (0.015–0.028 rad
so với 0.011–0.157 rad với ảnh dataset ở cùng tư thế). Đặt robot và bàn giống lúc thu dữ liệu rồi kiểm tra bằng
`scene.jpg` (bước B).

Nếu sau khi đặt đúng cảnh mà robot vẫn không khởi động: cách sửa gốc là train lại với đoạn đứng yên đầu episode
được cắt bớt (cửa sổ train bắt đầu từ ~8 frame trước khi tay chuyển động), để model học "đứng yên thì bắt đầu".

### Các bước

```bash
cd ~/projects/flux-action   # /nvme/hieuhm19/projects/flux-action
```

**A. Policy server** (tmux `server`, giữ chạy suốt buổi; cần ~25 GB GPU trống lúc bật, ~18–24 GB khi chạy):

```bash
CUDA_DEVICE=cuda:0 bash scripts/astribot/serve_policy.sh checkpoint/infer/step-2300-model-bf16
# đợi: "Policy server listening on 127.0.0.1:5555"
```

Chế độ nhanh (~1.1 s thay vì ~4.2 s mỗi lần dự đoán; offline không kém hơn):
thêm `--num-inference-steps 2 --guidance-scale 1.0`.

**B. Dry-run và kiểm tra cảnh** (bridge đang chạy, robot không di chuyển):

```bash
bash scripts/astribot/run_robot.sh --max-cycles 3
.venv/bin/python scripts/astribot/replay_log.py --no-model --log ~/Desktop/astribot_logs/flux_action/<thư mục mới nhất>
```

Trong log phải có `State check: every value lies within the training range.`, 3 camera không có
`WARNING: aspect`, 3 dòng `cycle=...`. Mở `scene.jpg`: hàng trên là robot, hai hàng dưới là các episode
held-out bắt đầu gần tư thế đó nhất; bàn, cốc, khay, bình và góc nhìn cổ tay phải trông giống nhau.

**C. Chạy thật, tăng dần** (người đứng cạnh, tay ở nút E-stop):

```bash
bash scripts/astribot/run_robot.sh --execute --prepare-ready-pose --execute-steps 8 --max-cycles 1
bash scripts/astribot/run_robot.sh --execute --prepare-ready-pose --max-cycles 3
bash scripts/astribot/run_robot.sh --execute --prepare-ready-pose            # Ctrl-C để dừng
```

Chương trình hỏi `Enter 's' ...` trước khi gửi lệnh đầu tiên. Mỗi chu kỳ in
`cycle=... seed=... first_step=...rad chunk_arm_motion=...rad grippers=[...]->[...]`; `WARNING`/`CRITICAL
tracking lag` khi robot bám lệnh chậm (3 lần CRITICAL liên tiếp thì tự dừng); `stale Astribot input(s)` khi
camera hoặc khớp ngừng gửi (tự dừng). Khi kết thúc hay lỗi, client gửi lệnh giữ vị trí đã gửi gần nhất.

**Nút về HOME (`h`):** đang chạy thì gõ `h` rồi Enter. Policy dừng ngay, robot giữ lệnh cuối cùng, và client in
bảng kế hoạch (mỗi bộ phận sẽ quay bao nhiêu độ). Gõ `h` + Enter lần nữa thì robot về HOME, chuyển động
cosine trong `--home-duration` giây (mặc định 4 s, giống vmo). Gõ thứ khác thì robot đứng giữ và chương trình
kết thúc. Có `--yes` thì robot về HOME ngay sau lần `h` đầu tiên. Chuyển động bắt đầu từ lệnh đã gửi gần nhất,
không phải từ vị trí đo được, nên tay không bị giật. HOME **mở cả 2 gripper**: nếu robot đang cầm cốc thì lấy
cốc ra trước khi xác nhận. Ở dry-run, `h` chỉ dừng và in bảng, robot không di chuyển.

**D. Về HOME sau khi đã dừng** (`run_robot.sh` đã thoát; không bao giờ để 2 chương trình cùng gửi lệnh):

```bash
bash scripts/astribot/move_home.sh --dry-run        # chỉ in bảng kế hoạch, robot không di chuyển
bash scripts/astribot/move_home.sh                  # hỏi 's' rồi về HOME trong 4 s
bash scripts/astribot/move_home.sh --duration 8     # chậm hơn, khi có khớp phải quay hơn ~30°
bash scripts/astribot/move_home.sh --pose ready     # về tư thế sẵn sàng thay vì HOME
```

**E. Xem lại một lần chạy** (GPU rảnh, server có thể vẫn chạy trên GPU khác):

```bash
.venv/bin/python scripts/astribot/replay_log.py --log ~/Desktop/astribot_logs/flux_action/<run>     --device cuda:1 --seeds 0,1,2,3
```

Với seed đã ghi, replay phải ra đúng chunk của robot (`|replay-logged|max=0.0000rad`); cột `seedN` cho biết
chuyển động phụ thuộc vào nhiễu lấy mẫu đến mức nào.

Mỗi lần chạy có một thư mục `~/Desktop/astribot_logs/flux_action/<thời gian>_<execute|dry>/` (in ra ở dòng
`Run directory:`):

| File | Nội dung |
|---|---|
| `video/grid.mp4` | video thời gian thực 1280×720: camera đầu, bảng trạng thái (mode, pha, chu kỳ, chunk motion, gripper, thời gian), 2 camera cổ tay |
| `video/<camera>.mp4`, `video/session_info.json` | từng camera 640×360 (H.264, 30 fps); thời lượng, số frame, số frame bị bỏ |
| `console.log` | bản sao mọi dòng client in ra, kể cả traceback khi lỗi (không cần `\| tee` nữa) |
| `images.*.mp4`, `ticks.npz`, `cycles.npz`, `windows/`, `meta.json` | log episode để replay (~5 MB mỗi lần dự đoán): mỗi tick một frame, state, lệnh, chunk, seed, ảnh request không nén |

Video quay từ lúc gõ `s` (dry-run: ngay khi sẵn sàng) tới khi chương trình dừng, gồm cả lúc về tư thế sẵn sàng,
lúc chờ server và lúc về HOME. Video lấy frame bằng timer riêng, nên không bị đứng hình trong ~1.1 s server
lấy mẫu, khác với `images.*.mp4` (chỉ có frame ở các tick điều khiển). Khi máy mã hoá không kịp, video bỏ bớt
frame (con số nằm trong `session_info.json`), còn vòng điều khiển không bị ảnh hưởng.

### Test không cần robot

```bash
.venv/bin/python -m pytest -q tests/test_astribot_runtime.py                                     # .venv
/usr/bin/python3 -m pytest -q -p no:cacheprovider --noconftest tests/test_astribot_runtime.py   # Python của ROS
bash scripts/astribot/test_fake_robot.sh --max-cycles 3                                          # dry-run, robot giả
FAKE_EXECUTE=1 bash scripts/astribot/test_fake_robot.sh --prepare-ready-pose --max-cycles 3      # nhánh --execute
(sleep 30; echo h) | FAKE_EXECUTE=1 bash scripts/astribot/test_fake_robot.sh --prepare-ready-pose  # phím h -> HOME
.venv/bin/python scripts/astribot/evaluate_export.py --checkpoint checkpoint/infer/step-2300-model-bf16 \
    --device cuda:1                                                                             # 20 episode held-out
```

Robot giả phát lại episode held-out 376 (bắt đầu ở tư thế sẵn sàng) trên `ROS_DOMAIN_ID=77`, chỉ localhost, và
từ chối chạy ở domain 25. Với `FAKE_EXECUTE=1` nó bám lệnh hoàn hảo; cuối test in số lệnh và bước lớn nhất
của mỗi khớp (tay ≤ 0.032 rad ở 250 Hz, gripper ≤ 20 mỗi tick).

### Tham số `robot_client.py`

| Tham số | Ý nghĩa |
|---|---|
| `--execute` | gửi lệnh thật (không có thì dry-run) |
| `--prepare-ready-pose` | đưa thân/đầu/tay/gripper về tư thế sẵn sàng trước (cần `--execute`) |
| `--execute-steps N` | số action chạy mỗi lần dự đoán (1–32, mặc định 32) |
| `--max-cycles N` | dừng sau N lần dự đoán |
| `--seed-mode cycle\|fixed`, `--seed S` | seed lấy mẫu: đổi mỗi chu kỳ (mặc định) hoặc cố định như export |
| `--server HOST:PORT` | mặc định `127.0.0.1:5555` (server đổi cổng bằng `PORT=...`) |
| `--max-joint-delta`, `--max-gripper-delta` | giới hạn bước mỗi tick (0.15 rad, 20) |
| `--tracking-critical-limit N` | dừng sau N mẫu CRITICAL liên tiếp (mặc định 3) |
| `--allow-out-of-range-state` | cho `--execute` dù state lệch xa khoảng train (chỉ sau khi đã kiểm tra) |
| `--home-duration S` | thời gian về HOME sau phím `h` (mặc định 4 s) |
| `--no-record`, `--no-record-grid`, `--record-fps` | tắt video / chỉ quay từng camera, không có grid / fps của video (mặc định 30) |
| `--log-dir`, `--no-log`, `--no-log-windows` | thư mục chứa các lần chạy (mặc định `~/Desktop/astribot_logs/flux_action`); tắt log episode; bỏ ảnh request không nén |

## Thay đổi trong code

Mọi tính năng mới đều mặc định tắt, nên các config DROID/SO-101 gốc chạy y như cũ (210 test gốc pass,
thêm `tests/training/test_finetune_extras.py`):

- `training/distributed.py`: `cpu_offload` cho FSDP2 và process group cho 1 GPU.
- `training/trainer.py`: `cpu_offload`, `offload_text_encoder`, validation định kỳ (không làm lệch RNG
  của train), W&B có resume và fallback offline, file `STOP`, log tốc độ và ETA.
- `policy.py`: `set_compute_device` (để tính toán trên GPU trong khi master nằm ở CPU).
- `data/lerobot/index.py`: đọc tên kênh dạng `[[...]]` của LeRobot v3.
