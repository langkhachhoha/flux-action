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

## Thay đổi trong code

Mọi tính năng mới đều mặc định tắt, nên các config DROID/SO-101 gốc chạy y như cũ (210 test gốc pass,
thêm `tests/training/test_finetune_extras.py`):

- `training/distributed.py`: `cpu_offload` cho FSDP2 và process group cho 1 GPU.
- `training/trainer.py`: `cpu_offload`, `offload_text_encoder`, validation định kỳ (không làm lệch RNG
  của train), W&B có resume và fallback offline, file `STOP`, log tốc độ và ETA.
- `policy.py`: `set_compute_device` (để tính toán trên GPU trong khi master nằm ở CPU).
- `data/lerobot/index.py`: đọc tên kênh dạng `[[...]]` của LeRobot v3.
