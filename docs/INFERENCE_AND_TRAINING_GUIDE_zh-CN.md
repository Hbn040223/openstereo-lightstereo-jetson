# LightStereo-S 推理与训练速查

这份指南对应本仓库的 LightStereo-S-KITTI 权重与 OpenStereo v2。权重已经在 `checkpoints/LightStereo-S-KITTI.ckpt`，因此 **Jetson 上不会重新下载模型权重**。

> 先决条件：左右图必须是同一时刻、完成双目校正（rectified）、RGB、尺寸完全一致的图像。未经校正的左右图会造成视差错误；模型无法自行补偿相机内外参误差。

## 0. 首次部署（Jetson Orin NX）

```bash
git clone https://github.com/Hbn040223/openstereo-lightstereo-jetson.git
cd openstereo-lightstereo-jetson

./scripts/bootstrap_upstream.sh  # 下载并固定 OpenStereo 源码；不下载模型权重
./scripts/install_jetson.sh      # 安装 JetPack 对应的 Python 依赖
./scripts/verify_jetson.sh       # 校验权重、CUDA 和模型加载
```

如要仅检查权重是否损坏：

```bash
./scripts/verify_checkpoint.sh
```

期望 SHA-256：`3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a`。

## 1. 最常用：一对图像推理

在本仓库根目录执行：

```bash
./scripts/infer_pair.sh \
  --left  /绝对路径/left.png \
  --right /绝对路径/right.png \
  --output outputs/disparity_color.png \
  --gpu 0
```

例如，使用已有 KITTI2015 的测试图：

```bash
./scripts/infer_pair.sh \
  --left  /home/huangbining/datasets/KITTI2015/testing/image_2/000000_10.png \
  --right /home/huangbining/datasets/KITTI2015/testing/image_3/000000_10.png \
  --output outputs/kitti_000000_10_color.png \
  --gpu 0
```

参数含义：

| 参数 | 改动后的结果 |
| --- | --- |
| `--left` | 左目图像；这是视差的参考坐标系。 |
| `--right` | 与左图配对的右目图像；顺序反了会得到不正确视差。 |
| `--output` | 彩色视差可视化 PNG 的保存位置。不会改变模型计算。 |
| `--gpu 0` | 使用哪一张 GPU；多卡机器按可见 GPU 编号填写。 |
| `--checkpoint PATH` | 切换权重；必须与 LightStereo-S 架构和训练域匹配。默认使用仓库内权重。 |

输出 PNG 是**彩色视差图**，便于观察，并不是米制深度，也不是原始浮点视差。若需深度，先导出原始视差 `d`，再以已经标定的焦距 `f`（像素）和基线 `B`（米）计算 `Z = f × B / d`；不要把伪彩色 PNG 当作深度值。

## 2. 直接调用 OpenStereo 原始脚本

脚本封装不适合批量处理时，可在 `vendor/OpenStereo` 中执行：

```bash
cd vendor/OpenStereo
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python tools/infer.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --left_img_path  /绝对路径/left.png \
  --right_img_path /绝对路径/right.png \
  --pretrained_model ../../checkpoints/LightStereo-S-KITTI.ckpt \
  --savename ../../outputs/disparity_color.png
```

不要省略 `--pretrained_model`：配置文件中的 `PRETRAINED_MODEL` 默认是空字符串。`xFormers not available` 对 LightStereo-S 只是可选加速模块的提示，不是推理失败。

## 3. 批量推理模板

下例假定左右图同名、分别位于 `left/` 与 `right/`。先用 1 对图验证结果，再执行整批。

```bash
ROOT=/path/to/openstereo-lightstereo-jetson
LEFT_DIR=/path/to/left
RIGHT_DIR=/path/to/right
OUT_DIR=${ROOT}/outputs/batch
mkdir -p "${OUT_DIR}"

for left in "${LEFT_DIR}"/*.png; do
  name=$(basename "${left}")
  right="${RIGHT_DIR}/${name}"
  [ -f "${right}" ] || { echo "缺少右图: ${right}"; continue; }
  "${ROOT}/scripts/infer_pair.sh" \
    --left "${left}" --right "${right}" \
    --output "${OUT_DIR}/${name}" --gpu 0
done
```

## 4. 速度基准

```bash
./scripts/benchmark_pair.sh \
  --left /绝对路径/left.png \
  --right /绝对路径/right.png \
  --gpu 0
```

该脚本会预热一次并统计 5 次同步前向传播，不含进程启动、读图、保存 PNG 的时间。不同分辨率、功耗模式、JetPack/PyTorch 版本都会影响结果，因此只比较同一设备、同一输入尺寸的数值。

## 5. 训练前必须修改的两处

完整训练更适合显存较大的 x86 GPU 服务器；Orin NX 更适合部署或小批量微调。进入源码目录：

```bash
cd vendor/OpenStereo
```

1. 修改 `cfgs/data_basic.py` 中的数据集根目录。例如：

```python
'KittiDataset15': '/home/huangbining/datasets/KITTI2015',
'KittiDataset12': '/home/huangbining/datasets/KITTI2012',
'SceneFlowDataset': '/home/huangbining/datasets/SceneFlow',
```

2. 检查 YAML 中的样本列表是否存在。LightStereo 的 KITTI 配置使用：

```text
data/KITTI12/kitti12_train194.txt
data/KITTI15/kitti15_train200.txt
data/KITTI15/kitti15_val20.txt
```

列表文件不是图片本身；它记录每一个训练样本的相对路径。数据集目录结构或文件名与 OpenStereo 官方约定不同，就需要重新生成这些 TXT 文件，不能只改根目录。

## 6. 完整训练与微调命令

### 6.1 单卡：从 KITTI 权重微调自己的已标注数据

先复制配置，避免改坏官方配置：

```bash
cd vendor/OpenStereo
cp cfgs/lightstereo/lightstereo_s_kitti.yaml cfgs/lightstereo/my_lightstereo_s.yaml
```

编辑 `my_lightstereo_s.yaml`：将 `DATASET`、`DATA_SPLIT` 改成自己的数据集定义和列表，并填写：

```yaml
MODEL:
  CKPT: -1
  PRETRAINED_MODEL: /absolute/path/openstereo-lightstereo-jetson/checkpoints/LightStereo-S-KITTI.ckpt
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python tools/train.py \
  --cfg_file cfgs/lightstereo/my_lightstereo_s.yaml \
  --save_root_dir ../../output \
  --extra_tag my_finetune \
  --workers 4 \
  --pin_memory \
  --fix_random_seed
```

若在之前的服务器 Conda 环境而不是 Jetson venv 中训练，将 `../../.venv/bin/python` 换成当前环境的 `python`。训练结果默认位于：

```text
output/<数据集名>/LightStereo/my_lightstereo_s/my_finetune/
```

### 6.2 多卡 DDP：SceneFlow 预训练

```bash
cd vendor/OpenStereo
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  --dist_mode \
  --cfg_file cfgs/lightstereo/lightstereo_s_sceneflow.yaml \
  --save_root_dir ../../output \
  --extra_tag sceneflow_2gpu \
  --workers 8 \
  --pin_memory
```

`--nproc_per_node` 必须等于 `CUDA_VISIBLE_DEVICES` 中可见 GPU 数。预训练完成后，把产生的 checkpoint 路径写入 KITTI 或自定义微调 YAML 的 `MODEL.PRETRAINED_MODEL`。

### 6.3 恢复被中断的训练

在同一 YAML 中将 `MODEL.CKPT` 从 `-1` 改成需恢复的 epoch 编号；保持 `--save_root_dir`、配置文件名和 `--extra_tag` 与原实验一致。它恢复的是该实验目录下的训练状态；`PRETRAINED_MODEL` 则表示从另一份权重开始初始化，两者用途不同。

## 7. 配置项：修改位置与实际影响

| 修改位置 | 作用 | 建议与风险 |
| --- | --- | --- |
| `DATA_PATH_DICT`（`cfgs/data_basic.py`） | 指向真实数据集根目录。 | 必改；路径错误会在训练启动前报找不到数据集。 |
| `DATA_SPLIT.TRAINING/EVALUATING` | 决定训练/验证分别看到哪些样本。 | 影响泛化与评价可信度；不要让训练集和验证集混用。 |
| `RandomCrop.SIZE` | 训练裁剪尺寸。 | 更大：上下文更多、显存/耗时更高；更小：更省资源但可能损失远距离或大视差细节。尺寸最好为 32 的倍数。 |
| `RightTopPad.SIZE` | 推理/验证时的最小补零尺寸。 | 改大通常增加显存和耗时；它不是缩放。对已部署的固定分辨率，维持当前设置即可。 |
| `MODEL.MAX_DISP` 与 `EVALUATOR.MAX_DISP` | 最大可预测/评估视差范围。 | 必须彼此一致。增大可覆盖更近目标，但成本体和显存开销增大；对现成权重不要随意改，改后应重新训练或严谨验证。 |
| `OPTIMIZATION.BATCH_SIZE_PER_GPU` | 每张 GPU 的训练批量。 | 增大吞吐但提高显存；显存不足先减小它。改变 batch 后通常应按比例调整学习率。 |
| `OPTIMIZATION.AMP` | 是否混合精度。 | `true` 通常更快、更省显存；若出现 NaN 或不稳定，先降学习率再考虑关闭。 |
| `OPTIMIZER.LR` / `SCHEDULER.MAX_LR` | 学习率。 | 太大：损失震荡或 NaN；太小：收敛慢。二者用 YAML 锚点关联时要一起看。微调通常应显著低于从头训练。 |
| `NUM_EPOCHS` | 训练总轮数。 | 更长可能提升效果，也更易过拟合；以验证集 EPE/D1 为准，而非只看训练损失。 |
| `PRETRAINED_MODEL` | 加载模型初始权重。 | 用于 SceneFlow → KITTI 或 KITTI → 自有数据的迁移；架构不匹配会加载失败。 |
| `CKPT` | 恢复同一实验的训练状态。 | 仅用于断点续训，不能代替 `PRETRAINED_MODEL`。 |
| `EVAL_INTERVAL` | 每多少 epoch 验证一次。 | 更频繁便于观察，但训练更慢。 |
| `MAX_CKPT_SAVE_NUM` | 保留 checkpoint 数量。 | 过大占满磁盘；过小可能丢失最优/早期模型。 |

## 8. 常见结果异常与处理顺序

1. **左右图尺寸不同**：先重新双目校正并裁成同尺寸；脚本会主动停止。
2. **视差整体错误或接近全黑**：确认左/右相机顺序没有反、同步曝光、校正参数和基线正确。
3. **近处物体被截断**：检查真实最大视差是否超过 `MAX_DISP=192`；这通常需要调整模型配置后重新训练。
4. **自己的相机效果比 KITTI 差**：优先收集并标注本相机数据进行微调，而不是先盲目增大模型或分辨率。
5. **显存不足**：训练时依次减小 `BATCH_SIZE_PER_GPU`、`RandomCrop.SIZE`，再启用 `AMP`；推理时使用较小的原图尺寸或减少 Pad 尺寸。
6. **只得到彩色 PNG**：这是原始 `tools/infer.py` 的既定行为；如需测距，添加原始 `.npy` 视差导出，并使用已标定的 `f`、`B` 计算深度。

## 9. 不应修改的内容

- 不要删除或覆盖 `checkpoints/LightStereo-S-KITTI.ckpt`；部署前运行 `./scripts/verify_checkpoint.sh`。
- 不要把数据集、输出、虚拟环境或 SSH 密钥提交到 Git；本仓库的 `.gitignore` 已排除它们。
- 不要为了消除 `xFormers not available` 而强行安装 xFormers；对当前 LightStereo-S 推理并非必需。
