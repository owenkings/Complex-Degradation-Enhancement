# 复杂降质图像感知增强 (Project README)

本项目为课程设计项目，旨在对比**图像增强**（Restormer）与**特征增强**（VGG16+Mamba）两种策略在复杂降质场景下对分类任务的改善效果。

## 目录结构

```text
/root/autodl-tmp/owen/task1/
├── data/                       # 数据存放目录
│   ├── CUB-C/                  # 训练集 (CUB-200 降质版)
│   │   ├── origin/             # 干净图像
│   │   ├── fog/                # 降质图像 (雾)
│   │   ├── brightness/         # ...
│   │   └── ...
│   └── ImageNet-C/             # 测试集 (ImageNet 降质版)
│       ├── fog/
│       ├── synset_mapping.txt  # 类别映射文件
│       └── ...
├── task1/                      # 实验一：图像增强 (Image Enhancement)
│   ├── train_task1.py          # 训练 Restormer 脚本
│   ├── eval_task1_imagenetc_vgg16.py # 评估脚本 (ImageNet-C + VGG16 + PSNR/SSIM)
│   ├── plot_task1_loss.py      # 绘图脚本
│   ├── eval_task1_cubc_psnr.py # CUB-C 数据集 PSNR/SSIM 评估 (可选)
│   ├── eval_task1_cubc_vgg16.py# CUB-C 数据集 VGG16 分类评估 (可选)
├── task2/                      # 实验二：特征增强 (Feature Enhancement)
│   ├── train_task2.py          # 训练 Mamba Enhancer 脚本
│   ├── eval_task2.py           # 评估脚本
│   ├── mamba_enhancer.py       # Mamba 模型定义
│   └── vgg_feature_wrapper.py  # VGG 特征提取封装
├── task3/                      # 实验三：特征解码 (Feature Decoder)
│   ├── train_decoder.py        # 训练解码器
│   ├── run_task3.py            # 联合推理脚本
│   └── decoder.py              # 解码器模型定义
├── utils/                      # 工具代码
│   ├── dataset.py              # 数据集加载 (CubCTrainDataset)
│   ├── seed_utils.py           # 随机种子固定
│   └── metrics.py              # 评价指标计算
│── README.md                   # 项目说明文档
└── requirements.txt            # 项目依赖列表
```

## 实验环境准备

建议使用 Python 3.10+ 环境。由于 `mamba_ssm` 对 CUDA 和 PyTorch 版本有特定要求，请严格按照以下步骤安装。

### 1. 安装基础依赖
```bash
pip install -r requirements.txt
```
*注意：如果 `mamba_ssm` 安装失败，请参考下方的手动编译步骤。*

### 2. 手动编译 Mamba (如果 requirements.txt 安装失败)
由于预编译包可能与当前 PyTorch 版本不兼容，建议从源码编译安装 `mamba_ssm`：

```bash
# 1. 确保 PyTorch 版本正确 (匹配 CUDA 12.1)
pip install "torch==2.3.0+cu121" "torchvision==0.18.0+cu121" --index-url https://download.pytorch.org/whl/cu121

# 2. 安装因果卷积依赖
pip install causal_conv1d==1.6.0

# 3. 源码编译安装 mamba_ssm (禁用构建隔离和二进制包)
pip install mamba_ssm==2.3.0 --no-binary mamba_ssm --no-build-isolation
```

## 实验一：图像增强 (Task 1)

**核心逻辑**：使用 Restormer 模型对降质图像进行“去噪/修复”，然后将修复后的图像输入标准的 VGG16 进行分类。

### 1. 训练 (Training)
使用 CUB-C 数据集进行训练。模型学习将 `degraded` (降质) 映射回 `clean` (origin)。

*   **指令**：
    ```bash
    python -u task1/train_task1.py \
      --data-root data/CUB-C \
      --corruption all \
      --epochs 50 \
      --batch-size 4 \
      --accum-steps 8 \
      --amp \
      --lr 5e-5 \
      --print-every 10 \
      --save-dir task1/checkpoints_task1_restormer/all \
      --pretrained Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth
    ```
    *   `--corruption all`: 混合所有降质类型进行训练。
    *   `--save-dir`: 权重及日志保存路径。

### 2. 结果绘图 (Plotting)
训练完成后，绘制 Loss 曲线及验证集 PSNR/SSIM 变化图。

*   **指令**：
    ```bash
    python task1/plot_task1_loss.py \
      --log-path task1/checkpoints_task1_restormer/all/train_log.csv \
      --out-path task1/checkpoints_task1_restormer/all/loss_curve.png
    ```

### 3. 评估与测试 (Evaluation)
在 ImageNet-Val-C 上评估模型性能。此脚本会同时计算 **增强后的图像质量 (PSNR/SSIM)** 和 **VGG16 分类精度 (Accuracy)**。

*   **指令**：
    ```bash
    python task1/eval_task1_imagenetc_vgg16.py \
      --data-root data/ImageNet-C \
      --ckpt task1/checkpoints_task1_restormer/all/best_checkpoint.pth \
      --corruptions "fog,motion_blur" \
      --severities "1,2,3,4,5" \
      --save-json task1/logs/task1_imagenetc_results.json
    ```
    *   `--save-json`: 将详细结果保存为 JSON 文件，包含每种降质类型的 Accuracy, PSNR, SSIM。

*(可选) 在 CUB-C 验证集上评估 PSNR/SSIM：*
```bash
python task1/eval_task1_cubc_psnr.py --ckpt task1/checkpoints_task1_restormer/all/best_checkpoint.pth --corruption all
```

## 实验二：特征增强 (Task 2)

**核心逻辑**：不修复图像，而是提取 VGG16 浅层特征 (Frozen)，通过 Mamba Enhancer 进行“特征去噪”，再送入 VGG16 深层进行分类。

### 1. 训练 (Training)
使用 CUB-C 数据集训练 Mamba Enhancer。

*   **指令**：
    ```bash
    python task2/train_task2.py \
      --data-root data/CUB-C \
      --epochs 20 \
      --batch-size 32 \
      --save-dir task2/checkpoints
    ```

### 2. 评估 (Evaluation)
在 ImageNet-C 上评估特征增强后的分类准确率。

*   **指令**：
    ```bash
    python task2/eval_task2.py \
      --data-root data/ImageNet-C \
      --enhancer-path task2/checkpoints/mamba_enhancer_best.pth \
      --dataset-type imagenet-c \
      --corruption "fog,motion_blur" \
      --severity "1,2,3,4,5"
    ```

## 实验三：基于VGG浅层表征空间的图像增强 (Task 3)

**核心逻辑**：基于Task 2中训练好的Feature Enhancer，设计一个解码器 (Feature Decoder)，将增强后的VGG浅层特征映射回图像空间。

### 1. 训练解码器 (Training Decoder)
使用 CUB-C (Origin) 清晰图像训练解码器。

*   **指令**：
    ```bash
    python task3/train_decoder.py \
      --data-root data/CUB-C \
      --epochs 20 \
      --batch-size 32 \
      --save-dir task3/checkpoints
    ```

### 2. 联合推理与可视化 (Inference)
联合使用 Task 2 的 Enhancer 和 Task 3 的 Decoder 对测试集进行图像增强，并计算 PSNR/SSIM。

*   **指令**：
    ```bash
    python task3/run_task3.py \
      --data-root data/CUB-C \
      --enhancer-ckpt task2/checkpoints/mamba_enhancer_best.pth \
      --decoder-ckpt task3/checkpoints/feature_decoder_best.pth \
      --save-results \
      --output-dir task3/results
    ```
    *   `--save-results`: 将保存 `降质 | 增强 | 清晰` 的对比图到 `task3/results`。

---

## 常见问题 (FAQ)

1.  **数据集说明**
    *   **训练集**：`data/CUB-C`。包含 `origin` (GT) 和多种降质版本。
    *   **测试集**：`data/ImageNet-C`。用于验证模型在真实大规模通用数据集上的泛化能力。

2.  **Mamba 安装失败**
    *   请务必按照“实验环境准备”中的步骤，使用 `--no-build-isolation` 重新编译安装 `mamba_ssm`，以解决 `undefined symbol` 或 PyTorch 版本不匹配问题。
