# CADF-Net

## 1. 环境要求

### 系统要求

- Python 3.7+
- CUDA支持（推荐，用于GPU加速）
- MATLAB（用于评估PSNR/SSIM指标）

### 硬件要求

- 内存：至少8GB RAM
- 存储：至少10GB可用空间
- GPU：NVIDIA GPU（推荐，用于训练和推理）

## 2. 安装步骤

### 2.1 克隆项目

```bash
git clone git@github.com:ang-plus/CADF-Net.git
cd CDSNet
```

### 2.2 创建Python虚拟环境（推荐）

```bash
# 使用conda
conda create -n cdsnet python=3.8
conda activate cdsnet

# 或使用venv
python -m venv cdsnet_env
# Windows
cdsnet_env\Scripts\activate
# Linux/Mac
source cdsnet_env/bin/activate
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

### 2.4 安装PyTorch（根据你的CUDA版本）

```bash
# 对于CUDA 11.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 对于CUDA 10.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu102

# 仅CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3. 配置项目

### 3.1 数据集准备（重要！）

#### 方法一：自动下载（推荐）

```bash
# 下载并准备所有数据集
python data_tools/download_datasets.py /path/to/your/datasets

# 验证数据集完整性
python data_tools/verify_datasets.py
```

#### 方法二：手动下载

1. **DIV2K数据集**：

   ```bash
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
   ```

2. **Flickr2K数据集**：

   ```bash
   wget http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
   ```

3. **测试数据集**：
   - 百度网盘：https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg (密码: sjtu)
   - Google Drive：https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing

### 3.2 数据集目录结构

确保数据集按以下结构组织：

```
datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/           # 800张高分辨率图像
│   └── DIV2K_train_LR_bicubic/   # 低分辨率图像
│       ├── X2/                   # 2倍下采样
│       ├── X3/                   # 3倍下采样
│       └── X4/                   # 4倍下采样
├── Flickr2K/
│   ├── Flickr2K_HR/              # 2650张高分辨率图像
│   └── Flickr2K_LR_bicubic/      # 低分辨率图像
│       ├── X2/
│       ├── X3/
│       └── X4/
└── benchmark/                    # 测试数据集
    ├── HR/
    └── LR/
```

### 3.3 修改环境配置文件

编辑 `env/env.json` 文件，将数据集路径修改为你的实际路径：

```json
{
  "path": {
    "dataset_paths": {
      "Flickr": "/path/to/your/datasets/Flickr2K",
      "DIV2K": "/path/to/your/datasets/DIV2K",
      "DF2K": {
        "Flickr": "/path/to/your/datasets/Flickr2K",
        "DIV2K": "/path/to/your/datasets/DIV2K"
      }
    }
  }
}
```

### 3.4 下载预训练模型

从以下链接下载预训练模型，并解压到 `./train_logs/` 目录：

| 模型            | 下载链接                                                                 |
| --------------- | ------------------------------------------------------------------------ |
| CDSNet_X2_DIV2K | [百度网盘](https://pan.baidu.com/s/1dJhTlhloaiYn9yImk6pa1Q) (密码: sjtu) |
| CDSNet_X2_DF2K  | [百度网盘](https://pan.baidu.com/s/1IK_bzB5gp2tK67zF-VV4Lg) (密码: sjtu) |
| CDSNet_X3_DIV2K | [百度网盘](https://pan.baidu.com/s/19J5uONEOYWxAbEMWIF9qDA) (密码: sjtu) |
| CDSNet_X3_DF2K  | [百度网盘](https://pan.baidu.com/s/1mXL7AOUwyC91UDcEWFCh2Q) (密码: sjtu) |
| CDSNet_X4_DIV2K | [百度网盘](https://pan.baidu.com/s/1kGasS_wslZy4OyzaHTukvg) (密码: sjtu) |
| CDSNet_X4_DF2K  | [百度网盘](https://pan.baidu.com/s/1ovxRa4-wOKZLq_nO6hddsg) (密码: sjtu) |

### 3.5 下载测试数据集

下载benchmark数据集：[百度网盘](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (密码: sjtu)
解压到 `./benchmark/` 目录。

## 4. 运行项目

### 4.1 测试预训练模型

```bash
# 测试X4倍超分辨率模型（DF2K训练）
python test.py -v "CDSNet_X4_DF2K" -s 1000 -t tester_Matlab --test_dataset_name "Urban100"

# 测试X2倍超分辨率模型（DIV2K训练）
python test.py -v "CDSNet_X2_DIV2K" -s 1000 -t tester_Matlab --test_dataset_name "Set5"
```

### 4.2 训练模型

```bash
# 训练X4倍超分辨率模型（DIV2K数据集）
python train.py -v "CDSNet_X4_DIV2K" -p train --train_yaml "train_CDSNet_X4_DIV2K.yaml"

# 训练X2倍超分辨率模型（DF2K数据集）
python train.py -v "CDSNet_X2_DF2K" -p train --train_yaml "train_CDSNet_X2_DF2K.yaml"
```

### 4.3 评估结果

运行MATLAB脚本 `Evaluate_PSNR_SSIM.m` 来计算PSNR和SSIM指标。

## 5. 常见问题

### 5.1 CUDA相关错误

- 确保安装了正确版本的CUDA
- 检查PyTorch版本与CUDA版本兼容性
- 使用 `nvidia-smi` 检查GPU状态

### 5.2 内存不足

- 减少batch_size
- 使用CPU模式：`python test.py -c -1`

### 5.3 路径错误

- 检查 `env/env.json` 中的路径配置
- 确保数据集文件存在且路径正确

### 5.4 依赖包版本冲突

- 使用虚拟环境隔离依赖
- 检查requirements.txt中的版本要求

## 6. 目录结构说明

```
CDSNet/
├── components/          # 网络组件
├── data_tools/         # 数据加载工具
├── env/               # 环境配置
├── ops/               # 操作模块
├── train_scripts/     # 训练脚本
├── test_scripts/      # 测试脚本
├── train_yamls/       # 训练配置文件
├── utilities/         # 工具函数
├── train_logs/        # 训练日志和模型
├── test_logs/         # 测试日志
├── benchmark/         # 测试数据集
├── SR/               # 超分辨率结果
└── system/           # 系统日志
```

## 7. 性能优化建议

- 使用SSD存储数据集以提高读取速度
- 使用多GPU训练（如果可用）
- 调整数据加载器的工作进程数
- 使用混合精度训练减少内存使用
