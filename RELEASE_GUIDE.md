# 匿名基础代码说明

这份目录独立保留 Fair Best-of-N 的基础算法、运行入口和 reward model
训练代码。算法核心沿用原实现，新的输入接口不依赖原项目目录。原项目的
历史结果、论文图、检查点、缓存、账号配置、集群脚本均未收录。

## 保留内容

| 内容 | 文件与入口 |
| --- | --- |
| ITP 认证与采样 | `pessimism/core/itp.py` |
| BGP / LCB-Greedy 分配 | `pessimism/core/lcb_greedy.py` |
| 自动 beta 的认证集合、半径、初始化 | `pessimism/core/radii.py` |
| Uniform-BoN、Greedy-BoN、Uniform-ITP | `pessimism/baselines.py` |
| 六种组合的统一运行入口 | `pessimism/run.py`、`scripts/run_algorithms.sh` |
| 合成数据生成与测试 | `scripts/make_synthetic_pool.py`、`tests/` |
| 成对偏好 RM：LoRA / QLoRA | `reward_training/train_rm.py` |
| 成对偏好 RM：全参数 FSDP | `reward_training/train_rm_fsdp.py` |
| 多指标回归 RM | `reward_training/train_criteria_rm.py` |
| 语法、算法与匿名信息检查 | `scripts/check_release.sh`、`scripts/audit_release.py` |

六种组合是 Uniform-BoN、Greedy-BoN、Uniform-ITP-fixed、Uniform-ITP-auto、
BGP-fixed、BGP-auto。固定 beta、alpha、初始化预算和分配批量属于参数设置，
没有按每次历史实验重复复制代码。

这是一份基础实现发布包，不是全部论文图表的一键复现实验包。特定结果的
绘图、自动调参、理想化筛选与 oracle 诊断没有收录。

## 无真实模型或数据的测试

以下命令均在本目录执行。算法检查只需要 NumPy；依赖安装完成后无需联网、
GPU、RM、Judge 或数据集。建议把 Python 环境放在上传目录之外。

```bash
python -m pip install -r requirements.txt
PYTHON_BIN=python bash scripts/check_release.sh
PYTHON_BIN=python bash scripts/run_algorithms.sh --synthetic
```

只运行算法测试和检查匿名信息：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
python -B scripts/audit_release.py
```

训练代码的离线测试、数据格式和完整命令见[训练说明](reward_training/README.md)。
CPU 测试不代表已完成真实大模型训练或多 GPU FSDP 验证。

## 外部数据运行

```bash
python -B -m pessimism.run --help
python -B -m pessimism.run --input '<SCORED_POOL_JSON>' --budget 255 --seed 2026
```

`<SCORED_POOL_JSON>` 只是占位符，必须由使用者自行替换。输入为各组有序候选的
归一化 proxy 分数、候选 ID 和误差界；可选 Judge 分数仅用于结果评估。
具体 JSON 格式与参数见[算法运行说明](pessimism/README.md)。

## 训练入口

```bash
python -m pip install -r reward_training/requirements.txt
bash reward_training/scripts/train_rm.sh --help
bash reward_training/scripts/train_rm_fsdp.sh --help
bash reward_training/scripts/train_criteria_rm.sh --help
```

训练脚本通过命令行接收用户自己的模型、数据和输出位置，未设置任何真实
RM、Judge 或数据集路径；不包含下载地址、私人账号或跟踪服务密钥。

## 上传范围

将本目录本身作为新 repository 的根目录。不要上传上一层原项目目录，也不要
把运行后生成的模型、数据、结果、环境、日志加入这份源码。`.gitignore` 已覆盖
常见产物。上传前再运行 `python -B scripts/audit_release.py`。

本目录没有沿用原项目的 Git 历史。代码匿名化不改变仓库账号、提交作者或
托管平台公开的其他元数据，这些由发布时使用的账号与提交配置决定。

## 本次验收结果

- 69 项算法、核心公式和命令行测试通过。
- 6 项 RM 离线测试通过，包含真实执行两步的小模型 LoRA 训练、保存重载及
  五属性回归训练；使用的模型为临时随机初始化，数据为临时合成数据。
- 把整个目录复制到独立临时位置后，两套测试仍全部通过，六种算法的 shell
  示例可从其他工作目录启动，无需原工程目录。
- Python / shell 语法、四个训练与验证脚本的帮助入口、源码匿名检查通过。
- 7 个算法核心、包入口和基线文件与原实现逐字节一致。

完整 GPU QLoRA、混合精度训练和多 GPU FSDP 未执行；FSDP 启动配置仅验证了
Accelerate 能够正确解析。离线检查使用已有依赖环境，未重新安装依赖。
