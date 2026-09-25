# Reward-model 训练基础代码

所有模型、数据和输出位置都由运行参数提供；代码没有实际模型仓库 ID、数据仓库 ID 或部署路径。没有附带权重、训练数据或训练日志。默认关闭实验追踪，脚本不含账号、令牌或集群配置。运行后生成的检查点、日志和配置会记录使用者自行提供的位置，建议输出到此发布目录之外。

## 内容与对应入口

| 入口 | 功能 | 运行脚本 |
| --- | --- | --- |
| `train_rm.py` | 独立领域标量 RM；Bradley–Terry 成对偏好损失；LoRA / 4-bit QLoRA | `scripts/train_rm.sh` |
| `train_rm_fsdp.py` | 同一成对偏好目标的全参数多 GPU FSDP 训练 | `scripts/train_rm_fsdp.sh` |
| `validate_rm.py` | 单独验证 LoRA 适配器或完整标量 RM 检查点 | `scripts/validate_rm.sh` |
| `shp_rm/` | 本地数据校验、聊天模板、保留回答尾部的截断、模型加载、损失、指标及检查点检查 | 由上述入口调用 |
| `tests/test_training.py` | 本地合成数据和随机初始化小模型的 CPU 验证 | 下方 `unittest` 命令 |

代码保留标量序列分类头 `score`，LoRA 适配器包含训练后的 `score` 权重。默认 LoRA 目标是 `q_proj k_proj v_proj o_proj`；使用其他架构时须按其模块名称修改 `--lora_target_modules`。成对训练模型的 tokenizer 必须具有聊天模板。若模型没有独立 padding token，按 tokenizer 的词表提供 `--pad_token`，避免 EOS 与 padding 共用影响末 token 池化。

## 依赖和运行环境

以下命令从 `reward_training/` 目录运行。推荐 Python 3.10–3.12；依赖版本来自原实现。选择适合本机的 PyTorch 安装方式；Linux CUDA 环境才使用 QLoRA 依赖。

```bash
python -m pip install -r requirements.txt
# 仅需要 4-bit QLoRA 或 FSDP 的 8-bit optimizer 时：
python -m pip install -r requirements-qlora.txt
```

脚本使用 `PYTHON_BIN`，否则使用 `PYTHON`，否则使用 `python3`。所有脚本均支持 `--help`。下面的环境变量须由使用者设置为自己的本地资源；示例没有真实路径。`RM_OUTPUT_DIR` 应指向发布目录外的新输出目录。

## 成对偏好数据

`DATASET_DIR` 指向一个本地根目录，必须包含：

```text
<domain>/train.json
<domain>/validation.json
```

每个文件可为 JSON 数组或 JSON Lines。每条记录需包含 `domain`、`history`、`human_ref_A`、`human_ref_B`、`labels`；`domain` 值必须为 `<domain>_train` 或 `<domain>_validation`，与所在文件一致。`labels=1` 表示 A 优于 B，`labels=0` 表示 B 优于 A。空文本或非法标签会被过滤。训练和验证加载器仅允许 `train` 与 `validation`，不会读取 `test.json`。

```bash
# 单设备 LoRA；使用适合硬件的精度参数，如 --bf16 或 --fp16。
bash scripts/train_rm.sh \
  --model_path "${MODEL_DIR:?set MODEL_DIR}" \
  --dataset_path "${DATASET_DIR:?set DATASET_DIR}" \
  --domain "${DOMAIN:?set DOMAIN}" \
  --output_dir "${RM_OUTPUT_DIR:?set RM_OUTPUT_DIR}" \
  --no-load_in_4bit --bf16

# 4-bit QLoRA；需要 requirements-qlora.txt 和 CUDA。
bash scripts/train_rm.sh \
  --model_path "${MODEL_DIR:?}" --dataset_path "${DATASET_DIR:?}" \
  --domain "${DOMAIN:?}" --output_dir "${RM_OUTPUT_DIR:?}" \
  --load_in_4bit --bf16

# 独立验证适配器；所有资源位置均显式传入。
bash scripts/validate_rm.sh \
  --model_path "${MODEL_DIR:?}" --adapter_path "${ADAPTER_DIR:?}" \
  --dataset_path "${DATASET_DIR:?}" --domain "${DOMAIN:?}" \
  --output_dir "${VALIDATION_OUTPUT_DIR:?}" --no-load_in_4bit --bf16
```

验证参数应与训练时的 `--seq_length`、精度、padding 和量化设置一致。`--model_path` 在适配器重载时仍显式决定基座模型，不依赖旧适配器配置中的路径。训练完成后默认检查适配器重载前后奖励的一致性；如只是快速排查硬件问题，可使用 `--no-verify_reload`。

## 全参数 FSDP

`NUM_PROCESSES` 至少为 2。`WRAP_CLASS` 是所用模型解码层的 Python 类名，可从模型的 `_no_split_modules` 获得。包装脚本在临时目录创建通用 Accelerate 配置，采用 `FULL_SHARD`、`FULL_STATE_DICT` 和 `use_orig_params=True`；完成后移除临时配置。

```bash
bash scripts/train_rm_fsdp.sh \
  --num_processes "${NUM_PROCESSES:?}" \
  --wrap_class "${WRAP_CLASS:?}" --mixed_precision bf16 -- \
  --model_path "${MODEL_DIR:?}" --dataset_path "${DATASET_DIR:?}" \
  --domain "${DOMAIN:?}" --output_dir "${RM_OUTPUT_DIR:?}"

# 验证已合并保存的全参数检查点。
bash scripts/validate_rm.sh --full_model \
  --model_path "${FULL_CHECKPOINT_DIR:?}" \
  --dataset_path "${DATASET_DIR:?}" --domain "${DOMAIN:?}" \
  --output_dir "${VALIDATION_OUTPUT_DIR:?}" --no-load_in_4bit --bf16
```

脚本默认仅保存模型权重，不保存优化器状态，所以不支持精确续训。默认优化器是 `adamw_torch`；`--optim adamw_bnb_8bit` 需要额外 QLoRA 依赖。FSDP 的混合精度应通过包装脚本的 `--mixed_precision` 设置。此包装器面向单机多 GPU；多机用户需自行提供 Accelerate 启动配置。`--estimate_only --world_size N` 可在直接调用 `train_rm_fsdp.py` 时估算训练状态内存；估计不包含激活、通信缓冲或 CUDA 上下文，也不能证明模型能够放入显存。

## 离线测试与验证范围

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -m unittest discover -s tests -v

bash scripts/train_rm.sh --help
bash scripts/train_rm_fsdp.sh --help
bash scripts/validate_rm.sh --help
```

测试在系统临时目录中创建合成数据与随机初始化小模型，覆盖 A/B 映射、Bradley–Terry 梯度方向、测试集隔离、成对偏好指标计算、FSDP 参数检查、两步 LoRA 训练及保存重载。无需下载模型、真实数据或 GPU。它验证的是实现连通性，不是论文训练结果。完整 CUDA QLoRA、BF16/FP16 和多 GPU FSDP 需在相应硬件上另行运行。

整理时对副本的正确性修正：FSDP 梯度检查的分布式归约由所有 rank 参与；FSDP 检查及保存使用 Trainer 实际包装后的模型；FSDP 拒绝单进程启动；非量化适配器重载维持与训练一致的参数精度；截断检查不再绑定某个模型家族的 assistant 标记。移除了固定领域列表、固定数据条数和旧实验的硬件/路径默认值。
