# Qwen3-0.6B LoRA Research

可复现的单卡 LoRA 微调项目，目标环境为 Windows 11、Ubuntu 22.04 WSL2 和 RTX 5060 Laptop 8 GB。训练模式固定为 **non-thinking SFT**。目前仅完成环境与模型验证；**尚未导入数据集，也未进行训练**。

## 当前验证结果

| 项目 | 结果 |
| --- | --- |
| Python | 3.11.18 |
| PyTorch | 2.9.0+cu128 |
| GPU | RTX 5060 Laptop，计算能力 12.0 |
| 模型 | `Qwen/Qwen3-0.6B`，修订 `c1899de289a04d12100db370d81485cdf75e47ca` |
| 检查 | BF16 CUDA 运算、模型加载及 non-thinking 推理通过；数据契约测试和静态检查通过 |

推理验证时 GPU 峰值分配约 1.2 GB；这**不是训练显存估计**。

## 快速开始

在 WSL2 的仓库目录运行：

```bash
bash scripts/setup_wsl.sh
bash scripts/verify_wsl.sh
```

脚本会在 WSL 的 Linux 文件系统中创建隔离环境、缓存模型；验证结果写入被 Git 忽略的 `reports/local/`。`uv.lock` 锁定依赖版本和包哈希。运行脚本**不会生成完整环境归档**。

数据任务确定后，在 `configs/sft_lora_non_thinking.yaml` 填写任务、来源、许可和固定的 train/validation/test 路径。训练入口会拒绝空数据配置、思考内容、重复记录及跨划分重叠。基础模型和测试集均不参与参数更新。

## 项目结构

```text
configs/              固定模型与实验配置
data/                 数据格式说明；原始数据默认不提交
src/llm_research/     数据校验、训练与推理验证
scripts/              环境配置与验证入口
tests/                数据契约测试
experiments/          运行配置、指标与清单
models/adapters/      导出的 LoRA 适配器（默认不提交）
reports/              公开报告与本地验证结果
docs/                 可复现性与存储策略
```

运行 ID：`qwen3_0p6b_sft_lora_<dataset>_s<seed>_<yyyymmdd_hhmmss>`。模型权重遵循上游 [Qwen3-0.6B 模型仓库](https://huggingface.co/Qwen/Qwen3-0.6B)的 Apache-2.0 许可；本仓库不包含模型权重或数据集。

更多细节见 [复现说明](docs/reproducibility.md)与[存储策略](docs/storage_policy.md)。
