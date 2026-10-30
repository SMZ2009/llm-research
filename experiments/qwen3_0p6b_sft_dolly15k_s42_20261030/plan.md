# Qwen3-0.6B × Dolly 15k LoRA SFT

实验 ID：`qwen3_0p6b_sft_dolly15k_s42_20261030`。正式训练开始于 2026-10-30 23:02 CST。本文记录在查看最终 benchmark 结果前锁定的协议。

## 研究问题

Dolly 15k 单轮非思考模式 LoRA SFT 是否提高 Qwen3-0.6B 的独立指令遵循能力？主指标是 Google IFEval 541 题的严格提示级通过率。只有微调模型与基座的配对差值为正且配对 bootstrap 95% 区间下界大于 0，才称有可信提升；其余情形仅报告观察值和不确定性。

## 数据与方法

- 基座：`Qwen/Qwen3-0.6B`，Hub commit `c1899de289a04d12100db370d81485cdf75e47ca`，Apache-2.0。
- 训练数据：Databricks Dolly 15k，Hub commit `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`，CC-BY-SA-3.0。只保留非空指令及回答；指令加上下文做大小写折叠与空白归一后保留第一条；剔除与 IFEval 完全相同的提示；种子 42 随机打乱，约 90/5/5 划分。实际数量和哈希见 `split_manifest.json`。
- 训练：BF16、LoRA r16/alpha32、q_proj/v_proj、batch 1、梯度累计 16、1 epoch、学习率 2e-4、最大长度 1024；仅回答 token 计入损失；Qwen3 chat template，`enable_thinking=False`。每 100 步保存检查点，最多保留 2 个；400 步验证一次。训练结束后导出适配器，删除临时检查点。
- 验证集只用于训练中损失监测；Dolly 留出测试集仅在训练后比较回答 token NLL，作为辅助指标。长度超过 1024 时截断；若提示已占满上限，该样本在两模型 NLL 中共同排除并报告数量。
- 独立 benchmark：Google Research IFEval 提交 `697787053593d98a82f0ffb008463f0eab81ff85`，输入文件 SHA-256 `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`；官方评分代码为 Apache-2.0，具体文件哈希见 `data/imports/ifeval_source.json`。
- 基座与适配器对同 541 题使用相同 Qwen3 非思考 chat template、BF16、贪心解码、最多 512 个新 token、相同 EOS/PAD 配置。按原始顺序保存逐样本响应；采用官方 `test_instruction_following_strict` 计分，`loose` 仅作辅助；语言检测种子固定为 0。对严格提示级通过率做 10,000 次配对 bootstrap，种子 42；保存配对逐题结果。
- 不进行人工盲评，自动评分无法覆盖有用性和事实性。完全相同的提示重叠已排除，语义近似污染未全面审计。

## 运行与留存

训练日志与运行清单在 WSL 运行目录；源数据、固定划分、逐样本生成和评分、适配器二进制均只在本机，GitHub 仅存配置、来源清单、哈希、汇总指标和报告。独立评测环境固定为 `absl-py==2.5.1`、`langdetect==1.0.9`、`nltk==3.9.2`、`immutabledict==4.2.2`，NLTK `punkt_tab.zip` SHA-256 `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`。

失败或非有限损失时停止并诊断。试跑已完成前向、反向、保存、恢复、导出；正式运行留存连续日志与进程 ID。评测只有训练成功后才启动。

运行偏差：第 400 步 TRL 内置验证 `eval_loss` 为 `NaN`；长度审计发现验证集中 9 条提示占满 1024 token，原因是对应样本没有可计分的回答 token。训练损失与梯度保持有限值，训练配置与划分保持冻结。训练后用固定的逐 token NLL 脚本对验证集的基座和适配器重新评分，双方共同排除这 9 条并公开排除数；内置 `eval_loss` 不用于结论。
