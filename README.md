# Tulu3 后训练实验

基于 Qwen2.5-3B 的 Tulu3 后训练实验，覆盖 SFT、DPO、RLVR/GRPO 准备和 OLMES 评测。

## 主要结果

| 指标 | Base | SFT | 变化 |
| --- | ---: | ---: | ---: |
| MMLU MC | 67.21% | 67.26% | +0.05 pp |
| MMLU RC | 48.62% | 47.96% | -0.66 pp |
| GSM8K | 63.91% | 73.54% | +9.63 pp |
| 115 项宏平均 | 57.97% | 57.75% | -0.22 pp |

SFT 后，GSM8K 的数学问答能力提升最明显。DPO 完成了固定 seed 的小规模训练和局部筛查。RLVR 完成数据、verifier、vLLM 和 GRPO 启动准备，并定位了单卡资源边界和 LoRA 权重同步适配点。

## 项目结构

```text
scripts/                  训练、数据准备与 OLMES 启动脚本
patches/                  Open-Instruct 和 OLMES 的本地兼容补丁
final_experiment_report.md  正式实验报告
experiment_diary.md       实验日记
experiment_diary.docx     实验日记 Word 版本
environment_eval_freeze.txt  OLMES 环境快照
```

模型、数据集、缓存、W&B 文件和原始评测输出未纳入 Git 仓库；它们体积较大，由本地存储管理。

## 环境

- 训练：NVIDIA A100 80GB，BF16
- 训练框架：AllenAI Open-Instruct
- 评测框架：OLMES
- 评测：MMLU MC、MMLU RC、GSM8K

OLMES 的精确 Python 包版本见 `environment_eval_freeze.txt`。

## 常用命令

准备 SFT 数据子集：

```bash
python scripts/prepare_sft_subset.py
```

启动 SFT：

```bash
bash scripts/run_sft_bf16_subset.sh
```

启动 DPO：

```bash
bash scripts/run_dpo_10k.sh
```

准备并启动 RLVR smoke：

```bash
python scripts/prepare_rlvr_smoke_subset.py
bash scripts/run_rlvr_smoke_512.sh
```

运行 SFT 模型的完整 OLMES：

```bash
bash scripts/run_olmes_sft_full.sh
```

## 文档

- [正式实验报告](final_experiment_report.md)
- [实验日记](experiment_diary.md)
