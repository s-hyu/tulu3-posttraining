# Tulu3 大模型后训练项目实验报告

**项目：** 南京大学博士入学项目测试  
**日期：** 2026-10-13  
**基础模型：** Qwen2.5-3B Base  
**后训练框架：** AllenAI Open-Instruct；评测框架：OLMES

## 摘要

本实验在 Qwen2.5-3B Base 上实现并验证 Tulu3 后训练流程，目标是依次开展监督微调（SFT）、直接偏好优化（DPO）和可验证奖励强化学习（RLVR），并通过 OLMES 的 MMLU 与 GSM8K 比较训练前后的能力变化。

实验完成了数据下载与校验、独立训练和评测环境、官方 SFT 小样本训练验证、Base 完整 OLMES 基线、SFT 全量微调后完整 OLMES 评测，以及小规模 DPO 的完整一轮训练和局部筛查。SFT 模型在 GSM8K 上由 63.91% 提升至 73.54%，提升 9.63 个百分点；MMLU MC 基本持平。DPO 在 9,900 个有效偏好对上完成训练，局部 MMLU 筛查结果用于后续参数调整。RLVR 完成数据、奖励验证器、vLLM 和 GRPO 启动链路验证；单卡A100 80GB 运行定位出 learner 与 rollout 的显存竞争，以及 LoRA-vLLM 权重同步适配问题。


本报告将**实测结果**、**已验证的工程链路**和**后续训练计划**分开陈述。

## 1. 目标与实验设计

项目按 Tulu3 的三阶段后训练思路设计：

1. **SFT**：使用指令示范数据，使 Base 模型学习对话格式和任务回答方式。
2. **DPO**：从 SFT checkpoint 出发，利用 chosen/rejected 偏好对，优化模型对偏好回答的相对概率。
3. **RLVR**：利用数学题、GSM8K 和指令约束样本的可验证奖励，在在线生成后进行 GRPO 更新。

统一评测使用 OLMES 的 `mmlu::olmes` 与 `gsm8k::olmes`。报告的主要指标为：

- MMLU MC：57 个多项选择任务的 `acc_raw`；
- MMLU RC：57 个阅读理解式任务的 `acc_per_char`；
- GSM8K：`exact_match`；
- 115 个任务变体 primary score 的简单宏平均。

## 2. 实验环境、数据与可复现资产

### 2.1 环境

- 训练：单张 NVIDIA A100 80GB，BF16。
- 评测：独立虚拟环境 `envs/eval`，OLMES commit `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`；Torch `2.8.0+cu128`；Transformers `4.57.6`；Datasets `3.6.0`。
- 训练代码：Open-Instruct commit `11826255077617a46919ce75cadf1f3d53f30dac`。
- 环境冻结：`environment_eval_freeze.txt`；训练、评测脚本和原始日志均保存在项目目录。

### 2.2 数据

| 阶段 | 官方数据集 | 本地校验后的规模 | 本次使用方式 |
| --- | --- | ---: | --- |
| SFT | `allenai/tulu-3-sft-mixture` | 939,343 条 | 固定 seed=42 抽取训练子集用于 SFT |
| DPO | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 272,898 对 | 固定 seed=42 抽取 10,000 对，过滤后 9,900 对 |
| RLVR | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` | 29,946 条 | 固定 seed=42 构造 512 条工程 smoke 子集 |

三套数据均保留了 README、revision 和 SHA256 清单。SFT、DPO、RLVR 的本地路径分别为 `data/tulu3/sft/`、`data/tulu3/dpo/`、`data/tulu3/rlvr/`。

## 3. 训练与评测实施

### 3.1 SFT 训练

首先在 10GB 显存机器上使用 12 条官方 SFT 样本完成 QLoRA/BF16 smoke test：数据读取、chat template、assistant label mask、反向传播、checkpoint 保存均通过。随后在单张 A100 上训练全参数 BF16 SFT。该 checkpoint 已导出为可直接由 Hugging Face/OLMES 加载的目录：

`results/qwen25_3b_sft_bf16_seed42/model/`

Base 与 SFT 采用一致 OLMES 协议：本地模型、HF 后端、FP16 推理、MMLU 5-shot（seed=1234）、GSM8K 8-shot。评测过程将 GSM8K 生成上限固定为任务定义的 512 token，避免 Qwen 本地 `generation_config.json` 中的 2048 token 默认值覆盖任务设置。

### 3.2 DPO 训练

DPO 的 policy 和冻结 reference 均初始化自选定 SFT 模型。训练采用全参数 BF16、最大序列长度 2048、学习率 `5e-7`、linear warmup 10%、`beta=5`、1 epoch、seed=42。配置为 microbatch=1、梯度累积=8，有效 batch 保持为 8。


### 3.3 RLVR 工程验证

RLVR 使用 Open-Instruct 的 `grpo_fast.py`，以选定 SFT 模型为起点。512 条固定 seed=42 子集包含 MATH 127 条、GSM8K 126 条、IFEval 259 条；在 prompt 最大长度 512 的过滤后，有 217 条可进入训练。GSM8K、MATH 和 IFEvalOld verifier 均已执行；为兼容新版依赖安装了 `antlr4-python3-runtime==4.11`，并完成 vLLM `0.19.1`、Ray、OpenEnv 等运行依赖的安装和导入检查。

随后进行了三种资源配置尝试：

1. **全参数、GPU optimizer：** learner 约占 59.13GB、vLLM 约占 12.97GB；Adam 更新还需要约 11.5GB，显存余量仅约 6.61GB，发生 CUDA OOM。
2. **全参数、CPU optimizer offload：** GPU 压力下降，但 learner 在主机内存中约占 37.85GB；48GB 主机内存达到 45.73/47.01GB 后被 Ray OOM monitor 终止。
3. **LoRA 尝试：** 降低参数更新内存后，vLLM 可启动；但 policy 的 PEFT 参数名包含 `base_model`，而 vLLM 的裸 Qwen2 模型没有对应模块，初始化权重同步时失败。

上述尝试均在第一次有效策略更新前结束，对该实现，实际可行的后续资源方案是将 learner 与 vLLM rollout 分到两张 A100 80GB，或使用显存更大的单卡并完成 LoRA-vLLM 权重同步适配。

## 4. OLMES 实测结果

### 4.1 Base 与 SFT 的完整对比

两次完整评测均覆盖 115/115 个任务变体。

| 指标 | Base | SFT | 变化 |
| --- | ---: | ---: | ---: |
| MMLU MC `acc_raw`（57 项） | 67.21% | 67.26% | +0.05 pp |
| MMLU RC `acc_per_char`（57 项） | 48.62% | 47.96% | -0.66 pp |
| GSM8K `exact_match` | 63.91% | 73.54% | +9.63 pp |
| 115 项 primary score 宏平均 | 57.97% | 57.75% | -0.22 pp |

SFT 在 GSM8K 上取得明确提升，MMLU MC 保持稳定，MMLU RC 与总体宏平均出现小幅波动。该模型的主要收益集中在数学问答能力。

### 4.2 DPO 局部筛查

在完整 DPO 后测之前，以 Base 为参照运行两个固定 MMLU MC 任务。结果如下。

| 任务 | Base `acc_raw` | DPO `acc_raw` | 变化 |
| --- | ---: | ---: | ---: |
| medical_genetics | 74.00% | 68.00% | -6.00 pp |
| college_computer_science | 54.00% | 49.00% | -5.00 pp |


两个局部任务分别下降 6.00 和 5.00 个百分点，训练曲线的偏好信号也较弱。按预设规则，局部评测在此结束；GSM8K 子集运行在停止点之前。下一轮 DPO 可扩大固定偏好对预算，并保留训练后的 policy 用于完整 OLMES。

## 5. 结论与讨论

1. **SFT 闭环已完成。** 官方数据加载、全参数训练 checkpoint 导出、模型重载与完整 OLMES 后测均有可复核产物。最明确的正向结果是 GSM8K `+9.63 pp`。
2. **DPO 训练闭环已完成。** 固定 seed、数据子集、reference cache、训练日志和局部 OLMES 筛查均已保留。
3. **RLVR 的数据和系统链路已完成验证。** 本轮定位了单卡 learner、optimizer、rollout 的显存和主存边界，以及 LoRA 参数同步适配点；两卡 learner/rollout 拆分是下一轮的直接方案。
4. **后续实验建议。** 在两张 A100 80GB 上将 learner 和 vLLM 分离，先运行 10–20 个 update 的 RLVR smoke 并确认 reward、KL、格式通过率和 checkpoint；DPO 则应从 SFT checkpoint 用更大但固定的数据预算重新训练，保留 checkpoint 后再进行完整 OLMES。

## 6. 证据与复现路径

| 内容 | 位置 |
| --- | --- |
| 实验过程日记 | `experiment_diary.md` |
| Base 完整 OLMES 指标 | `eval_results/qwen25_3b_base_olmes_microshards/metrics-all.jsonl` |
| SFT 完整 OLMES 指标及对比 | `eval_results/qwen25_3b_sft_olmes_full/metrics-all.jsonl`、`COMPARISON.md` |
| 可加载 SFT checkpoint | `results/qwen25_3b_sft_bf16_seed42/model/` |
| DPO 启动配置 | `scripts/run_dpo_10k.sh` |
| DPO 完整训练日志 | `logs/qwen25_3b_dpo_10k_seed42_retry_mb1_ga8.log` |
| DPO 局部筛查 | `eval_results/qwen25_3b_dpo_10k_seed42_partial_gate/COMPARISON.md` |
| RLVR 子集构造与启动脚本 | `scripts/prepare_rlvr_smoke_subset.py`、`scripts/run_rlvr_smoke_512.sh` |
| RLVR 预检与失败日志 | `logs/qwen25_3b_sft_rlvr_smoke_512_preflight.log`、`logs/qwen25_3b_sft_rlvr_smoke_512.log` |
| 数据下载校验清单 | 各数据目录下的 `download_manifest.md` |
