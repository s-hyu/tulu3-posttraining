# Tulu3 实验日记

这份日记记录项目推进过程中的主要操作、问题和结果。详细脚本、原始日志、模型和评测文件均保留在项目目录中，这里只记关键经过。

## 核心踩坑复盘

| 坑点 | 当时的判断或遗漏 | 实际后果 | 处理与经验 |
| --- | --- | --- | --- |
| 大模型下载 | 直接使用 Hugging Face 下载多 GB 权重，一开始把问题当成“网速慢” | 连接多次中断，单个权重分片无法完整落盘 | 大文件下载要同时准备镜像、断点续传和校验。最终用 ModelScope 镜像完成下载，并按 SHA256 校验。 |

| Checkpoint 策略 | 早期 smoke 结束后脚本自动清理中间 checkpoint，没有在启动前明确保存频率和保留数量 | 中断后可恢复状态有限，训练过程的可追溯性也变弱 | 正式训练应当保留最近若干 checkpoint，并在训练开始后先确认第一份 checkpoint 已保存。 |
| OLMES 依赖 | 按默认依赖直接安装，没有先固定 PyArrow、NumPy、Setuptools 的组合 | 安装出现版本冲突，评测环境启动受阻 | 评测环境独立创建，固定关键版本并保存环境快照；训练和评测环境不混用。 |
| 生成长度 | 默认信任 OLMES 的 GSM8K 设置，没有检查模型自带 `generation_config.json` | Qwen 的 `max_new_tokens=2048` 覆盖任务上限 512，GSM8K 变慢，并在大 batch 下触发显存压力 | 评测前检查模型配置、任务配置和实际传给 `generate()` 的参数；本次将生成上限固定为 512。 |
| 评测并发 | 只看 A100 的显存余量，忽略多个 OLMES 进程同时保留请求、预测和 tokenizer 缓存时的主机内存 | 多个 MMLU 分片出现 `exit 137`，评测需要补跑 | 并发数由 GPU 显存和主机内存共同决定。改为小分片、错峰并发，并依据已落盘结果补齐缺项。 |
| DPO 预算与显存 | 初始 microbatch=2 的全参数 DPO 对 3B 模型仍然偏重 | 第 6 步反向传播显存不足 | microbatch 调为 1、梯度累积调为 8，有效 batch 保持不变；reference logprob cache 复用，避免重复计算。 |
| DPO 结果判断 | 9,900 对、1 epoch 的结果容易受数据规模、学习率和偏好分布影响 | 两个 MMLU 快速筛查任务下降，训练信号也较弱 | 小规模 DPO 的正确用途是调通和筛选配置；下一轮扩大数据预算、保留 policy，并进行完整 OLMES。 |
| RLVR 资源规划 | 将全参数 learner、optimizer 和 vLLM rollout 同时放在单张 A100 上 | GPU 端先出现 Adam 更新 OOM；CPU offload 后主机内存又达到上限 | RLVR 要把训练显存、优化器状态、rollout 引擎和 KV cache 一起估算。更合理的方案是 learner 与 vLLM 分到两张 A100。 |
| LoRA 与 vLLM 同步 | LoRA 降低了 learner 的训练显存，但权重同步模块仍按裸 Qwen2 参数名工作 | PEFT 的 `base_model` 参数路径与 vLLM 参数名冲突 | LoRA-RLVR 还需要补充参数名映射或采用原生支持 PEFT 同步的实现。 |

这次最重要的经验是：训练能启动不等于实验成立；需要同时确认 checkpoint、评测协议、资源上限和结果完整性。

## 本次执行中需要承担的操作问题

下面几项属于本次执行过程中的决策和监督不足，而不只是机器或框架本身的问题。

1. **checkpoint 的检查做得太晚。** 早期训练没有在开始后优先确认第一份 checkpoint 是否真的保存，导致中断风险在长时间运行后才暴露。后续流程应先跑几十到几百步预检，检查 W&B、loss、显存、checkpoint 和恢复命令，再放开长训练。

2. **对耗时任务的停止条件不够果断。** 模型下载和 OLMES 初次并发运行都出现了长时间等待。更好的做法是事先给下载、单任务评测和完整评测设置时间上限；超过上限立刻汇报当前吞吐、剩余时间和替代方案，而不是继续盲等。

3. **RLVR 的资源预估应该放在安装和补丁之前。** 单卡上同时运行全参数 learner、优化器和 vLLM rollout 的显存与主存需求，本应先用模型参数量、优化器状态和 KV cache 做预算，再决定是否进入训练。实际是在依赖、数据和代码适配完成后才碰到资源上限，压缩了可用于真正训练的时间。

4. **LoRA 方案的 vLLM 权重同步兼容性应先做独立检查。** PEFT 参数名与裸 Qwen2 的同步映射问题可以用一次最小 broadcast 测试提前发现。先做这个小测试，可以避免把完整 GRPO 启动、Ray 和 vLLM 初始化都跑完后才定位到同步阶段。


## 10 月 12 日：先把环境和最小训练跑通

项目最初在一台 RTX 3080 10GB 的机器上启动。先分别准备了训练环境和 OLMES 评测环境，训练使用 Open-Instruct，评测使用 OLMES。

模型下载是当天第一个问题。Hugging Face 大文件下载经常中断，速度也很不稳定，于是改用 ModelScope 下载 Qwen2.5-3B。本地模型权重下载完成后，做了 SHA256 校验，并建立本地缓存链接，让 Open-Instruct 能离线加载模型。

随后用 10 条手写英文问答跑了 QLoRA smoke test：Qwen2.5-3B、4-bit 基础模型、BF16 计算、batch size 1，共 10 步。训练正常结束，loss 有限，adapter 可以重新加载并生成内容。

生成检查中，模型偶尔会继续生成下一轮问答，输出里还出现了“鲯”。检查后确认这不是乱码，而是 Base 模型在极小英文样本训练后继续预测的普通 token。这个小实验的作用是验证训练、保存和重载，聊天质量要由后续正式训练和评测判断。

当天也在官方 SFT 数据上做了 12 条样本的小测试。QLoRA/BF16 运行 3 步，loss 从 2.4716 降到 0.9718，说明官方数据的读取、chat template、label mask、反向传播和保存流程都正常。

## 10 月 12 日：OLMES 和三套数据准备

OLMES 单独装在评测环境中。安装过程比预想长，因为需要下载 Torch、CUDA 运行库和多个 Git 依赖。中间遇到两个版本问题：PyArrow 和 NumPy 的版本要求冲突，以及新版 Setuptools 缺少旧版 alpaca_eval 还在导入的 `pkg_resources`。固定 PyArrow 和 Setuptools 后，OLMES 可以正常列出 MMLU、GSM8K 任务，也成功跑完 2 条 MMLU 小样本。

训练数据下载同样走了 ModelScope 镜像。三套数据都下载完成并做了校验：

- SFT 数据约 93.9 万条，用来学习指令回答方式；
- DPO 数据约 27.3 万个 chosen/rejected 偏好对；
- RLVR 数据约 3 万条，覆盖 GSM8K、MATH 和 IFEval。

抽看样本后，三套数据的结构与预期一致：SFT 是 user/assistant 对话；DPO 是同一问题下的优选和非优选回答；RLVR 带有可供 verifier 检查的标准答案或约束。

## 10 月 13 日：Base 模型完整基线

换到 A100 80GB 后，先做 Qwen2.5-3B Base 的完整 OLMES 评测。评测包含全部 MMLU MC、MMLU RC 和 GSM8K，共 115 个任务变体。

第一次并发过高，主机内存出现峰值，部分进程被系统结束。后来改为每 4 个任务一组、错峰 4 路并发，最终补齐全部任务。

GSM8K 一开始特别慢，原因是模型配置中的 `max_new_tokens=2048` 覆盖了 OLMES 的 512 token 设置。把生成长度固定为 512 后，速度恢复正常；评测结束后也恢复了原模型配置。

Base 基线结果：

| 指标 | Base 得分 |
| --- | ---: |
| MMLU MC | 67.21% |
| MMLU RC | 48.62% |
| GSM8K | 63.91% |
| 115 项宏平均 | 57.97% |

## 10 月 13 日：SFT 后测

选定的 SFT 模型已经导出为可直接加载的目录。先做了小范围对照：3 个 MMLU MC 科目的平均分从 57.38% 升至 58.71%；GSM8K 的 132 题固定子样本从 58.33% 升至 67.42%。

之后使用与 Base 一致的 OLMES 配置完成完整后测，覆盖 115 个任务变体。结果如下：

| 指标 | Base | SFT | 变化 |
| --- | ---: | ---: | ---: |
| MMLU MC | 67.21% | 67.26% | +0.05 pp |
| MMLU RC | 48.62% | 47.96% | -0.66 pp |
| GSM8K | 63.91% | 73.54% | +9.63 pp |
| 115 项宏平均 | 57.97% | 57.75% | -0.22 pp |

这轮 SFT 最明显的收获是数学问答能力：GSM8K 提升 9.63 个百分点。MMLU MC 保持稳定，MMLU RC 和总平均有小幅波动。完整模型和评测记录均已保留。

## 10 月 13 日：DPO 小规模训练

DPO 从选定 SFT 模型开始。数据从官方偏好集中按 seed=42 抽取，过滤后保留 9,900 对，训练 1 个 epoch。

第一次使用 microbatch 2、梯度累积 4，在第 6 步遇到显存不足。改为 microbatch 1、梯度累积 8 后，有效 batch 仍然保持为 8。训练复用已经缓存的 reference log probability，最终完成 1,238 步，耗时约 42 分钟。

训练期间 loss 大致在 0.684 到 0.703 之间，末期 reward accuracy 约为 0.525。随后先跑两个 MMLU MC 任务作为快速筛查：medical genetics 从 74% 到 68%，college computer science 从 54% 到 49%。按照时间安排，DPO 在局部筛查后结束，下一轮应增加偏好数据规模并保留训练后的 policy 进行完整评测。

训练结束后曾导出 DPO policy。为腾出空间做 RLVR，随后清理了该权重目录；数据子集、训练配置、W&B 记录、reference cache 和训练日志仍然保留。

## 10 月 13 日：RLVR/GRPO 准备与资源问题

RLVR 先准备了固定 seed 的 512 条 smoke 子集，覆盖 MATH、GSM8K 和 IFEval。经过 prompt 长度过滤后，217 条样本可以进入训练。三类 verifier、数据解析、tokenizer 和本地 Parquet 读取都完成了检查。

为了跑 GRPO，又补齐了 vLLM、OpenEnv、Docker、Backoff 和数学解析依赖，并处理了新版 MCP import、IFEval verifier 名称和本地 Parquet 读取方式的兼容问题。预检阶段成功加载 SFT 模型，创建 learner 与 vLLM rollout actor，并初始化 vLLM。

正式尝试暴露了三个资源与兼容性问题：

1. 全参数训练时，learner 和 vLLM 同时驻留显存，Adam 更新阶段显存不足；
2. optimizer 改为 CPU offload 后，主机内存达到 45.73/47.01GB，Ray 结束了 policy worker；
3. LoRA 方案让 vLLM 顺利启动，但同步权重时出现 PEFT 参数路径与裸 Qwen2 参数命名不一致的问题。

这一阶段已经完成数据、奖励验证器、vLLM 和 GRPO 启动准备。后续适合使用两张 A100：一张负责 learner，一张负责 rollout；同时补上 LoRA 与 vLLM 的权重映射适配。

## 小结

本项目完成了从环境、数据、SFT、OLMES 到 DPO 和 RLVR 准备的完整推进。最稳定的结果来自 SFT：GSM8K 提升 9.63 个百分点。DPO 已完成小规模训练与局部筛查；RLVR 已定位单卡资源边界和需要处理的兼容性点。后续继续推进时，优先安排双卡 RLVR smoke，并把 DPO 的训练后模型保留下来做完整评测。
