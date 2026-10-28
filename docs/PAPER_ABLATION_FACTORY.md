# GPT-1 消融研究工程规范（对应论文与实现）

原论文：Alec Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training*
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

本文件区分：**原论文报告**、**本项目实现**、**尚未复现实证**。仓库代码不能代替原论文实验结果。

## 1. 论文证据映射

| 原论文位置 | 实验机制 | 本仓库实现 | 验收 / 状态 |
| --- | --- | --- | --- |
| `3.1 Eq.1–2 | 自回归 LM、学习位置 Embedding、共享输出投影 | `models/gpt_decoder.py`、`trainers/pretrain_trainer.py` | 已有单测；尚未完整规模训练 |
| `3.2 Eq.3–5 | 分类目标 + λ 倍辅助 LM | `trainers/finetune_trainer.py` | 损失分量单独日志；λ=0/0.5 |
| `3.3 Fig.1 | NLI 拼接；句对双顺序；选择题逐候选打分 | `tasks/protocol.py`、`data/collators.py` | 参照 `tests/paper_contract` |
| `4.1 | 12L/768D/12H/3072，max=512，warmup=2000，cosine | `configs/model/gpt_small.yaml`、`configs/pretrain_books.yaml` | 论文配置，不意味着已复现 |
| `4.1 | 40k BPE merges；BookCorpus 长距离连续文本 | `data/datasets.py`、`data/text_bpe.py` | 需验证合规数据的语料同一性与采样 |
| Table 5 | w/aux、w/o pretrain、w/o aux、LSTM w/aux | `configs/paper_ablations.yaml` | 运行前分别准备 Transformer/LSTM checkpoint |
| Fig.2(left) | 前 K 层迁移 | `transfer_layers` + report by depth | 覆盖 K=0,3,6,9,12 |
| Fig.2(right) | 预训练阶段的 zero-shot 能力变化 | `experiments/zero_shot.py`、`cli/zero_shot_race.py` | RACE 部分实现；CoLA/SST-2/DPRD 的论文启发式及完整轨迹待补齐 |
| Table 2–4 | NLI、分类、相似度、QA | `data/datasets.py` / `tasks/metrics.py` | GLUE 子集接通；QA 仍需完整基准 |

## 2. 原论文 Table 5 的数值（仅用于对照，不是本仓库结果）

| arm | paper avg |
|---|---:|
| Transformer + aux LM | 74.7 |
| Transformer - pretrain | 59.9 |
| Transformer - aux LM | 75.0 |
| LSTM + aux LM | 69.1 |

注意 avg 是 8 项异构任务的简单算术平均，包含 MCC / accuracy / F1 / Pearson 的原论文报告单位（百分制）。
此指标不是严格同尺度统计量。**不得使用不同任务平均分声称统计显著性。**
推荐首先报告每任务主指标、随机种子的 mean±sample SD、配对差值和 bootstrap CI（CI 尚待实现）。

## 3. Factory Mode（严禁配置分支耦合）

`experiments/factory.py` 负责把 (suite, arm, task, seed, transfer depth, λ)
实例化为 immutable `Job`。每个 Job 保存 `config_hash`，独立路径：
`runs/paper_ablations/<suite>__<variant>__<task>__kK__lamX__sSEED/`。

运行目录预期：
```text
resolved.yaml       # 实际展开后的完整配置
provenance.json     # 配置摘要、任务、状态（并非完整数据集指纹）
runner.log          # 所有子进程输出
events.jsonl        # 训练/验证指标流
tb/                 # TensorBoard
metrics.json        # best epoch 与任务主指标
checkpoints/best.pt # backbone + classification/choice head
```

禁止不同实验写入同一个目录。已存在目录若配置哈希不同则明确拒绝覆盖。
`--resume` 跳过已有匹配配置的完成状态，不是中断步骤级 optimizer/RNG 恢复。

## 4. 运行方式

```bash
python -m gpt1_factory.cli.ablation_factory --cfg configs/ablations.yaml
# 只生成 3 组 smoke 计划，不训练

# 必须先完成 Transformer / LSTM 预训练，并准备统一 tokenizer artifact：
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_books.yaml
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_lstm.yaml

python -m gpt1_factory.cli.ablation_factory --cfg configs/paper_ablations.yaml --execute --resume --fail-fast
python -m gpt1_factory.cli.report_ablations --runs runs/paper_ablations --out runs/ablation_report
tensorboard --logdir runs/paper_ablations
```

对于单次 smoke，可使用 `configs/ablations.yaml`，其基础配置为 `configs/finetune_smoke.yaml`（128 token，与 mini 模型匹配）。完整实验矩阵包含 162 个任务运行（不含预训练），
运行时间与资源远高于 smoke；预训练 checkpoint 路径必须先存在。

## 5. 公平性与识别策略

- 同一比较内固定数据集 split、Tokenizer artifact、样本过滤规则、seq_len、seed、优化步骤预算。
- `no_pretrain` 只去掉初始化迁移，仍加载 **同一个 token-id 映射**。
- LSTM 使用独立 LSTM 预训练，而非加载 Transformer 权重；对比时记录总参数量 / FLOPs / wall-clock。
- Figure 2 中 `K=0` 与随机初始化不同。**目前代码 K=0 仍转移非 Transformer block 权重，不能严格称为 embedding-only。** 必须先修正权重过滤契约才能作同义比较。
- 内部只使用验证集选模型，官方 test 标签不可泄露入调参过程。
- `qnli` 和 `rte` 需要正确的成对输入字段；MRPC / QQP 的主指标为 F1、STS-B 为 Pearson。
- 原论文的实际 tokenizer 与原始 BooksCorpus 数据可能受可用性和许可限制。替代数据结果只可标为 conceptual reproduction。
- 需要按 epoch 记录训练数据与验证数据哈希，后续补数据指纹、可中断完整状态、bootstrap CI、zero-shot。
- 可视化有两层：训练过程的 TensorBoard 曲线；跨运行报告的本地 HTML 可筛选对照表。

## 6. 分阶段验收

- Stage A（当前）：配置隔离、Table 5/transfer/λ 矩阵、指标输出、LSTM 同 token 协议、离线 HTML 报告、单元测试。
- Stage B：独立数据 manifest（SHA-256/版本/许可/划分）、严格 BooksCorpus 协议与 token 采样审计。
- Stage C：Figure 2 全部零样本启发式与逐 checkpoint 轨迹；当前仅完成 RACE 候选答案概率评分和评估 CLI。
- Stage D：>=3 个种子的完整任务性能、paired bootstrap CI、显存/算力计量、误差分析、独立验证。

**当前阶段尚未执行远程大模型训练，也不应宣称重现 Table 5 性能。**

## 7. 零样本与适配可视化（新模块）

`experiments/zero_shot.py` 给出平均 **答案 Token 条件对数概率** 的评分函数；
`cli/zero_shot_race.py` 用同一固定 RACE 验证子集遍历预训练检查点，保存 `(pretrain_step, accuracy)`。
这仅覆盖 Fig.2(right) 的 **RACE 一条曲线**；CoLA、SST-2、DPRD 的原文启发式仍需要单独实现。

```bash
python -m gpt1_factory.cli.zero_shot_race \
  --model-config configs/model/gpt_small.yaml \
  --tokenizer-artifact runs/exp_pretrain_books/tokenizer \
  --checkpoints runs/exp_pretrain_books/checkpoints/step_10000.pt \
  --limit 250 --out runs/zero_shot_race/metrics.json
```

`experiments/plots.py` 绘制按任务分开的 Table 5 比较柱状图（可显示跨种子 SD）与
Fig.2(left) 迁移深度曲线；`dashboard.html` 提供离线筛选表。跨任务的指标不叠在同一数值坐标上。

## 8. 配对统计口径

配对分析固定 `suite/task/seed/transfer depth`，将 `aux_lm_lambda` 及架构类型
视作允许改变的处理变量。否则 “no_aux vs full” 因 λ 不同而无法比较。
如控制组重复或指标类型不一致则拒绝生成不可靠的配对差值。
目前显示的是差值均值与样本 SD，不是显著性检验或置信区间。
