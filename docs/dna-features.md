# 冻结 DNA 支路：ModernBERT → BGC 向量 → UMAP

这条支路直接读取整条 BGC 核苷酸序列，使用已有 MLM checkpoint 的
backbone，独立输出一个 768 维 BGC 向量。没有新训练，也没有和蛋白向量融合。
UMAP 复用已有脚本；类别仅用于着色，不参与编码或降维拟合。

支持两种显式记录的上下文策略：`--sequence-mode full` 将完整 BGC 的 BPE token
送入一次 backbone forward；`--sequence-mode windowed` 保留原来的分窗对照。
为兼容已有调用，未指定时仍默认 `windowed`；下文的全长实验命令明确选择 `full`。

## 本地配置审查（2026-10-22）

审查文件为 `BGC_transformer/config.json`、`tokenizer.json`、
`tokenizer_config.json` 和 `trainer_state.json`。该目录被 `.gitignore`
排除；服务器上的真实权重尚未在本地验证。以下解释针对标准
Hugging Face Transformers **5.1.0**，与配置记录的版本一致。

| 项目 | 实际配置 / 含义 |
| --- | --- |
| 架构 | `ModernBertForMaskedLM`；双向 encoder + MLM prediction head |
| 隐藏表示 | 768 维 |
| Transformer 层数 | 22 |
| Attention heads | 12，每头 64 维 |
| Attention 类型 | 8 层 full attention，14 层 sliding attention |
| Full attention 层 | 从 0 开始编号：0、3、6、9、12、15、18、21 |
| 局部 attention 配置 | `local_attention=128`；具体边界由实现的半窗口 mask 决定 |
| MLP | gated GELU：768 → 2304，拆成两个 1152 分支，`GELU(a) * b`，再 → 768 |
| Norm / 残差 | pre-LayerNorm；首层 attention 前无额外 norm，embedding 已有 norm；末尾 final norm |
| 词表 | 4096，BPE，4087 次 merge，5 个特殊 token |
| 参数量 | 按标准架构且共享词表权重计数：backbone 113,477,376；完整 MLM 114,072,064 |
| 配置上下文长度 | `max_position_embeddings=2500`；保留为 `windowed` 模式的默认窗长，非 `full` 模式的截断上限 |
| 配置 dropout | embedding / attention / MLP 均为 0 |

参数量已通过本地标准架构的 meta-device 实例核对，不代表检查过服务器权重。

### 四处容易误读的配置

1. **实际使用 RoPE。** `rope_parameters` 指定 full attention 的 theta 为
   160000，sliding attention 为 10000。顶层 `rope_theta=3000000` 和
   `position_embedding_type="absolute"` 不控制标准 5.1.0 backbone 的实际位置编码。
   如果训练时使用过自定义模型源码，需要另外核查。
2. **`classifier_pooling="mean"` 不会给 MLM 自动生成 BGC 向量。**
   这是 sequence classification 类使用的设置。此处显式从 backbone 最后一层
   hidden states 做内容 token 平均，是新的推理读出选择。
3. **config 和 tokenizer 的特殊编号不一致。** tokenizer 的真实编号是
   UNK=0、CLS=1、SEP=2、PAD=3、MASK=4；config 中 CLS/BOS=50281、SEP/EOS=50282
   已超出 4096 词表。实际 tokenizer 模板使用正确的 1/2，因此这些陈旧字段
   本身不能证明训练失败。本支路只在内存中同步这些字段，记录原值，不修改
   原始文件，不 resize 词表，不初始化新 embedding。PAD 不一致则直接报错。
4. **tokenizer 的巨大 `model_max_length` 是未设置上限的占位值。**
   它不表示任意长度都可运行。标准 5.1.0 ModernBERT 按实际输入长度生成
   `position_ids` 并计算 RoPE，没有长度为 2500 的绝对位置 embedding 表，
   也不在 backbone forward 中据此拒绝更长序列。配置中的 2500 不能单独证明
   训练时的真实 token 长度。

Tokenizer 没有大小写 normalizer，普通词表只含 A/C/G/T 及其组合。
输入先转大写；N 和其他 IUPAC 模糊碱基保留，并由 tokenizer 编为 UNK。
元数据记录模糊碱基、UNK token 数和 UNK 覆盖碱基数；不会删除它们来缩短序列。

`trainer_state.json` 截至 step 13920 / epoch 6.67，最后记录的训练 loss 约
3.375。日志没有 `eval_loss`，`best_metric` 和 `best_model_checkpoint` 为 null。
不能据此将它称为最佳 checkpoint、判断已经训练完成或证明检索效果。
训练数据清单、与 MIBiG 的重叠、mask 比例及反向互补增强仍未知。

**2026-10-28 的上下文策略更新：** 用户已向 checkpoint 提供者确认，训练时
整条序列进入一次 forward，没有截断或切块。此前按配置 2500 分窗，是训练
输入方式尚不明确时采用的推理基线。现在增加全长模式，以恢复整条 BGC 内的
attention；无需先添加跨块 attention。这个人工确认不等于本工具复核过训练
源码、有效运行参数或数据清单，导出仍不声称已完整验证训练预处理。

全长模式不会修改 `max_position_embeddings`、RoPE 参数、词表或 checkpoint
文件，不新增网络层，不更新权重。继续只在内存中同步上文的特殊 token 编号，
按所选设备、精度及 attention 后端运行已有 backbone。

源码依据：[ModernBERT 5.1.0 模型实现](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/modernbert/modeling_modernbert.py)、
[配置实现](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/modernbert/configuration_modernbert.py)。

## 推理流程与维度

1. 一个 GBK 文件必须只有一个 record。取该 record 的完整 DNA，包含编码区和
   非编码区，保留文件中的方向；不拼接 CDS、不自动裁切 region。
   BGC ID 使用文件名去扩展名，与原蛋白支路一致。
2. 使用训练时的 tokenizer 对整条序列做 BPE，不截断。
   得到 `M` 个内容 token，并检查 offsets 完整覆盖原 DNA、没有缺失或重复。
3. `full`：加一组 CLS/SEP，token IDs / attention mask 为 `[1,M+2]`，只执行
   **一次** backbone forward。位置编号覆盖整个输入，输出 `[1,M+2,768]`，
   内部各头 Q/K/V 为 `[1,12,M+2,64]`。全局 attention 层可连接整条 BGC
   中的 token；局部层继续使用 checkpoint 原有的 sliding attention。
4. `windowed`：按内容 token 划分无重叠窗口，默认每窗最多 2498 个内容 token，
   另加 CLS/SEP。位置编号每窗重新开始。对于一批 `W` 个窗口，输入为
   `[W,L]`、输出为 `[W,L,768]`，默认 `L<=2500`。不同窗口之间没有 attention。
5. 两种模式都只取 backbone 最后一层，不计算 MLM 词表预测。
   attention 保留 CLS/SEP；pooling 排除 CLS、SEP 和 padding，保留内容 UNK。
   全部内容 hidden states 求和除以 `M`，得到 `[768]`，最后 L2 归一化。
   分窗模式等价于先算每窗均值，再按各窗内容 token 数加权；短尾窗不会与满窗等权。
6. 对所有 BGC 堆叠成 `[BGC_count,768]`。UMAP 基于原始 768 维向量的 cosine
   邻居拟合，输出二维坐标用于绘图。

这一读出是 **BPE token 等权**，不是碱基等权，也不是基因等权；两种模式都不做
反向互补平均。全长和分窗的 token 覆盖及读出规则相同，但 contextual states
可能不同，因此长 BGC 的向量无需相同，也不能预设全长的检索效果一定更好。

| 参数 | `full` | `windowed` |
| --- | --- | --- |
| `--max-tokens` 不指定 | 不施加配置 2500 的长度门槛；完整输入一次 forward | 使用 checkpoint 配置长度作为每窗上限 |
| 显式 `--max-tokens N` | 含 CLS/SEP 的整条输入资源上限；超限报错，绝不截断或退回分窗 | 含 CLS/SEP 的每窗上限；必须不超过配置值 |
| `--window-batch-size` | 必须为 1 | 每次 forward 可批量编码的独立窗口数 |
| 超长或显存不足 | 明确失败，不静默改用另一上下文策略 | 继续既定窗长，不暗中丢弃尾部 |

## 服务器运行

在项目根目录运行。建议使用独立环境，固定 Transformers 版本；GPU 版 PyTorch
沿用服务器已匹配 CUDA 的安装。

```bash
python -m pip install -e '.[dna,visualization]'
```

服务器的 checkpoint 目录应包含 `config.json`、`tokenizer.json`、
`tokenizer_config.json` 和 `model.safetensors`（也支持标准 HF 分片或
`pytorch_model.bin`）。只有本地配置时会直接报缺少权重，不会产生随机 embedding。

### 先筛选长度不超过 40 kbp 的完整记录

本次实验队列定义为 **完整 GenBank record 的 DNA 长度 ≤ 40,000 bp**，包含
恰好 40,000 bp 的记录。长度来自实际序列，计入 N 和其它 IUPAC 模糊碱基；
不是 BPE token 数、CDS 长度之和或 GenBank 中某个 region 的长度。
超限记录整条排除，不截取前 40 kbp，不推测 BGC 边界。

在服务器的项目根目录运行筛选脚本：

```bash
python -m experiment.filter_gbk_by_length \
  --gbk /path/to/mibig_gbk_4.0 \
  --max-bp 40000 \
  --output outputs/mibig_le40kb
```

`--max-bp` 默认就是 `40000`，可改为其它正整数 bp 上限。筛选只需要
Biopython 和 Python 标准库，不加载 tokenizer、checkpoint、PyTorch 或 GPU。
输入可以是单个 GenBank 文件或目录；目录递归发现 `.gb`、`.gbk`、`.genbank`。
每个文件须包含一个 record，文件名去扩展名作为 BGC ID，重复 ID 会报错。
输出目录应放在输入目录外，已有筛选产物不会被覆盖。

输出包括：

- `gbk/`：符合上限的原始 GenBank 文件副本，文件内容不变，直接作为后续 `--gbk`。
- `bgc_ids.tsv`：保留记录的行号、BGC ID 和 bp 长度，也可作为 `--ids-from` 清单。
- `inventory.tsv`：全部输入的选择状态、实际长度、来源和排除原因，包含超限记录。
- `failures.tsv`：无法读取或不符合 DNA 输入规则的记录。
- `metadata.json`：上限、长度定义、数量、来源和执行状态。

`inventory.tsv` 的正常状态为 `selected`、`length_excluded` 或 `invalid`；
符合长度但因本次筛选失败而未导出的记录标为 `not_exported`。
`metadata.json` 的 `eligible_count` 表示符合条件的数量，`selected_count`
表示实际导出的数量；失败时 `export_complete` 为 `false`。

超限是明确的队列选择，不是解析失败。若遇到坏文件，默认完成扫描并保存审计，
然后以失败退出，不悄悄产出一个缺失记录的可用队列；明确允许排除坏文件时才加
`--skip-invalid`。没有任何符合条件的有效记录也会失败。确认筛选成功后再继续
验证和编码；下列命令使用筛选后的 `gbk/`，无需修改或重新生成 tokenizer 输入。

40 kbp 是本次实验的长度规则，不能据此断定所有超限记录都是异常数据；它也会
排除真实的长模块化 PKS/NRPS BGC。之后比较蛋白/DNA 表示时，应固定相同保留
ID 的查询与候选集合，避免把队列变化当作表示改善。

### L40S 48 GB 的后端选择

全长编码优先使用与训练环境兼容的 **Flash Attention 2 + bfloat16**；下面的
服务器命令显式选择它们。`dna` extra 不自动安装 `flash-attn`，需要服务器
已有与 GPU、CUDA 和 PyTorch 匹配的版本；不支持时会报错，不静默换成 SDPA。
实际训练精度和实现若不同，应先对齐并记录，不把配置文件的 `dtype` 字段当成
训练时是否使用 autocast 的完整证据。

48 GB 不是所有长度都能容纳的保证。Transformers 5.1.0 的 SDPA 长序列局部
attention 路径会生成 `[B,1,L,L]` 布尔 mask，`L=100000` 时单个 mask 约
10 GB，`L≈340000` 时约 116 GB，尚未计入其他张量。Flash Attention 2
避免该稠密局部 mask，但 8 个全局 attention 层仍有平方级计算量。
冻结推理省去了梯度存储，并不消除这些成本。
参见 [5.1.0 mask 实现](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/masking_utils.py)。

### 先验证全长编码

验证脚本复用同一份冻结 backbone，以相同后端和精度比较全长/分窗；不加载
第二套训练参数，不改 checkpoint。默认扫描符合 bp 上限的输入队列的 BPE 长度，
再按长度排序等距抽取最多 5 条记录，覆盖最短到最长。扫描清单、未抽中 ID 和显式排除
记录都会保存；默认抽样包含最长记录，其运行时间可能明显高于普通 BGC。

```bash
python -m experiment.validate_dna_encoding \
  --gbk outputs/mibig_le40kb/gbk \
  --max-bp 40000 \
  --checkpoint /path/to/trained/checkpoint-13920 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attention-backend flash_attention_2 \
  --window-max-tokens 2500 \
  --sample-count 5 \
  --output outputs/validation_dna_full_le40kb
```

可用 `--all` 验证整个队列，或用 `--ids-from /path/to/bgc_ids.tsv` 验证全部
指定 ID（此时不再抽样）。`--max-tokens` 可选，用于声明含 CLS/SEP 的整条输入
资源上限；超限会报告失败，不会产生截断向量。输出目录必须是新的目录。
`--max-bp` 是独立的、包含边界的核苷酸长度筛选；在 tokenization 前排除超长
记录并保存审计，不会将 bp 当作 token 上限。未指定时保持原先不按 bp 筛选的行为。

验证包括：实际输入是否为完整 token 序列加一组 CLS/SEP、全长是否只发生一次
forward、内容 pooling 与独立手工求均值是否一致、向量是否有限且 L2 归一化、
参数是否冻结且运行前后哈希一致。对于分窗模式也只需一窗的短样本，两路应在
数值容差内一致，并默认重复一次全长编码检验稳定性；长样本的两路 cosine
仅作诊断，不要求相同或强行设定“改善”阈值。`--repeat-all` 可重复全部选中样本。

主要输出：

- `summary.json`：选择规则、来源、硬件、参数完整性和总体验证结论。
- `inventory.tsv`：扫描得到的长度、选择状态与排除记录。
- `results.tsv`：逐样本验证、两路 cosine、耗时和 CUDA 峰值显存。
- `failures.tsv`：解析、OOM、执行或数值验证失败，不静默回退分窗。
- `full_embeddings.npy` / `full_bgc_ids.tsv`、`windowed_embeddings.npy` /
  `windowed_bgc_ids.tsv`：各路成功结果；失败后两路 ID 可能不同，必须按 ID 对齐。

直接给验证脚本原始目录和 `--max-bp` 时，超限记录在 `inventory.tsv` 标为
`length_excluded`，保留长度及来源哈希；`summary.json` 记录 bp 上限和排除数量。

只有选中的全部记录都通过两路执行和验证检查，脚本才以 0 退出；出现失败时
保留已完成的结果和失败清单，以非零退出。验证通过只证明所选输入执行符合约定，
不证明检索质量提升，也不证明未抽中的所有长度都能运行。验证目录中的两路数组
是诊断产物；完整 `bgc_vectors_v1` 导出使用下一条命令生成。

### 导出整条 BGC 特征

保留原 `outputs/features_dna` 作为历史分窗基线，全长结果写入新目录：

```bash
python -m experiment.extract_dna_features \
  --gbk outputs/mibig_le40kb/gbk \
  --max-bp 40000 \
  --checkpoint /path/to/trained/checkpoint-13920 \
  --sequence-mode full \
  --device cuda:0 \
  --dtype bfloat16 \
  --attention-backend flash_attention_2 \
  --window-batch-size 1 \
  --output outputs/features_dna_full_le40kb
```

这个命令没有 `--max-tokens`，所以不按配置 2500 截断或分窗。
`--max-bp 40000` 在提取时再次检查实际输入长度：超过上限的记录在编码前
排除并记录，保留记录始终完整编码。筛选后的目录可直接使用；不需要在其中
额外排除已被长度规则移除的 `BGC0002977`，否则会因排除 ID 不存在而报错。
也可将 `--gbk` 指回原始目录并使用 `--max-bp 40000`，在提取时执行同一长度
规则；此方式不生成筛选后的 GBK 副本。两个 DNA CLI 的 bp 上限默认均不启用，
所以旧命令仍保留其原有输入范围。

如果发生 OOM，调整实际后端、精度或显式实验队列；不要把失败记录的分窗向量
混入标为全长的导出。CUDA OOM 和显式全长资源上限失败始终使导出失败，
`--skip-invalid` 不能将它们当成无效 DNA 静默排除；失败 metadata 保存已完成、
失败和未尝试 ID，便于另建明确队列重跑。

若需要比较全长与分窗的检索效果，应额外以相同后端、精度和队列导出分窗对照，
避免把上下文变化与此前 SDPA/float32 到 Flash Attention/bfloat16 的变化混在一起：

```bash
python -m experiment.extract_dna_features \
  --gbk outputs/mibig_le40kb/gbk \
  --max-bp 40000 \
  --checkpoint /path/to/trained/checkpoint-13920 \
  --sequence-mode windowed \
  --device cuda:0 \
  --dtype bfloat16 \
  --attention-backend flash_attention_2 \
  --max-tokens 2500 \
  --window-batch-size 1 \
  --ids-from outputs/features_dna_full_le40kb/bgc_ids.tsv \
  --output outputs/features_dna_windowed_bf16_le40kb
```

### 评估及可视化

先在共同 ID 的原始向量空间计算 cosine 近邻及预先定义的相关性指标，再查看
UMAP。全长/分窗向量的 cosine 差异只能说明表示改变，不能单独说明更接近生物学
目标。已有蛋白支路也应使用相同查询、候选和排除规则。

```bash
python -m visualization.umap_features \
  --features-dir outputs/features_dna_full_le40kb \
  --labels data/BGC_4.0/MAC_metadata.pkl \
  --metric cosine --n-neighbors 15 --min-dist 0.1 --random-state 42 \
  --output outputs/umap_dna_full_le40kb
```

若权重目录没有 tokenizer/config，可在验证或提取命令中追加：

```bash
--tokenizer BGC_transformer --config BGC_transformer/config.json
```

这些文件必须与权重实际使用的词表、架构匹配。形状校验不能发现同大小词表的
token 排列错误，因此不要替换为另一个 tokenizer。加载只访问本地文件，
拒绝缺失、额外或不匹配的 checkpoint 参数。SHA256、依赖版本、实际后端和精度
写入 metadata；完整权重哈希会增加首次启动的磁盘读取时间。

CLI 的默认值仍为 SDPA/float32，便于 CPU 和既有调用；长序列服务器实验请显式
选择已验证的后端和精度。分窗模式可以增大 `--window-batch-size`；全长模式
每次处理一条完整 BGC，要求该值为 1。

### 与蛋白支路比较同一批 BGC

以本次成功导出的 `outputs/features_dna_full_le40kb/bgc_ids.tsv` 确定 DNA 队列，
从已有 ESM/MAC/MAP 导出按 ID 取交集，再在同一组 ID 上比较。不要直接把旧的
全部蛋白 ID 清单传给筛选后的 GBK 目录：其中超过 40 kbp 的 ID 已不存在，
脚本会按约定报告缺失，而不会默默取交集。

若希望从原始 GBK 目录出发，只编码旧蛋白队列中满足长度限制的记录，可同时用
以下输入参数：

```bash
--gbk /path/to/mibig_gbk_4.0 \
--ids-from outputs/features_mac_member0/bgc_ids.tsv \
--max-bp 40000
```

先检查全部请求 ID 存在，再按长度筛选。最终比较仍以成功导出的 ID 为准，
不依赖旧文件的行顺序。

不使用 bp 筛选时，仍可明确用 `--exclude-id BGC0002977` 排除已知约 4.15 Mb
的整基因组记录，并记录原因。长度筛选已经排除它时无需重复指定。

其它无效输入默认使任务失败；明确允许排除时使用 `--skip-invalid`，并检查
`failures.tsv`。已有输出目录中的结果不会被覆盖。

## 输出与解释

- `embeddings.npy`：float32、L2 归一化的 `[N,768]` 矩阵。
- `bgc_ids.tsv`：行号、BGC ID、内容 BPE token 数；不含每窗添加的 CLS/SEP。
- `metadata.json`：架构、权重和 tokenizer 哈希、全长/分窗模式、读出策略、完整覆盖的坐标、
  GC 比例（分母为 A/C/G/T 数）、模糊/UNK 数、排除列表、版本和源码哈希。
- `failures.tsv`：明确排除及失败记录；bp 超限的阶段为 `length_filter`，不计作
  无效 DNA。`metadata.json` 的 `length_exclusions` 保留超限记录的实际 bp 长度、
  来源和哈希，`length_excluded_count` 与 `invalid_count` 分别计数。
  `status=failed` 的导出不能拿去画图。
- UMAP 输出沿用原脚本，DNA token 数色条明确标注 BPE 单位，避免和蛋白 domain 数混淆。

UMAP 是降维可视化，这里没有额外执行聚类算法。不同支路各自拟合的二维坐标
不能逐点比较距离或坐标位置；类别岛更清楚也不自动等于检索更好。下一步评估应在
原始向量空间做近邻比较，并同时检查类别、物种、GC、长度及训练集重叠。

本地测试使用人工构造的小型 ModernBERT checkpoint 验证加载、超过配置长度的
单次全长 forward、完整覆盖、冻结、padding/特殊 token 排除、向量归一化、
显式资源上限、失败记录、导出 ID 对齐和 UMAP 接口。
长度筛选还覆盖 39,999 / 40,000 / 40,001 bp 边界、模糊碱基计数、无效记录审计、
原始 GBK 字节保留、写入失败回滚，以及筛选目录 / ID 清单直接接入全长提取。
这些是实现正确性检查，不是师兄真实权重的性能实验；实际 DNA UMAP 需在服务器生成。

2026-10-28 本地回归：Python 3.12、PyTorch 2.6.0 CPU、Transformers 5.1.0，
`python -m unittest discover -s tests -v` 共 124 项测试，123 项通过，1 项 CUDA
检查因无 GPU 跳过。真实 checkpoint、Flash Attention 2 和 L40S 的实际峰值显存
仍由上述服务器验证命令确认。
