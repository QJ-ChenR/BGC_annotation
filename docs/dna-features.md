# 冻结 DNA 支路：ModernBERT → BGC 向量 → UMAP

这条支路直接读取整条 BGC 核苷酸序列，使用已有 MLM checkpoint 的
backbone，独立输出一个 768 维 BGC 向量。没有新训练，也没有和蛋白向量融合。
UMAP 复用已有脚本；类别仅用于着色，不参与编码或降维拟合。

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
| 配置上下文长度 | `max_position_embeddings=2500`，本支路以此作为保守推理窗上限 |
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
   它不表示模型支持那么长的上下文。2500 也不是代码数学上不能越过的硬界限，
   但缺乏训练依据时，不应直接外推到更长输入。

Tokenizer 没有大小写 normalizer，普通词表只含 A/C/G/T 及其组合。
输入先转大写；N 和其他 IUPAC 模糊碱基保留，并由 tokenizer 编为 UNK。
元数据记录模糊碱基、UNK token 数和 UNK 覆盖碱基数；不会删除它们来缩短序列。

`trainer_state.json` 截至 step 13920 / epoch 6.67，最后记录的训练 loss 约
3.375。日志没有 `eval_loss`，`best_metric` 和 `best_model_checkpoint` 为 null。
不能据此将它称为最佳 checkpoint、判断已经训练完成或证明检索效果。
训练数据清单、与 MIBiG 的重叠、真实截断/分窗方式、mask 比例及反向互补增强仍未知。

源码依据：[ModernBERT 5.1.0 模型实现](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/modernbert/modeling_modernbert.py)、
[配置实现](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/modernbert/configuration_modernbert.py)。

## 推理流程与维度

1. 一个 GBK 文件必须只有一个 record。取该 record 的完整 DNA，包含编码区和
   非编码区，保留文件中的方向；不拼接 CDS、不自动裁切 region。
   BGC ID 使用文件名去扩展名，与原蛋白支路一致。
2. 使用训练时的 tokenizer 对整条序列做 BPE，不截断。
   得到 `M` 个内容 token，并检查 offsets 完整覆盖原 DNA、没有缺失或重复。
3. 按内容 token 划分无重叠窗口。每窗最多 2498 个内容 token，另加 CLS 和 SEP，
   总长度最多 2500。位置编号在每个窗口重新开始。
4. 对一批 `W` 个窗口，token IDs / attention mask 为 `[W,L]`，其中 `L<=2500`。
   backbone 输出 `[W,L,768]`。内部每头 Q/K/V 为 `[W,12,L,64]`；只执行
   backbone，不计算 MLM 的 `[W,L,4096]` 词表预测。
5. attention 保留 CLS/SEP；pooling 排除 CLS、SEP 和 padding，保留内容 UNK。
   对所有窗口的内容 hidden states 求总和后除以 `M`，得到 `[768]`，最后 L2 归一化。
   等价于先算每窗均值，再按各窗内容 token 数加权。末尾短窗口不会与满窗等权。
6. 对所有 BGC 堆叠成 `[BGC_count,768]`。UMAP 基于原始 768 维向量的 cosine
   邻居拟合，输出二维坐标用于绘图。

这一读出是 **BPE token 等权**，不是碱基等权，也不是基因等权。
不同窗口之间没有 attention；完整覆盖不等于学习到了整条长 BGC 的跨窗关系。
用户已选择将此作为训练预处理尚未核实情况下的初始基线。默认也不做反向互补平均。

## 服务器运行

在项目根目录运行。建议使用独立环境，固定 Transformers 版本；GPU 版 PyTorch
沿用服务器已匹配 CUDA 的安装。

```bash
python -m pip install -e '.[dna,visualization]'
```

服务器的 checkpoint 目录应包含 `config.json`、`tokenizer.json`、
`tokenizer_config.json` 和 `model.safetensors`（也支持标准 HF 分片或
`pytorch_model.bin`）。只有本地配置时会直接报缺少权重，不会产生随机 embedding。

```bash
python -m experiment.extract_dna_features \
  --gbk /path/to/mibig_gbk_4.0 \
  --checkpoint /path/to/trained/checkpoint-13920 \
  --device cuda:0 \
  --dtype float32 \
  --attention-backend sdpa \
  --max-tokens 2500 \
  --window-batch-size 1 \
  --output outputs/features_dna

python -m visualization.umap_features \
  --features-dir outputs/features_dna \
  --labels data/BGC_4.0/MAC_metadata.pkl \
  --metric cosine --n-neighbors 15 --min-dist 0.1 --random-state 42 \
  --output outputs/umap_dna
```

若权重目录没有 tokenizer/config，可在提取命令中追加：

```bash
--tokenizer BGC_transformer --config BGC_transformer/config.json
```

这些文件必须与权重实际使用的词表、架构匹配。形状校验不能发现同大小词表的
token 排列错误，因此不要替换为另一个 tokenizer。加载只访问本地文件，
拒绝缺失、额外或不匹配的 checkpoint 参数。SHA256、依赖版本、实际后端和精度
写入 metadata；完整权重哈希会增加首次启动的磁盘读取时间。

默认 SDPA 不要求安装 flash-attn。确认服务器 GPU 支持后，可改用
`--dtype bfloat16` 并适当增大窗口 batch；这是另一精度设置，要保留对应运行元数据。
`--attention-backend flash_attention_2` 需要服务器已有兼容依赖和设备。

### 与蛋白支路比较同一批 BGC

可追加 `--ids-from outputs/features_mac_member0/bgc_ids.tsv`，选择同一 ID 集合。
导出按 ID 排序，比较时仍以 ID 对齐，不依赖旧文件的行顺序。缺失 GBK 会报错。

已知 `BGC0002977` 在此前数据检查中是约 4.15 Mb 的整基因组记录，而不是普通
局部 BGC。本支路不会暗中截短或排除它。若此次只研究正常 BGC，可显式追加
`--exclude-id BGC0002977`；这会记录排除原因。蛋白/DNA 对比应使用相同排除规则。

其它无效输入默认使任务失败；明确允许排除时使用 `--skip-invalid`，并检查
`failures.tsv`。已有输出目录中的结果不会被覆盖。

## 输出与解释

- `embeddings.npy`：float32、L2 归一化的 `[N,768]` 矩阵。
- `bgc_ids.tsv`：行号、BGC ID、内容 BPE token 数；不含每窗添加的 CLS/SEP。
- `metadata.json`：架构、权重和 tokenizer 哈希、读出策略、完整覆盖的窗口坐标、
  GC 比例（分母为 A/C/G/T 数）、模糊/UNK 数、排除列表、版本和源码哈希。
- `failures.tsv`：明确排除及失败记录。`status=failed` 的导出不能拿去画图。
- UMAP 输出沿用原脚本，DNA token 数色条明确标注 BPE 单位，避免和蛋白 domain 数混淆。

UMAP 是降维可视化，这里没有额外执行聚类算法。不同支路各自拟合的二维坐标
不能逐点比较距离或坐标位置；类别岛更清楚也不自动等于检索更好。下一步评估应在
原始向量空间做近邻比较，并同时检查类别、物种、GC、长度及训练集重叠。

本地测试使用人工构造的小型 ModernBERT checkpoint 验证加载、完整覆盖、冻结、
padding/特殊 token 排除、向量归一化、导出 ID 对齐和 UMAP 接口。
这些是实现正确性检查，不是师兄真实权重的性能实验；实际 DNA UMAP 需在服务器生成。
