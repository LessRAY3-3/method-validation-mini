# MSA异常的源码与特征诊断

诊断对象是本项目的20 aa Trp-cage预测，不是AlphaFold通用准确率评测。这里只读安装包源码，并运行不加载模型权重的特征预处理；未修改依赖、未启动结构推理。后续推理由独立的 `followup/plan.json` 管理，本文不替代它的结果验证。

## 结论先行

原始A3M与去重A3M并非“相同信息，只是数组补齐不同”。本地流程使原始输入进入模型前成为 **Q、H、H**，去重输入成为 **Q、H**：同源序列的重复行真实参与profile统计和MSA特征构造。它们还触发不同的掩码随机数流。固定种子不能保证不同形状的输入使用相同的掩码。

这已经解释了“为什么模型实际看到的输入不同”，但尚不能把2.826 Å异常构象唯一归因于重复权重、某一个掩码或数值计算误差。单靠结果差异不能完成这种因果分解。

## 证据版本与定位

本次实际安装：ColabFold **1.6.3**、alphafold-colabfold **2.3.20**、JAX **0.11.2**、NumPy **2.5.4**。源码路径均相对于本地 `.colabfold/lib/python3.13/site-packages/`。

| 文件/行号 | 已核查作用 |
|---|---|
| `colabfold/batch.py:1036,1123–1126` | 反序列化本例带 `#20\t1` 头部的A3M；以header和sequence共同去重 |
| `alphafold/data/pipeline.py:53–83` | 序列级去重代码被注释，MSA数组保留重复序列 |
| `colabfold/batch.py:1587–1599` | 单query任务按真实MSA行数截断max_seq/max_extra_seq |
| `colabfold/alphafold/models.py:160–170,202–212,229–232` | 无模板时复用model_3_ptm运行结构、交换model_1参数；设置MSA容量和recycle |
| `alphafold/model/tf/input_pipeline.py:80–112,144–155` | 抽样→掩码→聚类→固定尺寸，并对recycle重采样 |
| `alphafold/model/tf/data_transforms.py:164–189,388–418` | 保留query在第0行；其余行打乱；掩码按真实MSA形状生成 |
| `alphafold/model/tf/np_backend.py:317–334,444–458,552–559,587–593` | 随机状态、ensemble迭代、空extra分支、按行统计profile |
| `alphafold/model/tf/random_philox.py:86–91` | 随机流每次迭代的偏移依赖数组元素数 |

上述8个主要源码文件的SHA-256及45个特征处理案例见 [trace_summary.json](../results/prediction_msa_diagnosis/features/trace_summary.json)。本地源码是本次行为的直接证据；不能用当前网站main分支行号冒充已安装版本。ColabFold官方上游和本地文件已存在行号/实现差异。

外部交叉核对：[ColabFold上游](https://github.com/sokrypton/ColabFold/blob/main/colabfold/batch.py)、[alphafold-colabfold上游的MSA构造](https://github.com/sokrypton/alphafold/blob/main/alphafold/data/pipeline.py)、[DeepMind原版MSA构造](https://github.com/google-deepmind/alphafold/blob/main/alphafold/data/pipeline.py)。前两者与原版不能混为一谈：所查fork关闭了序列去重，而DeepMind原版保留了该去重。

## 为什么原始4行变3行，而去重输入是2行

设Q为query序列、H为本例唯一不同的homolog序列。

| 输入 | 输入记录 | 反序列化后真实行 | 不同序列数 | 有效max_seq |
|---|---|---|---:|---:|
| query_only | Q | Q | 1 | 1 |
| deduplicated_saved_msa | Q、H | Q、H | 2 | 2 |
| raw_saved_msa | Q、Q、H、H | Q、H、H | 2 | 3 |
| duplicate_query | Q、Q、H，header互异 | Q、Q、H | 2 | 3 |
| duplicate_homolog | Q、H、H，header互异 | Q、H、H | 2 | 3 |

原始文件两个query的header同为 `>101`，header与序列都相同，因此合并；两个homolog虽序列相同，但header不同，因此留下两行。后续 `make_msa_features` 未再次按序列去重。

`fast_make_hhblits_profile` 把每一列的残基计数除以真实行数。以第一列为例，Q是N而H是gap：去重输入比例为1/2与1/2；raw为1/3与2/3；duplicate_query为2/3与1/3。这个差异已从实际特征数组验证，不是推测。这里的行数和max_seq都不是Neff。

## 抽样、掩码和聚类分别改变了什么

`sample_msa` 固定第0行query，随机打乱其余行，再抽取不超过max_seq的行。基线的max_seq等于所有真实行数，所以没有丢掉任何真实行，但顺序仍可能重排。

`make_masked_msa` 的替换比例是0.15；替换分布结合均匀氨基酸、列profile、原始氨基酸和MASK符号。profile的统计权重因此能影响替换结果。随后才建立cluster特征并做padding；这意味着事后补齐到同一容量无法让此前随机处理相同。

基线与本文45个纯特征案例中，`extra_msa_mask` 总和均为0：所有真实行都已进主MSA，额外容量1只是填充。`nearest_neighbor_clusters` 在没有extra行时返回空assignment。因此不能把这批基线的异常优先归结为“某条额外序列被分到错误的最近邻”。但若使用raw `--max-msa 2:1`，会有真实extra序列，聚类机制便发生改变；该对照不能当作只改变形状的干净实验。

## 确认的随机流差异

本地NumPy后端在各ensemble/recycle迭代中固定操作种子，Philox均匀随机数的起点使用 `iteration * n * 256`，其中n取决于生成数组的元素数。真实MSA行数不同，后续recycle的随机流偏移就不同。

seed=20261110时，query行被掩码的残基位置为：

| recycle/ensemble索引 | raw，3行 | dedup，2行 |
|---:|---|---|
| 0 | 10、11 | 10、11 |
| 1 | 2、10、11、17、18 | 1、2、20 |
| 2 | 7、10、18、20 | 1、9、13、14 |
| 3 | 8、15、20 | 7、10、18、20 |

这证明了形状改变会改变本例同种子的实际掩码，但不等于证明某一个位置导致错误折叠。该后端注明它是在复现TensorFlow随机操作的计数方式；本文没有把这种确定性差异判为随机数实现bug。

## 纯特征实验已经排除了什么

特征导出脚本：[export_feature_trace.py](../results/prediction_msa_diagnosis/features/export_feature_trace.py)。它只调用A3M读取、feature构造和 `np_example_to_features`，不调用模型初始化、参数加载或 `predict`。

5个输入 × 3个seed × 3种预处理方案，共45个案例。方案为正常处理、最终容量强制3、mask比例0。每个案例保留 `true_msa`、`bert_mask`、`msa_mask`、`msa_feat`、`extra_msa`、`extra_msa_mask`；另存原始MSA和profile。NPZ均为算法数据，不含模型权重。

- **raw与duplicate_homolog：** 三个seed的上述全部特征逐元素完全相同。仅换这些header，没有改变送入模型的所保存特征。
- **raw与duplicate_query：** 同为3行时，三个seed的BERT mask均完全相同；序列内容/profile与整体MSA特征不同。这是比raw对dedup更接近控制形状的对照。
- **raw与dedup强制最终容量3：** 最终形状相同，但特征、row mask和后续recycle的query mask仍不同。真实2行在掩码之后才补齐，不能等同3条真实行。
- **mask比例设0：** 三个seed中两种输入的query行 `msa_feat` 完全相同，BERT mask全零；其他行和整体张量仍不同。关闭mask是否改变异常预测，本文没有运行推理，不能从这个特征结果直接下结论。

逐元素比较记录：[feature_comparisons.json](../results/prediction_msa_diagnosis/features/feature_comparisons.json)、[no_mask_feature_comparison.json](../results/prediction_msa_diagnosis/features/no_mask_feature_comparison.json)。

## --max-msa能否直接固定容量

不能用单query CLI中的 `--max-msa 3:1` 实现raw与dedup都3行。单query预处理仍取 `min(真实行数, 请求上限)`，dedup会降到2。指定2:1虽能让主MSA形状都2，但raw会把一条homolog移到extra，进而改变聚类；并非只改变padding。

若要研究纯形状影响，应在独立的研究运行器中固定处理后的张量与mask，仅改变被明确标零的padding；不得靠添加真实重复序列冒充padding，也不需要修改安装包。即使完成这种测试，也要区别“浮点/编译形状影响”与“输入信息/掩码影响”。

## 原始异常在recycle中何时显现

从留存日志读取seed=20261110的pLDDT：

| 输入 | recycle 0 | recycle 1 | recycle 2 | recycle 3 |
|---|---:|---:|---:|---:|
| query-only | 70.5 | 71.2 | 80.6 | 87.3 |
| dedup | 76.8 | 72.6 | 73.9 | 73.2 |
| raw | 75.9 | 73.0 | 84.8 | 88.8 |

置信度轨迹在recycle 0就不同，recycle 2开始出现显著分离：query-only和raw提高，dedup没有同样提升。由于原始运行没有保存每个recycle的坐标，不能把这张表写成“recycle 2发生了错误折叠”，也不能确定结构异常最早出现在哪一轮。

## 最小判别试验与解释边界

1. **重复异常seed：** 同输入/权重/软件/seed重跑并核对坐标SHA，先分辨确定性结果与运行异常。这是必要检查，但不能解释机制。
2. **同形状重复行：** Q,Q,H与Q,H,H，保持header互异、同seed。纯特征结果已确认二者同shape同mask；若结构不同，可排除“只因3行尺寸不同”，但仍不能区分重复内容、profile和其他真实行特征各自的贡献。
3. **无掩码对照：** 在独立运行器显式修改配置的mask比例为0，先导出特征，再对异常seed比较dedup与raw。无mask是诊断干预，不是宣称更准确的推荐默认参数。若异常消失，只能说明结果对该干预敏感，不能定位单个掩码因果。
4. **保留全程坐标的更多recycle：** 对异常输入保存每轮坐标并相对38模型算RMSD，区分只是迭代不足还是持续停留在另一构象；禁止只选最有利一轮报告。
5. **预先列出的额外seed：** 检查异常出现频率是否只绑定原seed，仍限于此一短肽，不能形成总体准确率结论。

## 已完成的后续对照如何约束解释

主任务随后完成14次短CPU对照，全部验证结果见 [followup/summary.json](../results/prediction_msa_diagnosis/followup/summary.json)，完整数值见 [targeted_results.csv](../results/prediction_msa_diagnosis/followup/targeted_results.csv)。下列RMSD均为对38个NMR模型的Cα最优叠合RMSD中位数，单位Å。

| 对照（均为原异常seed，除另注明） | RMSD | pLDDT | 可以得出的结论 |
|---|---:|---:|---|
| 原dedup重复，3 recycle | 2.82613 | 73.1745 | 坐标及置信度数组完全相同；异常是该配置可重复的结果 |
| Q,Q,H，3行 | 2.31454 | 74.2040 | 同3行尺寸仍可得到较差结果，排除“3行尺寸本身保证成功” |
| Q,H,H，3行 | 0.90206 | 88.7915 | 特征和坐标/置信度均与原raw相同，支持header去重链路解释 |
| raw主MSA容量2、extra容量1 | 0.92707 | 87.2470 | 主MSA同为2行仍可得到接近NMR的结构，排除“2行尺寸本身必然失败” |
| dedup延长至12 recycle | 0.85145 | 91.1905 | 这次增加迭代后结果改善；不能据此保证其他任务单调改善 |

新增三个预先列出的seed（20261111、20261112、20261113）下，dedup的RMSD为0.88392、0.85502、0.94085 Å；连同原三seed，只有原seed20261110出现2.82613 Å这一明显较高值。这是六个已选seed上的描述，不能把1/6称为模型总体失败率。

12-recycle日志前四轮与原运行相同，第6轮pLDDT升至80.7，第7轮87.5，最后91.2。能够确认的是置信度轨迹及最终结构改善；没有逐轮坐标时，不定位结构转换恰好发生在第6轮。增加recycle也增加了重采样轮次，因此尚不能区分“更多迭代本身”与“后续掩码抽样”的各自作用。

综合证据现在支持：**该异常是特定输入表示、种子与迭代预算下可重复、但并非对所有seed稳定出现的结果；真实重复行内容与随机预处理路径都参与了差异，纯粹的最终MSA尺寸不是充分解释。** 尚未支持“homolog有害”“重复越多越准确”“某个掩码位置造成错误”“所有AlphaFold版本都这样处理重复序列”“AlphaFold源码存在bug”等结论。本文的源码链路特指所记录的ColabFold/alphafold-colabfold版本及其本地NumPy预处理后端。

