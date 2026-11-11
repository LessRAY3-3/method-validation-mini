# 起点与有约束NPT时长对照：执行记录

## 启动

2026-11-11 已获准执行完整四格方案。A复用新增种子基线，B/C/D各三条11 ns，新增99 ns。已于2026-11-11 22:32（Asia/Shanghai）完成九条新增模拟及四组分析。新增99 ns；四格主体复用A三条，共12条132 ns；另有最初探索三条33 ns，项目正式生产合计15条165 ns。

冻结协议：`config/experiments/md_factorial_protocol.json`，SHA-256 `ad42ba21cc13d23aa3c599c7f746de60ecf1da4c8ce154b8b73fe829a58b4824`。冻结96项输入、7项源码。远端逐项校验通过，启动前58项单元测试通过；分支100→102 ps真实续跑、完整分析零差值试验通过。

Ubuntu队列PID19110，`results/md_factorial/queue_status.json`记录实际状态；日志`setup/md_factorial/queue.log`。首快照为B/20261113有约束NPT正在延长，GPU利用率85%。此快照不是完成证据。

## 第二起点

model24已准备、最小化完成：蛋白304原子身份/顺序/拓扑参数与原体系一致，2526水和1CL，共7883原子；原体系2678水。两最小化起点内部CA3–18 RMSD为0.617665 Å。显式离子放置seed20261111与原未指定seed不同；跨起点效果包含准备差异。完整准备命令和哈希见`setup/md_factorial/model24/preparation_provenance.json`。

## 分工

4070：固定环境运行九条正式动力学，再做需要GROMACS的统一指标提取。M3：已启动六worker对已有六条坐标的独立CPU复算（PID46185），独立解析GRO/PDB、Bio.SVDSuperimposer、38参考/3窗口、Q阈值与局部距离，计划2052个参考标量与66060个序列标量核验。其结果须以完成记录为准；后续新轨迹沿用此核验流程。本地M4负责调度、轻量检查和报告。

## 恢复限制

动态阶段可按同一冻结协议恢复。分析目录已存在时禁止覆盖；若后处理失败，保留部分输出、查明问题后用新目录重新分析。旧六条、失败尝试和冻结源码保留。三页PDF现已在本轮审核后更新，旧三重复版已归档，详见output/pdf/legacy/md_three_replicates_before_factorial_20261111/。

## M3基线独立核验完成

`results/validation/md_geometry_m3_independent.json`记录六worker的实际完成结果：2052项参考标量与66060项时间序列数值比较均通过；最大误差分别8.88e-16 Å与4.44e-15 Å。15项输入/源码哈希一致，沿用现有NumPy1.26.4/Biopython1.88，无安装。每worker约2.8秒，因此当前任务已结束，M3并非持续满载。该审计基于已导出的完整蛋白坐标，不重新验证PBC导出，也不是新增动力学。新轨迹完成并提取坐标后可沿用相同审计。

## 四格完成与回传核验

九条均完成11 ns生产，完成清单及轨迹日志回传本地。各阶段无LINCS WARNING/Fatal error；九份complete.json所列输出、96项分析输出、64项分析输入哈希全部通过。见`results/md_factorial/local_transfer_verification.json`。所有三条分支均保留，未按结果追加种子或选择性延长。结果图见`results/md_factorial/analysis/factorial_trajectories.png`；科学解读见[四组对照结果](MD四组对照结果解读.md)。

## M3四格独立核验完成

十二条主体轨迹均完成独立坐标解析及Bio.SVDSuperimposer复算：5472项参考标量、171756项逐帧指标、324行Q网格、648行接触占有率及108行阈值统计均通过。计数与连续采样跨度完全一致，最大逐帧数值误差4.44e-15。29项传输输入/源码哈希一致。证据`results/validation/md_factorial_m3_independent.json`；六worker耗时5.94秒。仍不等于独立动力学复跑或PBC导出验证。

## 本地证据归档

四格原始输入、12条主体轨迹、九条新分支输出、分析与M3审计已归档为`artifacts/md-factorial-evidence.tar.gz`，1,069,945,008字节、466个成员，解压读取逐项哈希核验通过。SHA-256 `4324750be870970efd5556feb957135ca971765f99f14a0569bc0709a8466388`。归档未上传GitHub；普通仓库保留小型结果、协议与代码，原始数据通过此本地证据包单独提供。清单见`results/validation/md_factorial_evidence_manifest.json`。

三页PDF更新完成并逐页渲染审核，输入11项哈希和生成脚本哈希核验通过，QA见`output/pdf/factorial_report_qa.json`。未推送GitHub。
