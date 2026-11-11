# 三条独立初速度MD：复核结果与解释

2026-11-10。正式实验均在Ubuntu RTX 4070 Laptop GPU、GROMACS2026.3 CUDA上完成；每条100 ps NVT、100 ps NPT、11 ns生产，无新增模拟。三个seed为20261110/11/12。原始远端分析保存在`results/md_replicates_gpu/aggregate_analysis/`，本次补充/复核位于[`aggregate_analysis_audited`](../results/md_replicates_gpu/aggregate_analysis_audited/)。

## 审核结论和修复

三条生产轨迹均完整，1101帧覆盖0–11000 ps、间隔10 ps；原始XTC/TPR传回本地后的SHA-256与远端记录一致。没有删去不利轨迹、没有只延长某个seed。

发现并修复两个分析交付缺口：

- 旧`heavy_min`实际为**整残基所有重原子**的最短距离，包含主链，不能称为侧链接触。保留旧字段、原结果，新增`sidechain_min`明确排除N/CA/C/O/OXT/OT1/OT2。补充指标是对已声明侧链分析的实现纠正，不重新挑选位点。
- 旧结果仅汇总8–11 ns及其中三个1 ns窗口，虽然全程时间序列和RMSF已保存，但缺少预定0–11 ns汇总。现补全全程表格；没有用末段替代全程。

RMSD采用共同历史NPT末态参考，以Backbone拟合并测量；三条没有各自改用自己的NPT末态。另列Cα对NMR model1的拟合RMSD。RMSF则将Cα以proper Kabsch拟合NMR model1，再计算各窗口相对自身时间均值的波动，单位nm；这与历史gmx rmsf默认再拟合口径不同，不直接混表。所有内部窗口左闭右开，最后含11 ns端点。

## 重复真实性和模拟健康

检查三份NVT TPR中的初始坐标与速度数组：坐标相同、速度数组三者不同。NVT的`gen_vel=yes`、`gen_seed`和`ld-seed`分别等于各自seed；NPT和production均`gen_vel=no`，继续对应速度，`ld-seed`仍分别对应各seed。这是相同最小化构象的独立初速度重复，不是三种独立起始构象。

全部九个阶段的日志均确认PP短程及PME在GPU上执行，LINCS WARNING与Fatal error均为0。全程能量记录每1 ps保存，三条各11001样本，全部有限值。三条温度均值299.982、299.941、299.985 K；时间序列SD约3.55 K。压力均值1.58、−0.80、−1.05 bar，时间序列SD约259–264 bar；不能把瞬时压力的较大起伏、或均值未恰好等于1 bar直接当作数值故障，也不能据此宣称压力统计收敛。未进行有效样本数/相关时间修正，SD不代表均值误差。

## 全程和末段并列结果

下表每格都是**一条轨迹的时间均值**。统计独立单位为轨迹，n=3；1101帧不能算1101个独立重复。

| seed | 全程RMSD nm | 8–11 ns RMSD nm | 全程Q8 | 8–11 ns Q8 | 全程Y3–P19侧链距离 Å | 8–11 ns侧链距离 Å |
|---|---:|---:|---:|---:|---:|---:|
|20261110|0.10638|0.09285|0.94663|0.95027|5.0733|4.5986|
|20261111|0.09343|0.08838|0.96199|0.96396|4.3718|4.4297|
|20261112|0.09333|0.08536|0.96840|0.97231|4.2925|4.4137|

Q8为NMR model1中Cα间距≤8 Å、序列间隔至少3位的接触，轨迹距离≤参考距离1.2倍则记为存在。7/9 Å敏感性分析也保留。Q不是折叠概率或自由能。

末段三条主链RMSD均值较接近，Q维持较高，支持这三次短时模拟保持接近参考的整体构象。但第一条全程RMSD更高，Y3–P19侧链距离也更大；不能因为末段接近就抹去全程差异。NMR model1的Y3–P19侧链最短重原子距离为3.975 Å，末段三条为4.414–4.599 Å，明显不同于旧全残基距离3.854–3.944 Å。该差异说明主链参与的最短距离可能掩盖侧链几何变化，不证明某个化学相互作用或机制。

全程Cα RMSF的最高三条均值位于残基20（0.1466 nm）、1（0.1223 nm）、15（0.0998 nm）；末段仍以20和1最高。拟合后的端部波动较大，但没有使用RMSF阈值定义普遍稳定性或动力学机制。

## 独立核验和复现

本地GROMACS2026.4重新分析相同三份正式轨迹；正式动力学始终全部在Ubuntu2026.3执行，分析版本分开记录。与原逐帧结果相比：主链RMSD最大差≤1e-7 nm，Rg≤1e-6 nm，Q与全重原子距离相同，氢键数完全一致。GRO导出保留0.001 nm坐标精度，不能赋予更多小数物理意义。

每条抽取0、8、11 ns，用Bio.SVDSuperimposer独立检验Cα拟合RMSD，并用显式残基对循环检验Q8；9帧全部一致，RMSD最大数值差<1e-14 Å。见[`independent_audit.json`](../results/md_replicates_gpu/aggregate_analysis_audited/independent_audit.json)。4项MD单元测试通过，包含刚体不变性、重叠/缺失帧拒绝、缺失重复拒绝和主链/侧链区分。

```bash
.conda/bin/python scripts/analyze_md_replicates.py --gmx gmx \
  --input-root results/md_replicates_gpu \
  --output-dir results/md_replicates_gpu/aggregate_analysis_recheck --keep-coordinates
# audit_md_outputs.py核验已有aggregate_analysis_audited，要求保留protein_whole.gro
.conda/bin/python scripts/audit_md_outputs.py --gmx gmx
```

完整数值见`per_trajectory_windows.csv`、`trajectory_level_summary.csv`、`per_residue_rmsf.csv`及`residue_trajectory_level_summary.csv`。不能从这33 ns推断热力学收敛、折叠速率、真实药效或结合机制；也不将历史Mac单条11 ns混入这组三重复。
