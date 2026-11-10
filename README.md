# 计算方法验证型小项目

这是一个教学项目：用共晶重对接检查 Vina 流程，用 1 ns 短肽 MD 练习标准分析，再学习结构预测置信度的边界。它不是完整药物研发管线，也不用于提出候选药物。

## 数据和问题

- `1IEP`：c-Abl 激酶与共晶配体 imatinib（STI）。问题是“把配体取出后，能否用同一套准备与搜索流程找回应有的结合姿势”。
- `1L2Y`：20 aa Trp-cage 短肽，第 1 个 NMR 模型。问题是“如何得到一条可分析的轨迹，并正确描述它的限制”。
- 同一短肽序列已分别用 ESMFold 和 ColabFold/AlphaFold2 实际预测。预测坐标不是实验结构，也不是结合机制证据。

原始文件来源和 SHA-256 见 `data/raw/SOURCES.tsv`。

## 环境

在 macOS arm64 上使用项目内 micromamba 环境和 Homebrew 的 GROMACS/PyMOL：

```bash
bash scripts/bootstrap.sh
```

直接对系统 Python 3.9 执行 `pip install vina` 会退回源码构建，并因缺少 Boost 失败；因此使用带原生 arm64 二进制的 conda-forge 包。实际版本和完整依赖见 `report/软件版本.txt` 与 `config/environment.yml`（环境完成后生成）。

## 第 1—2 天：重对接

### 原理

- 受体准备决定原子类型、氢和电荷；配体准备必须保留可靠键级、质子化与三维坐标。
- 搜索盒限定 Vina 搜索的空间。盒子应覆盖结合位点，但过大会增加无效搜索。
- `exhaustiveness` 控制搜索努力，不是打分函数的“准确度旋钮”。
- Vina score 用于姿势排序；重对接是否找回晶体姿势由相对共晶配体的重原子 RMSD 判断。输出表中的 `rmsd l.b./u.b.` 只是不同 docking mode 间的距离，不是晶体 RMSD。

### 操作

```bash
bash scripts/prepare_docking.sh
bash scripts/run_docking.sh
pymol -cq scripts/visualize_docking.pml
```

固定盒子为 `20 × 20 × 20 Å`，中心 `(15.190, 53.903, 16.917)`。运行 `exhaustiveness=32` 的三个种子，并用种子 42 比较搜索强度 8 与 32。`num_modes=9` 是上限；默认 `energy_range=3 kcal/mol` 过滤后实际保存 4/4/6/4 个，共 18 个姿势。

### 检查点

- `data/prepared/` 中有受体/配体 PDBQT 和盒子文件。
- `results/docking/all_poses.csv` 同时列出打分和晶体 RMSD。
- `run_summary.csv` 分开报告最高打分姿势的 RMSD 与最低 RMSD 姿势。
- `figures/docking_overlay.png` 中晶体配体为品红，预测为青色；另检查 `docking_box.png`。
- 2 Å 只是常用参考线。超过时不要隐藏，应检查准备、搜索、配体柔性和打分误差。

## 第 3—4 天：短肽 MD

### 原理

力场定义势能函数；溶剂、周期性边界和反离子定义模拟体系。最小化用于消除不合理接触，NVT/NPT 用于温控和压控平衡，生产轨迹才用于后续分析。2 fs 步长配合含氢键约束，PME 用于长程静电。

```bash
bash scripts/run_md.sh
bash scripts/extend_md.sh
bash scripts/analyze_md_extended.sh
bash scripts/extend_md.sh md_extended md_extended2 5000
TOTAL_NS=11 TAG=_11ns bash scripts/analyze_md_extended.sh
```

AMBER99SB-ILDN/TIP3P、1.0 nm 最短溶质—盒壁距离、100 ps NVT、100 ps NPT，300 K、1 bar。先跑 1 ns，再从检查点连续延长到 6 ns 和 11 ns；所有参数见 `config/*.mdp`。

### 分析前的必要处理

周期性边界会让完整分子在坐标文件中看似断开，因此先做 `trjconv -pbc mol -center`。再用 `-fit rot+trans` 统一坐标系，主要用于 RMSF 和可视化；`gmx rms` 自身也会做最小二乘拟合。之后计算：

- 骨架 RMSD：相对生产起始结构（NPT 末态）的整体偏差。
- Cα RMSF：逐残基围绕平均位置的波动，不等于 B-factor。
- 回旋半径：整体紧致程度，不直接等于折叠状态。
- 蛋白内部氢键数：依赖距离/角度判据，不直接等于结合自由能。

1 ns 时 RMSD 后半段上升；6 ns 时最后三个窗口均值为 0.228、0.175、0.112 nm，仍未平台化。延长到 11 ns 后，8–9、9–10、10–11 ns 的 RMSD 均值为 0.099、0.090、0.091 nm，窗口内漂移减弱，显示末段趋稳。**这只是平台化迹象，不证明热力学收敛、折叠或结合机制。**

## 第 5 天：结构预测置信度

实际 ESMFold 请求与 ColabFold CPU 运行：

```bash
curl -fsSL -X POST --data 'NLYIQWLKDGGPSSGRPPPS' \
  https://api.esmatlas.com/foldSequence/v1/pdb/ \
  -o results/prediction/esmfold_1L2Y.pdb
bash scripts/setup_colabfold.sh
bash scripts/run_colabfold.sh
```

- pLDDT 是逐残基局部结构置信度，通常越高越好；ColabFold PDB 将其写在 B-factor 字段。本 ESMFold 接口观测到 0–1 范围，按 ESMFold 约定转换为 0–100。
- PAE 描述残基对相对位置的预计误差，单位是 Å，通常越低越好；它特别适合判断不同区域/结构域的相对摆放是否可信。
- 本例 ESMFold 平均 pLDDT 为 91.15；ColabFold 为 85.30，但 pTM 仅 0.11，平均 PAE 3.85 Å、最大 14.70 Å，且 MSA 很浅（实际 max_seq=4、max_extra_seq=1）。这说明不能只看 pLDDT。
- ESMFold 是单序列模型，不使用 MSA；MSA coverage 仅用于 ColabFold 输出。
- 即使 pLDDT 高、PAE 低，也不能直接推出配体结合构象、侧链口袋、质子化、关键水或动态机制。
- 本 ESMFold API 不提供 PAE，因此记为“不适用”，不补造矩阵。

ColabFold 官方 notebook：<https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb>。本次实际使用 1 个 `alphafold2_ptm` 模型、3 次 recycle、固定种子 20261110，无模板、无 Amber relaxation。

## 交付

- `results/docking/`：所有姿势、RMSD/打分表和运行摘要。
- `results/md/`：拓扑、轨迹、四项分析和数值摘要。
- `figures/`：可检查的图像。
- `report/技术报告.md`：2—3 页正文，含参数、对照、失败点和限制。
- `report/简历与面试话术.md`：方法验证型项目表述，不包装产业经验。
