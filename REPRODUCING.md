# 复现与核验指南

请在项目根目录执行命令。下面分为三个层次：检查已保存的小型结果、从原始轨迹重新分析、重新执行实验。三者不等价。

## 1. 不下载大文件：只读检查发布结果

```bash
git clone --branch v1.0.0-method-validation https://github.com/LessRAY3-3/method-validation-mini.git
cd method-validation-mini
python3 scripts/release_smoke.py
```

要求Python 3.9或更新版本，仅使用标准库，不需要GPU、GROMACS、RDKit或模型权重。此入口核对已发布小型证据和统计运算，不启动实验、不覆盖原始结果，也不能替代原始轨迹分析。实际检查范围由程序输出列明，包括30项正式对接姿势文件哈希、四格小型分析文件、720项分支差值、报告来源及M3审核状态。它不会重算57个姿势的几何RMSD。

项目另有数值/逻辑测试及更完整的结构检查：

```bash
python -m unittest discover -s tests -v
python scripts/validate_project.py
```

这两条命令需要科学Python环境；缺少依赖时应配置相应环境，而不能把跳过检查当成验证成功。开发环境包含NumPy、Biopython、RDKit、Matplotlib、spyrmsd等；`config/environment.yml`记录的是macOS Apple Silicon实际环境，含平台相关构建号，不是跨平台通用安装文件。`validate_project.py`不等于所有历史和新实验的统一全流程验收；发布检查与专题验证记录应联合阅读。

## 2. 下载原始MD证据并重新分析

版本页面：[v1.0.0-method-validation](https://github.com/LessRAY3-3/method-validation-mini/releases/tag/v1.0.0-method-validation)。以下为该版本的固定附件路径，发布完成以页面实际附件为准。

| 证据包 | 内容 | SHA-256 |
|---|---|---|
| [四格主体](https://github.com/LessRAY3-3/method-validation-mini/releases/download/v1.0.0-method-validation/md-factorial-evidence.tar.gz) | 12条主体轨迹，冻结输入/源码，九条新增分支输出及分析；1,069,945,008字节 | `4324750be870970efd5556feb957135ca971765f99f14a0569bc0709a8466388` |
| [最初GPU三条](https://github.com/LessRAY3-3/method-validation-mini/releases/download/v1.0.0-method-validation/md-gpu-replicates-evidence.tar.gz) | 更早的三条11 ns；128,033,627字节 | `36024d89a767389573cdf1885cbddf7ae89644317f55a0013fc10507a6a6d28f` |

四格主体已含新种子验证三条A组，不能再将A组重复计数。四格132 ns加最初33 ns等于正式165 ns，不含历史CPU pilot和平衡时间。

在**独立的分析checkout**中下载及解压，避免覆盖个人工作。下面仅取四格包；附件中的SHA-256文件也可在版本页面下载。

```bash
mkdir -p artifacts
curl -fL --retry 3 -o artifacts/md-factorial-evidence.tar.gz \
  https://github.com/LessRAY3-3/method-validation-mini/releases/download/v1.0.0-method-validation/md-factorial-evidence.tar.gz
python3 - <<'PY'
import hashlib
from pathlib import Path
p = Path('artifacts/md-factorial-evidence.tar.gz')
h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
expected = '4324750be870970efd5556feb957135ca971765f99f14a0569bc0709a8466388'
assert h.hexdigest() == expected, 'Checksum mismatch: do not extract'
print('Archive SHA-256 verified')
PY
tar -xzf artifacts/md-factorial-evidence.tar.gz
```

包会恢复相同相对路径下的冻结脚本、参数和数据；不要在已有个人修改的checkout中直接解压。包内文件清单：`results/validation/md_factorial_evidence_manifest.json`；原三条对应`gpu_replicates_evidence_manifest.json`。压缩包已逐成员读取核验，下载后仍应校验压缩包哈希。

### 四格轨迹后处理

需要NumPy、Biopython、Matplotlib及能读原TPR的GROMACS。正式动力学/四格分析使用Ubuntu上的GROMACS 2026.3 CUDA；本地2026.4的若干交叉分析仅证明特定指标一致，不保证任意版本兼容。

```bash
# 输出目录必须不存在；该命令不运行动力学。
python scripts/analyze_md_factorial.py \
  --gmx /path/to/gmx \
  --baseline results/md_replicates_gpu_confirmation \
  --factorial-root results/md_factorial \
  --output-dir results/md_factorial/my_reanalysis
```

每条先使用自身TPR导出完整蛋白，再核对304个蛋白原子的身份和顺序，最后比较共同参考。model24体系水数与model1不同，不能直接套用旧体系的全体系拓扑。新输出与`results/md_factorial/analysis/`中表格按cell、seed、window、metric对照；跨平台浮点误差需要数值容差，不要求PNG或JSON字节相同。

最初GPU三条使用另一证据包，后处理入口为：

```bash
python scripts/analyze_md_replicates.py \
  --input-root results/md_replicates_gpu --gmx /path/to/gmx \
  --total-ns 11 --output-dir results/md_replicates_gpu/my_reanalysis
```

这两条均要求新输出目录。GROMACS与轨迹文件齐全之前，不应把命令列出解释为已经在空白环境完成复跑。

## 3. 重新执行实验：环境和参数均需重建

| 模块 | 正式记录 | 实际入口与注意事项 |
|---|---|---|
| 对接 | `results/docking_benchmark_m3/environment_explicit.txt`；M3 Max/macOS，Vina1.2.7、Meeko0.8、spyrmsd0.9 | `python scripts/run_docking_benchmark.py --help`；先用新目录准备方案，再显式`--run`，不要覆盖15次正式运行 |
| MD | `results/md_replicates_gpu/setup/environment_explicit.txt`；Ubuntu/RTX4070 Laptop，GROMACS2026.3 CUDA | 见`config/experiments/md_factorial_protocol.json`及`report/MD起点与平衡对照方案.md`；先恢复A基线与model24输入，B/C/D有checkpoint依赖 |
| 预测 | `config/colabfold-requirements.txt`及各运行provenance；保存A3M、输入与处理参数 | `python scripts/run_prediction_ablation.py --help`；需要权重与兼容JAX/ColabFold环境，使用新输出目录 |

MD冻结驱动`run_md_factorial_queue.py`会检查源文件、输入和GROMACS二进制哈希。它用于原环境的受控执行/续跑，**不是可在任意机器跳过校验的一键重跑脚本**。新机器若二进制或输入不同，须创建另一个实验目录和明确记录的新协议，保留原协议及基线，不移除检查后冒充同一实验。短/长NPT分支共享NVT和短NPT历史，不能用重新生成速度的独立长NPT任务替代。

`bash scripts/bootstrap.sh`及`run_md.sh`、`run_docking.sh`是早期macOS基础练习入口，不会自动产生本版本的15次对接或四格MD。不要把历史单轨迹或四次基础对接与正式结果混并。详细参数及输入选择见[技术报告](report/技术报告.md)、[输入准备与核验](report/输入准备与核验.md)。

软件环境记录中可能保留原执行主机绝对路径以便追溯；需要按新机器配置路径。原始在线MSA服务、ESMFold接口和权重下载的未来可用性不保证。固定seed不能消除软件、硬件、输入形状与处理路径变化。

## 审核记录如何理解

- `results/md_factorial/local_transfer_verification.json`：本次回传文件哈希核验。
- `results/validation/md_factorial_m3_independent.json`：独立解析已导出的蛋白坐标并用Bio.SVDSuperimposer复算；不是新动力学，也不独立核验PBC导出。
- `results/validation/md_factorial_evidence_archive.json`：打包时的本地证据包核验；其中“not uploaded”是当时状态，发布状态以Release页面为准。
- `output/pdf/report_provenance.json`：三页报告的数据来源及哈希。

全部代码由现有科学软件构建，并有AI辅助实现和审核；用于个人实践展示的步骤应以本人能够解释、操作及复算的范围为准。
