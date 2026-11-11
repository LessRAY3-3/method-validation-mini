"""Build the three-page report from versioned numerical outputs.

Run: .conda/bin/python scripts/build_report_pdf.py
REPORT_FONT may name a Chinese-capable TrueType font. No simulation is run here.
"""
import csv
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/method-validation-report.pdf'
NAVY = colors.HexColor('#17324d')
TEAL = colors.HexColor('#147d86')
GRAY = colors.HexColor('#526372')
W, H = A4
LEFT, WIDTH = 45, W - 90


def load_json(path):
    return json.loads((ROOT / path).read_text())


def read_csv(path):
    with (ROOT / path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def register_font():
    candidates = [os.environ.get('REPORT_FONT', ''),
                  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                  '/usr/share/fonts/truetype/arphic/uming.ttc']
    for value in candidates:
        if value and Path(value).is_file():
            pdfmetrics.registerFont(TTFont('ReportCJK', value))
            return 'ReportCJK'
    # Standard PDF CJK fallback: viewer must supply the corresponding CJK font.
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


FONT = register_font()
BODY = ParagraphStyle('Body', fontName=FONT, fontSize=10, leading=16,
                      textColor=NAVY, wordWrap='CJK')
SMALL = ParagraphStyle('Small', parent=BODY, fontSize=8.2, leading=12,
                       textColor=GRAY)


def para(text, y, x=LEFT, width=WIDTH, small=False):
    p = Paragraph(text, SMALL if small else BODY)
    _, height = p.wrap(width, H)
    if y - height < 52:
        raise ValueError(f'Content overflows page: {text[:45]}')
    p.drawOn(C, x, y - height)
    return y - height - 9


def title(page, heading, subtitle):
    C.setFillColor(TEAL)
    C.rect(LEFT, H - 49, 34, 4, fill=1, stroke=0)
    C.setFillColor(NAVY)
    C.setFont(FONT, 20)
    C.drawString(LEFT, H - 82, heading)
    C.setFont(FONT, 9)
    C.setFillColor(GRAY)
    C.drawString(LEFT, H - 104, subtitle)
    C.setStrokeColor(colors.HexColor('#d7e2e6'))
    C.line(LEFT, 44, W - LEFT, 44)
    C.setFont(FONT, 8)
    C.drawString(LEFT, 30, 'METHOD VALIDATION MINI  |  2026-11-11  |  方法验证型教学项目')
    C.drawRightString(W - LEFT, 30, f'{page} / 3')
    return H - 127


def section(label, y):
    C.setFillColor(TEAL)
    C.setFont(FONT, 12)
    C.drawString(LEFT, y - 12, label)
    return y - 25


def table(rows, y, widths):
    cells = [[Paragraph(escape(str(v)), SMALL) for v in row] for row in rows]
    t = Table(cells, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f1f3')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    _, h = t.wrap(WIDTH, H)
    if y - h < 52:
        raise ValueError('Table overflows page')
    t.drawOn(C, LEFT, y - h)
    return y - h - 12


def figure(path, y, height, width=WIDTH, x=LEFT):
    C.drawImage(str(ROOT / path), x, y - height, width=width, height=height,
                preserveAspectRatio=True, anchor='c', mask='auto')
    return y - height - 8


def md_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    dest = ROOT / 'output/pdf/assets/md_factorial_report.png'
    dest.parent.mkdir(exist_ok=True)
    rows = read_csv('results/md_factorial/analysis/per_trajectory_windows.csv')
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for ax,key,label in zip(axs, ('sidechain_min_3_19_angstrom','core_ca_rmsd_nmr1_angstrom'),
                           ('Y3-P19 sidechain distance (A)','Core CA RMSD / NMR1 (A)')):
        for seed in (20261113,20261114,20261115):
            vals = [float(next(r['value'] for r in rows if r['cell']==c and int(r['seed'])==seed and r['window_ns']=='1-3' and r['metric']==key)) for c in 'ABCD']
            ax.scatter(range(4),vals,s=35,label=str(seed))
        ax.set_xticks(range(4),['A: M1 / short','B: M1 / long','C: M24 / short','D: M24 / long'],rotation=15)
        ax.set_ylabel(label);ax.set_title('Fixed early window: 1-3 ns')
        ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(dest,dpi=200);plt.close(fig)
    return dest


def build():
    global C
    OUT.parent.mkdir(parents=True, exist_ok=True)
    C = canvas.Canvas(str(OUT), pagesize=A4, invariant=1)
    C.setTitle('计算结构生物学方法验证 - 三页技术报告')
    C.setAuthor('method-validation-mini')
    md_plot = md_figure()
    docking = load_json('results/docking_benchmark_m3/summary.json')
    md = read_csv('results/md_factorial/analysis/descriptive_summary.csv')
    followup = load_json('results/prediction_msa_diagnosis/followup/summary.json')

    y = title(1, '计算结构生物学方法验证', '01  |  共晶重对接：搜索强度与随机种子')
    y = para('问题：参数增加是否提高重复成功率？以 1IEP / imatinib 重对接、1L2Y 短肽 MD 和结构预测分别检验流程。它们是不同体系的方法验证，不组成药物研发或结合机制验证管线。', y)
    y = section('固定输入与配对设计', y)
    y = para('沿用 Vina 官方教程的加氢 A 链受体与保留键级的配体 SDF，以 Meeko 制备 PDBQT；不声称完成质子化枚举。来源/哈希见 data/raw/SOURCES.tsv。Vina 1.2.7、Meeko 0.8.0、spyrmsd 0.9.0；M3 Max，同环境新跑全批次。', y)
    y = para('搜索盒 20 × 20 × 20 Å，中心 (15.190, 53.903, 16.917) Å；每次 cpu=1，两个任务并行，num_modes=9、energy_range=3。强度 1/8/32 各用 seed 42、2026、20261110/11/12，共 15 次。成功预定为晶体重原子 RMSD ≤ 2 Å。', y)
    rows = [['搜索强度', 'Top-1 / 任一保存姿势命中', 'RMSD中位数 / Å', '耗时中位数 / s']]
    for g in docking['groups']:
        rows.append([g['exhaustiveness'], f"{g['top1_successes']}/5 / {g['any_saved_successes']}/5", f"{g['median_top1_rmsd_angstrom']:.3f}", f"{g['median_vina_elapsed_seconds']:.1f}"])
    y = table(rows, y, [75, 190, 115, WIDTH-380])
    y = figure('results/docking_benchmark_m3/paired_seed_outcomes.png', y, 187)
    y = para('57 个保存姿势均经 spyrmsd 与 RDKit图映射+NumPy 双实现核验，差异小于 1e-8 Å。保持受体坐标系，不单独叠合配体。1/5与5/5的Wilson 95%区间为3.6%-62.4%与56.6%-100%；n=5仅作单体系描述。', y, small=True)
    y = section('解释与失败点', y)
    y = para('强度1有一条达到0.281 Å，却有四条失败；只展示最好一次会掩盖不稳定性。强度8至32未增加成功数，耗时约四倍。未观察到Top-1失败但任一保存姿势成功的案例；输出经能量窗口筛选，不能反推完整搜索历程。高残基接触恢复也不保证配体取向正确。', y)
    para('证据：results/docking_benchmark_m3/ 下的 protocol.json、all_poses.csv、run_summary.csv、environment_explicit.txt 与 local_verification.json。耗时受并行负载影响，不作跨机器性能比较。', y, small=True)
    C.showPage()

    y = title(2, '短肽 MD：起点与准备时长对照', '02  |  四格 × 三个种子 × 11 ns；四格132 ns，正式总量165 ns')
    y = para('Trp-cage / 1L2Y；GROMACS 2026.3 CUDA，Ubuntu / RTX 4070 Laptop；AMBER99SB-ILDN / TIP3P，2 fs，PME，300 K V-rescale、1 bar C-rescale。两批六条初速度重复后，开展起点 × 有位置约束NPT时长对照。', y)
    y = para('A/B为NMR模型1，C/D为模型24；A/C为100 ps NPT，B/D为1000 ps。均先100 ps有约束NVT，再11 ns无约束生产。每格seed20261113/14/15：A复用三条，B/C/D新增九条；另保留原探索三条。短/长分支共享NPT历史，不是12个独立起点。', y)
    y = section('Y3-P19侧链最短距离：各格三条时间均值的平均 / Å', y)
    rows = [['条件', '1-3 ns', '8-11 ns']]
    for cell,label in [('A','A：模型1 / 短NPT'),('B','B：模型1 / 长NPT'),('C','C：模型24 / 短NPT'),('D','D：模型24 / 长NPT')]:
        vals = [float(next(r['mean'] for r in md if r['kind']=='cell' and r['group']==cell and r['window_ns']==w and r['metric']=='sidechain_min_3_19_angstrom')) for w in ('1-3','8-11')]
        rows.append([label,*[f'{v:.3f}' for v in vals]])
    y = table(rows,y,[245,130,WIDTH-375])
    y = figure(md_plot, y, 150)
    y = para('图中每点为一条轨迹的窗口均值。早期组均值：B低于A，D高于C；内部Cα在全部38个NMR参考中方向一致，Q在9套接触定义中也支持相应方向。反例仍保留：B的seed15侧链>6 Å比例反而由A的42.5%升至69.5%。完整时间序列/RMSF/Rg/氢键及阈值网格另存。', y, small=True)
    y = section('结论与限制', y)
    y = para('早期偏离对准备条件的响应依赖所选起点；不能说延长NPT普遍改善结果。每格仅n=3，局部偏离可在其他时段再现，不能仅凭末段接近宣称收敛。模型24重溶剂化后为2526水，模型1为2678水，离子放置也不同，故不是纯粹的蛋白起点因果效应。', y)
    para('有约束NPT不证明蛋白内部充分平衡。九条新增轨迹/分析哈希核验通过，无LINCS警告；M3独立复算12条、5472项参考与171756项逐帧指标通过。证据：results/md_factorial/analysis/、config/experiments/md_factorial_protocol.json及report/MD四组对照结果解读.md；帧和参考模型不增加独立样本数。', y, small=True)
    C.showPage()

    y = title(3, '预测异常：复现、干预与边界', '03  |  极浅MSA敏感性；不是模型通用缺陷')
    y = para('ColabFold 1.6.3 / alphafold-colabfold 2.3.20，model_1_ptm、CPU、无模板/弛豫。对同一20 aa序列完成query-only、去重MSA、原始MSA各6个种子（18次主比较）；另有5次针对性干预、1次逐recycle坐标追踪。异常后的分析为探索性诊断。', y)
    y = para('去重输入某seed异常可精确复现：坐标、pLDDT、PAE与pTM逐值一致。对全部38个NMR模型逐一叠合，不挑最有利参考；内部3-18位单独拟合仍异常，排除仅由末端或整体叠合造成的假象。38个NMR模型不是38次独立性能实验。', y)
    rows = [['同一异常seed的干预', '平均pLDDT', 'NMR集合RMSD中位数 / Å']]
    names=['精确复跑去重Q,H','重复query：Q,Q,H','重复homolog：Q,H,H','原始输入限制容量2:1','去重输入，12次recycle']
    for name,r in zip(names,followup['targeted_runs']):
        rows.append([name,f"{r['mean_plddt']:.2f}",f"{r['median_ca_rmsd_angstrom']:.3f}"])
    y=table(rows,y,[230,100,WIDTH-330])
    y=figure('results/prediction_msa_diagnosis/recycle_trace/structural_recycle_trace.png',y,180)
    y=para('保存阶段0-12的坐标，确认第3阶段重现原异常、第12阶段重现后续较好结果；第6-8阶段偏差明显下降。这是模型迭代，不是物理时间、MD或真实折叠路径。pLDDT描述局部置信度，PAE描述相对定位不确定性，两者都不能替代实验结构对照。',y,small=True)
    y=section('源码追踪支持什么',y)
    y=para('本地预处理使原始Q,Q,H,H成为Q,H,H；去重为Q,H。45组纯特征追踪显示重复行权重与真实MSA形状会改变特征及随机mask；同seed不保证不同形状输入mask相同。六seed中去重仅原一条异常，不能说MSA有害；更多recycle也带来更多重采样，因果贡献尚未完全隔离。',y)
    para('复算：scripts/analyze_msa_followup.py、trace_msa_recycling.py；输入/源码/权重哈希与完整异常见results/prediction_msa_diagnosis/。项目含历史ESMFold基础运行，但query-only AlphaFold不是ESMFold，也不构成训练集外模型排名。',y,small=True)
    C.save()
    import hashlib
    inputs = ['results/docking_benchmark_m3/summary.json',
              'results/md_factorial/analysis/summary.json',
              'results/md_factorial/analysis/descriptive_summary.csv',
              'results/md_factorial/analysis/per_trajectory_windows.csv',
              'config/experiments/md_factorial_protocol.json',
              'results/validation/md_factorial_m3_independent.json',
              'results/md_factorial/local_transfer_verification.json',
              'results/docking_benchmark_m3/paired_seed_outcomes.png',
              'results/prediction_msa_diagnosis/recycle_trace/structural_recycle_trace.png',
              'results/prediction_msa_diagnosis/followup/summary.json',
              'results/prediction_msa_diagnosis/recycle_trace/summary.json']
    provenance = {'report': str(OUT.relative_to(ROOT)),
                  'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
                  'input_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
                  'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'md_analysis': 'Ubuntu CUDA GROMACS2026.3 four-cell analysis; independent M3 cached-coordinate audit; 12 branches with shared short/long histories',
                  'figure_sha256': hashlib.sha256(md_plot.read_bytes()).hexdigest()}
    (OUT.parent/'report_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    build()
