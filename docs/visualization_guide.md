# MAD-bias 可视化与报告生成指南

本文档提供了如何生成实验结果可视化图表和报告的详细说明。

## 前提条件

确保已安装以下Python库：
- pandas
- numpy
- matplotlib
- seaborn
- jinja2

可以使用以下命令安装：
```bash
pip install pandas numpy matplotlib seaborn jinja2
```

## 可视化图表生成

### 一键生成所有可视化和报告

最简单的方式是使用一键生成脚本：

```bash
python scripts/generate_all.py --open_report
```

这将自动生成所有可视化图表，并创建一个HTML报告，然后自动打开该报告。

### 命令行参数

一键生成脚本支持以下参数：

- `--data_dir`: 数据目录
- `--output_dir`: 输出目录
- `--single_dir`: 单智能体实验数据目录
- `--multi_dir`: 多智能体实验数据目录
- `--report_path`: 报告输出路径
- `--open_report`: 生成后自动打开报告

例如，指定特定的数据目录和输出目录：

```bash
python scripts/generate_all.py --data_dir data/processed --output_dir data/visualization --report_path data/report.html --open_report
```

## 单独生成可视化图表

如果只需要生成可视化图表，可以使用：

```bash
python scripts/generate_visualizations.py --type all
```

支持的参数：

- `--type`: 要生成的可视化类型，可选值为 `single`（单智能体）, `multi`（多智能体）或 `all`（全部）
- `--data_dir`: 数据目录
- `--output_dir`: 输出目录
- `--single_dir`: 单智能体实验数据目录
- `--multi_dir`: 多智能体实验数据目录

## 单独生成报告

如果只需要生成报告，可以使用：

```bash
python scripts/generate_report.py
```

支持的参数：

- `--output`: 报告输出路径
- `--single_dir`: 单智能体实验数据目录
- `--multi_dir`: 多智能体实验数据目录
- `--vis_dir`: 可视化图表目录

## 生成的文件

### 可视化图表

默认情况下，可视化图表将保存在 `data/processed/visualization` 目录下，包括：

- `bias_by_model.png`: 不同模型的偏见强度对比图
- `bias_by_topic.png`: 不同话题的偏见强度对比图
- `bias_vs_agent_count.png`: 智能体数量与偏见强度关系图
- `bias_vs_debate_rounds.png`: 辩论轮次与偏见强度关系图

### 报告

默认情况下，报告将保存为 `data/processed/report.html`，这是一个包含所有可视化图表和数据分析的HTML文件。

## 数据要求

为了生成完整的可视化和报告，实验数据应包含以下信息：

### 单智能体实验数据
- `model`: 模型名称
- `topic`: 话题
- `bias_strength`: 偏见强度

### 多智能体实验数据
- `agent_count`: 智能体数量
- `debate_rounds`: 辩论轮次
- `final_bias_strength`: 最终偏见强度
- `homogeneous`: 是否为同质化智能体组合（可选）

## 故障排除

如果遇到问题，请检查以下日志文件：

- `logs/visualization.log`: 可视化生成日志
- `logs/report.log`: 报告生成日志
- `logs/generate_all.log`: 一键生成脚本日志

这些日志文件包含详细的错误信息，可以帮助诊断问题。 