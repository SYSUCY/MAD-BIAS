# MAD-bias 实验脚本

本目录包含用于运行 MAD-bias 项目实验的各种脚本。

## 实验脚本说明

### 1. 运行所有实验 (`run_experiments.py`)

此脚本按顺序运行所有必要的实验，包括单智能体基线实验、多智能体同质化实验和异质化实验，并生成可视化和报告。

```bash
python scripts/run_experiments.py
```

### 2. 论文实验脚本 (`run_paper_experiments.py`)

此脚本专门针对论文中提到的三个研究问题设计实验：

- **RQ1**: 偏见如何随辩论轮次演变，是否支持群体极化理论？
- **RQ2**: 智能体组合构成（同质化vs异质化）如何影响偏见动态？
- **RQ3**: 哪些语言特征与多智能体辩论中的偏见相关联？

运行所有研究问题实验：
```bash
python scripts/run_paper_experiments.py
```

运行特定研究问题实验：
```bash
python scripts/run_paper_experiments.py --rq 1  # 只运行RQ1相关实验
python scripts/run_paper_experiments.py --rq 2  # 只运行RQ2相关实验
python scripts/run_paper_experiments.py --rq 3  # 只运行RQ3相关实验
```

### 3. Windows批处理脚本 (`run_experiments.bat`)

Windows用户可以使用此批处理脚本运行所有实验：

```bash
scripts\run_experiments.bat
```

## 实验数据

实验数据将保存在以下位置：

- 原始数据: `data/raw/`
- 处理后的数据: `data/processed/`
- 可视化结果: `data/processed/visualization/`
- 实验报告: `data/processed/report.html`

## 日志

实验日志将保存在以下位置：

- 通用日志: `logs/`
- 实验日志: `logs/experiments/`
- 论文实验日志: `logs/paper_experiments/`

## 注意事项

1. 确保已正确配置API设置（在`config/settings.py`中）
2. 实验可能需要较长时间，请耐心等待
3. 如果实验中断，可以使用断点续传功能继续实验
4. 实验结果将自动保存，可以随时生成可视化和报告 