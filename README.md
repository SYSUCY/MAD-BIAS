# MAD-bias: 多智能体辩论中的偏见研究

## 项目介绍
本项目旨在研究多智能体辩论场景下的偏见现象，验证群体极化效应对智能体偏见的影响。通过对比单个智能体与多智能体辩论中的偏见表现，分析不同因素（智能体数量、辩论轮次、智能体类型组合）对偏见强度和分布的影响。

## 研究假设
* 主假设：多智能体辩论场景下的偏见会因群体极化效应而强于单个智能体
* 备择假设：多智能体辩论可能通过相互制衡减弱偏见

## 项目结构
```
MAD-bias/
├── README.md                     # 项目说明文档
├── requirements.txt              # 项目依赖
├── config/                       # 配置文件
│   ├── models.py                 # 模型配置
│   ├── topics.py                 # 话题和提示词
│   └── settings.py               # 实验设置
├── src/                          # 源代码
│   ├── agents/                   # 智能体模块
│   │   ├── base_agent.py         # 基础智能体类
│   │   └── model_agent.py        # 不同模型智能体实现
│   ├── debate/                   # 辩论模块
│   │   ├── single_agent.py       # 单智能体回答
│   │   └── multi_agent.py        # 多智能体辩论
│   ├── evaluation/               # 评估模块
│   │   ├── bias_detector.py      # 偏见检测
│   │   ├── metrics.py            # 评估指标
│   │   └── qualitative.py        # 定性分析
│   ├── experiment/               # 实验模块
│   │   ├── controller.py         # 实验控制器
│   │   └── runner.py             # 实验运行器
│   └── visualization/            # 可视化模块
│       ├── charts.py             # 图表生成
│       └── data_process.py       # 数据处理
├── data/                         # 数据文件夹
│   ├── raw/                      # 原始回复
│   └── processed/                # 处理后的数据
└── scripts/                      # 脚本
    ├── run_experiment.py         # 运行实验
    ├── analyze_results.py        # 分析结果
    └── generate_charts.py        # 生成图表
```

## 使用方法
1. 安装依赖：
```
pip install -r requirements.txt
```

2. 配置API密钥：
在环境变量中设置 `OPENAI_API_KEY` 或直接修改 `config/settings.py` 中的配置。

3. 运行实验：
```
python scripts/run_experiment.py
```

4. 分析结果：
```
python scripts/analyze_results.py
```

5. 生成可视化图表：
```
python scripts/generate_charts.py
```

## 实验设计
本项目实现了完整的实验流程，包括：
1. 单智能体和多智能体在争议性话题上的观点表达
2. 不同轮次(1、2、3、4、5轮)的辩论过程
4. 同质化和异质化智能体组合的对比
5. 偏见强度评分、偏见类型分布和观点极化程度的测量与分析

## 数据分析与可视化
项目提供了全面的数据分析工具，可以：
1. 比较不同实验设置下的偏见强度
2. 分析偏见与智能体数量的关系
3. 研究偏见与辩论轮次的关系
4. 可视化偏见随辩论进行的演变过程

## 注意事项
- 本项目需要使用大语言模型API，请确保有足够的API使用额度
- 完整运行所有实验可能需要较长时间和一定的计算资源 
