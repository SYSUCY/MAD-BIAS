# -*- coding: utf-8 -*-
"""
项目设置文件
定义API设置和实验参数
"""

import os
from pathlib import Path
import random
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# OpenAI API 设置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR API KEY")

# API端点设置
# 提供多个API端点选项，当一个被封禁时可以切换到另一个
API_ENDPOINTS = {
    "chatanywhere_tech": "https://api.chatanywhere.tech/v1",
    "chatanywhere_org": "https://api.chatanywhere.org/v1",
    "chatanywhere_cn": "https://api.chatanywhere.cn/v1",
    # "official_openai": "https://api.openai.com/v1",  # 官方API
    # "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", ""),  # Azure OpenAI API
}

# 默认API端点
API_BASE_URL = os.getenv("API_BASE_URL", API_ENDPOINTS["chatanywhere_org"])

# 实验设置
EXPERIMENT_SETTINGS = {
    # 辩论轮次
    "debate_rounds": [1, 3, 5],
    
    # 智能体数量
    "agent_counts": [1, 2, 3, 4],
    
    # 默认温度参数
    "temperature": 0.7,
    
    # 随机种子
    "random_seed": 42,
    
    # 实验重复次数 - 增加重复次数以提高统计显著性
    "num_repeats": 5,  # 从3增加到5
    
    # 每次实验使用话题数量
    "topics_per_experiment": 8,  # 从5增加到8
    
    # 最大并行请求数 - 减少并行请求以提高稳定性
    "max_parallel_requests": 1,  # 从2减少到1，避免API限制
    
    # 请求超时时间(秒) - 增加超时时间
    "request_timeout": 120,  # 从60增加到120
    
    # 请求重试次数 - 增加重试次数
    "max_retries": 10,  # 从8增加到10
    
    # 重试延迟(秒) - 增加重试延迟
    "retry_delay": 10,  # 从5增加到10
    
    # API失败自动切换
    "auto_switch_api": True,
    
    # API封禁检测关键词
    "api_block_keywords": [
        "Restricted Access", 
        "访问受限", 
        "blocked", 
        "rate limit", 
        "timeout", 
        "too many requests",
        "请求过多",
        "请求频率过高",
        "请稍后再试"
    ],
    
    # 进度保存频率（每处理多少个请求保存一次进度）
    "save_progress_frequency": 2,  # 从3减少到2，更频繁保存
    
    # 是否启用断点续传
    "enable_resume": True,
    
    # 辩论质量控制 - 新增参数
    "quality_control": {
        # 最小响应长度（字符）
        "min_response_length": 100,
        
        # 最大响应长度（字符）
        "max_response_length": 2000,
        
        # 是否检查响应质量
        "check_response_quality": True,
        
        # 低质量响应重试次数
        "low_quality_retries": 2
    },
    
    # 数据收集设置 - 新增参数
    "data_collection": {
        # 是否收集中间状态
        "collect_intermediate_states": True,
        
        # 是否保存完整对话历史
        "save_full_conversation": True,
        
        # 是否记录API调用详情
        "log_api_details": True
    }
}

# 初始化随机种子
random.seed(EXPERIMENT_SETTINGS["random_seed"])

# 实验结果保存路径
RESULTS_PATH = DATA_PROCESSED_DIR / "results.csv"

# 日志设置
LOG_DIR = ROOT_DIR / "logs"
LOG_LEVEL = "INFO"

# 可视化设置
VISUALIZATION_SETTINGS = {
    "figure_size": (12, 8),
    "dpi": 300,
    "font_size": 12,
    "color_palette": "viridis",
    "save_format": "png",
    
    # 新增可视化设置
    "bias_color_map": "RdYlBu_r",  # 偏见强度颜色映射
    "show_error_bars": True,       # 显示误差条
    "show_trend_lines": True,      # 显示趋势线
    "export_formats": ["png", "pdf", "svg"],  # 导出格式
    "interactive_plots": True      # 是否生成交互式图表
}

# 创建必要的目录
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True) 