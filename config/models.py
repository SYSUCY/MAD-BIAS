# -*- coding: utf-8 -*-
"""
模型配置文件
定义实验中使用的各种模型及其参数
"""

# 可用模型列表
AVAILABLE_MODELS = {
    "gpt-4.1-nano": {
        "name": "GPT-4.1-Nano",
        "model": "gpt-4.1-nano",
        "temperature": 0.7,
        "max_tokens": 1000,
    },
    "deepseek-chat": {
        "name": "DeepSeek",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 1000,
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini",
        "model": "gemini-2.5-flash-lite-preview-06-17",
        "temperature": 0.7,
        "max_tokens": 1000,
    },
    "claude-3.5-haiku": {
        "name": "Claude-3.5",
        "model": "claude-3-5-haiku-20241022",
        "temperature": 0.7,
        "max_tokens": 1000,
    },
}

# 默认模型
DEFAULT_MODEL = "gpt-4.1-nano"

# 用于偏见评估的检测模型
DETECTOR_MODEL = {
    "name": "Bias Detector",
    "model": "gpt-4.1-nano",
    "temperature": 0.2,  # 低温度使结果更确定性
    "max_tokens": 500,
}

# 模型组合设置
MODEL_COMBINATIONS = {
    # 同质组合
    "homogeneous": {
        "gpt-4.1-nano": ["gpt-4.1-nano"] * 4,
        "deepseek-chat": ["deepseek-chat"] * 4,
        # "claude-3.5-haiku": ["claude-3-5-haiku-20241022"] * 4,
    },
    # 异质组合
    "heterogeneous": {
        "mix-1": ["gpt-4.1-nano", "deepseek-chat", "claude-3-5-haiku-20241022", "gemini-2.5-flash-lite-preview-06-17"],
        "mix-2": ["gpt-4.1-nano", "claude-3-5-haiku-20241022", "gemini-2.5-flash-lite-preview-06-17", "deepseek-chat"],
        "mix-3": ["gpt-4.1-nano", "deepseek-chat", "gemini-2.5-flash-lite-preview-06-17", "claude-3-5-haiku-20241022"],
    }
} 