# -*- coding: utf-8 -*-
"""
基础智能体类
定义智能体的基本属性和方法
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import uuid

class BaseAgent(ABC):
    """
    智能体基类，定义了智能体的基本属性和方法
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 name: str = "Agent",
                 model: str = "unknown",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体唯一标识符
            name: 智能体名称
            model: 使用的模型名称
            temperature: 温度参数
            max_tokens: 最大生成令牌数
        """
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = []  # 对话历史
        
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 智能体的回复
        """
        pass
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        添加对话到历史记录
        
        Args:
            role: 角色（"user" 或 "assistant"）
            content: 对话内容
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self.history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Returns:
            List[Dict[str, Any]]: 对话历史列表
        """
        return self.history
    
    def save_history(self, filepath: str) -> None:
        """
        保存对话历史到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def load_history(self, filepath: str) -> None:
        """
        从文件加载对话历史
        
        Args:
            filepath: 文件路径
        """
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
    
    def get_last_response(self) -> Optional[str]:
        """
        获取最后一条助手回复
        
        Returns:
            Optional[str]: 最后一条助手回复，如果没有则为None
        """
        for item in reversed(self.history):
            if item["role"] == "assistant":
                return item["content"]
        return None
    
    def __repr__(self) -> str:
        return f"{self.name}(id={self.agent_id}, model={self.model})" 