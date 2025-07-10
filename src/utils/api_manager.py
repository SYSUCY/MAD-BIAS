# -*- coding: utf-8 -*-
"""
API管理器模块
处理API端点的自动切换和错误恢复
"""

import os
import logging
import time
import random
from typing import Dict, List, Optional, Any

from config import settings

logger = logging.getLogger(__name__)

class APIManager:
    """API管理器，处理API端点的自动切换和错误恢复"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(APIManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化API管理器"""
        if self._initialized:
            return
            
        self.api_endpoints = settings.API_ENDPOINTS
        self.current_endpoint_key = os.getenv("API_ENDPOINT_KEY", "chatanywhere_org")
        self.current_endpoint = settings.API_BASE_URL
        self.blocked_endpoints = set()
        self.auto_switch = settings.EXPERIMENT_SETTINGS.get("auto_switch_api", True)
        self.block_keywords = settings.EXPERIMENT_SETTINGS.get(
            "api_block_keywords", 
            ["Restricted Access", "访问受限", "blocked"]
        )
        self.last_switch_time = 0
        self.switch_cooldown = 60  # 切换冷却时间（秒）
        self._initialized = True
        
        logger.info(f"API管理器初始化完成，当前端点: {self.current_endpoint_key} ({self.current_endpoint})")
    
    def get_current_endpoint(self) -> str:
        """获取当前API端点"""
        return self.current_endpoint
    
    def is_blocked_error(self, error_msg: str) -> bool:
        """检查错误是否为API封禁错误"""
        if not error_msg:
            return False
            
        return any(keyword in error_msg for keyword in self.block_keywords)
    
    def switch_endpoint(self, error_msg: Optional[str] = None) -> str:
        """
        切换到下一个可用的API端点
        
        Args:
            error_msg: 错误信息，用于判断是否为封禁错误
            
        Returns:
            str: 新的API端点URL
        """
        # 如果不是自动切换模式，则不切换
        if not self.auto_switch:
            logger.warning("API自动切换已禁用，继续使用当前端点")
            return self.current_endpoint
            
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_switch_time < self.switch_cooldown:
            logger.warning(f"API切换冷却中，还需等待 {self.switch_cooldown - (current_time - self.last_switch_time):.1f} 秒")
            return self.current_endpoint
            
        # 如果是封禁错误，将当前端点添加到封禁列表
        if error_msg and self.is_blocked_error(error_msg):
            logger.warning(f"检测到API封禁错误: {error_msg}")
            self.blocked_endpoints.add(self.current_endpoint_key)
            
        # 获取可用端点列表
        available_endpoints = [k for k in self.api_endpoints.keys() if k not in self.blocked_endpoints]
        
        # 如果没有可用端点，则重置封禁列表并重试
        if not available_endpoints:
            logger.warning("所有API端点都被封禁，重置封禁列表")
            self.blocked_endpoints.clear()
            available_endpoints = list(self.api_endpoints.keys())
            
        # 从可用端点中随机选择一个
        new_endpoint_key = random.choice(available_endpoints)
        self.current_endpoint_key = new_endpoint_key
        self.current_endpoint = self.api_endpoints[new_endpoint_key]
        self.last_switch_time = current_time
        
        logger.info(f"切换到新的API端点: {self.current_endpoint_key} ({self.current_endpoint})")
        return self.current_endpoint
        
    def reset(self) -> None:
        """重置API管理器状态"""
        self.blocked_endpoints.clear()
        self.current_endpoint_key = os.getenv("API_ENDPOINT_KEY", "chatanywhere_org")
        self.current_endpoint = self.api_endpoints[self.current_endpoint_key]
        logger.info(f"API管理器已重置，当前端点: {self.current_endpoint_key} ({self.current_endpoint})") 