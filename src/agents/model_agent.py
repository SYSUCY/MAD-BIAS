# -*- coding: utf-8 -*-
"""
模型智能体实现
使用OpenAI API实现智能体
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
import logging
import traceback

from openai import OpenAI
from openai import AsyncOpenAI
from openai import APITimeoutError

from config import models, settings
from .base_agent import BaseAgent
from src.utils.api_manager import APIManager

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "agent.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class OpenAIAgent(BaseAgent):
    """使用OpenAI API的智能体实现"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1-nano",
                 agent_id: Optional[str] = None,
                 name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        初始化OpenAI智能体
        
        Args:
            model_name: 模型名称
            agent_id: 智能体ID
            name: 智能体名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            api_key: API密钥
            base_url: API基础URL
        """
        # 从模型配置获取参数
        model_config = models.AVAILABLE_MODELS.get(
            model_name, 
            models.AVAILABLE_MODELS[models.DEFAULT_MODEL]
        )
        
        # 初始化基类
        super().__init__(
            agent_id=agent_id,
            name=name or model_config["name"],
            model=model_config["model"],
            temperature=temperature or model_config["temperature"],
            max_tokens=max_tokens or model_config["max_tokens"]
        )
        
        # 初始化API管理器
        self.api_manager = APIManager()
        
        # 初始化API客户端
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or self.api_manager.get_current_endpoint()
        
        # 获取超时设置
        self.request_timeout = settings.EXPERIMENT_SETTINGS.get("request_timeout", 60)
        
        self._init_clients()
        
    def _init_clients(self):
        """初始化API客户端"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout
        )
        
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout
        )
        
    async def generate_response(self, prompt: str) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 智能体的回复
        """
        # 添加用户输入到历史
        self.add_to_history("user", prompt)
        
        # 准备消息
        messages = []
        for item in self.history:
            if item["role"] in ["user", "assistant", "system"]:
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
        
        # 进行API调用，最多尝试3次
        max_retries = settings.EXPERIMENT_SETTINGS.get("max_retries", 3)
        retry_delay = settings.EXPERIMENT_SETTINGS.get("retry_delay", 5)
        
        for attempt in range(max_retries):
            try:
                # 使用asyncio.wait_for添加额外的超时保护
                response = await asyncio.wait_for(
                    self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    ),
                    timeout=self.request_timeout + 5  # 额外添加5秒缓冲
                )
                
                # 提取回复内容
                reply_content = response.choices[0].message.content
                
                # 添加回复到历史
                self.add_to_history("assistant", reply_content)
                
                return reply_content
                
            except asyncio.TimeoutError:
                error_msg = f"请求超时 (尝试 {attempt+1}/{max_retries})"
                logger.error(error_msg)
                
                # 切换API端点
                new_endpoint = self.api_manager.switch_endpoint("Timeout Error")
                self.base_url = new_endpoint
                self._init_clients()
                logger.info(f"已切换API端点至: {new_endpoint}，重试中...")
                
                # 如果不是最后一次尝试，则等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                
                # 如果所有尝试都超时，返回错误信息
                self.add_to_history("error", error_msg)
                return f"抱歉，生成回复时超时: {error_msg}"
                
            except Exception as e:
                error_msg = str(e)
                full_error = f"API调用错误: {error_msg}\n{traceback.format_exc()}"
                logger.error(full_error)
                
                # 检查是否为API封禁错误
                if self.api_manager.is_blocked_error(error_msg):
                    # 切换API端点
                    new_endpoint = self.api_manager.switch_endpoint(error_msg)
                    self.base_url = new_endpoint
                    self._init_clients()
                    logger.info(f"已切换API端点至: {new_endpoint}，重试中...")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                
                # 记录错误到历史
                self.add_to_history("error", full_error)
                
                # 返回错误信息
                return f"抱歉，生成回复时发生错误: {error_msg}"
            
    def generate_response_sync(self, prompt: str) -> str:
        """
        同步生成回复（非异步版本）
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 智能体的回复
        """
        # 添加用户输入到历史
        self.add_to_history("user", prompt)
        
        # 准备消息
        messages = []
        for item in self.history:
            if item["role"] in ["user", "assistant", "system"]:
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
        
        # 进行API调用，最多尝试3次
        max_retries = settings.EXPERIMENT_SETTINGS.get("max_retries", 3)
        retry_delay = settings.EXPERIMENT_SETTINGS.get("retry_delay", 5)
        
        for attempt in range(max_retries):
            try:
                # 设置超时
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # 提取回复内容
                reply_content = response.choices[0].message.content
                
                # 添加回复到历史
                self.add_to_history("assistant", reply_content)
                
                return reply_content
                
            except Exception as e:
                error_msg = str(e)
                full_error = f"API调用错误: {error_msg}\n{traceback.format_exc()}"
                logger.error(full_error)
                
                # 检查是否为API封禁错误或超时
                if self.api_manager.is_blocked_error(error_msg) or "timeout" in error_msg.lower():
                    # 切换API端点
                    new_endpoint = self.api_manager.switch_endpoint(error_msg)
                    self.base_url = new_endpoint
                    self._init_clients()
                    logger.info(f"已切换API端点至: {new_endpoint}，重试中...")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                
                # 记录错误到历史
                self.add_to_history("error", full_error)
                
                # 返回错误信息
                return f"抱歉，生成回复时发生错误: {error_msg}" 