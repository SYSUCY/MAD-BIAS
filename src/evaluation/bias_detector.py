# -*- coding: utf-8 -*-
"""
Bias Detection Module
Uses detection models to evaluate bias strength
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
import logging
import traceback
import re

from openai import OpenAI
from openai import AsyncOpenAI

from config import topics, models, settings
from src.utils.api_manager import APIManager

logger = logging.getLogger(__name__)

class BiasDetector:
    """
    Bias Detector
    Uses detection models to evaluate bias strength
    """
    
    def __init__(self, 
                 model: str = models.DETECTOR_MODEL["model"],
                 temperature: float = models.DETECTOR_MODEL["temperature"],
                 max_tokens: int = models.DETECTOR_MODEL["max_tokens"],
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize bias detector
        
        Args:
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            api_key: API key
            base_url: API base URL
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化API管理器
        self.api_manager = APIManager()
        
        # Initialize API client
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
    
    async def evaluate_bias(self, response: str, topic: str) -> Dict[str, Any]:
        """
        Evaluate bias in response
        
        Args:
            response: Agent's response
            topic: Discussion topic
            
        Returns:
            Dict[str, Any]: Bias evaluation results
        """
        # 准备提示词
        prompt = topics.BIAS_DETECTION_PROMPT.format(
            topic=topic,
            response=response
        )
        
        # 进行API调用，最多尝试3次
        max_retries = settings.EXPERIMENT_SETTINGS.get("max_retries", 3)
        retry_delay = settings.EXPERIMENT_SETTINGS.get("retry_delay", 5)
        
        for attempt in range(max_retries):
            try:
                # Record start time
                start_time = time.time()
                
                # API call with timeout protection
                result = await asyncio.wait_for(
                    self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a professional bias analyzer, skilled at analyzing bias in text."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    ),
                    timeout=self.request_timeout + 5  # 额外添加5秒缓冲
                )
                
                # Record end time
                end_time = time.time()
                
                # Extract response content
                evaluation = result.choices[0].message.content
                
                # Parse evaluation results
                parsed_result = self._parse_evaluation(evaluation)
                
                # Add metadata
                parsed_result.update({
                    "topic": topic,
                    "raw_evaluation": evaluation,
                    "duration": end_time - start_time,
                    "timestamp": time.time()
                })
                
                return parsed_result
                
            except asyncio.TimeoutError:
                error_msg = f"偏见评估超时 (尝试 {attempt+1}/{max_retries})"
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
                return {
                    "error": "Request timeout",
                    "error_traceback": error_msg,
                    "topic": topic,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                error_msg = str(e)
                full_error = f"Bias evaluation error: {error_msg}\n{traceback.format_exc()}"
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
                
                # 如果所有尝试都失败，返回错误信息
                return {
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "topic": topic,
                    "timestamp": time.time()
                }
    
    def _parse_evaluation(self, evaluation: str) -> Dict[str, Any]:
        """
        Parse evaluation results
        
        Args:
            evaluation: Evaluation text from detection model
            
        Returns:
            Dict[str, Any]: Parsed evaluation results
        """
        result = {
            "bias_strength": None,
            "bias_type": None,
            "polarization": None,
            "evidence_usage": None,
            "emotional_language": None
        }
        
        try:
            # Extract bias strength
            bias_strength_match = re.search(r"[Bb]ias strength.*?(\d+(?:\.\d+)?)", evaluation)
            if bias_strength_match:
                result["bias_strength"] = float(bias_strength_match.group(1))
            
            # Extract bias type
            bias_type_match = re.search(r"[Bb]ias type.*?[:：](.+?)(?:\d|\n|$)", evaluation)
            if bias_type_match:
                result["bias_type"] = bias_type_match.group(1).strip()
            
            # Extract polarization degree
            polarization_match = re.search(r"[Oo]pinion polarization.*?(\d+(?:\.\d+)?)", evaluation)
            if polarization_match:
                result["polarization"] = float(polarization_match.group(1))
            
            # Extract evidence usage
            evidence_match = re.search(r"[Ee]vidence usage.*?(\d+(?:\.\d+)?)", evaluation)
            if evidence_match:
                result["evidence_usage"] = float(evidence_match.group(1))
            
            # Extract emotional language
            emotional_match = re.search(r"[Ee]motional language.*?(\d+(?:\.\d+)?)", evaluation)
            if emotional_match:
                result["emotional_language"] = float(emotional_match.group(1))
                
        except Exception as e:
            logger.error(f"Error parsing evaluation results: {str(e)}")
            
        return result
        
    def evaluate_bias_sync(self, response: str, topic: str) -> Dict[str, Any]:
        """
        Synchronously evaluate bias in response
        
        Args:
            response: Agent's response
            topic: Discussion topic
            
        Returns:
            Dict[str, Any]: Bias evaluation results
        """
        # 准备提示词
        prompt = topics.BIAS_DETECTION_PROMPT.format(
            topic=topic,
            response=response
        )
        
        # 进行API调用，最多尝试3次
        max_retries = settings.EXPERIMENT_SETTINGS.get("max_retries", 3)
        retry_delay = settings.EXPERIMENT_SETTINGS.get("retry_delay", 5)
        
        for attempt in range(max_retries):
            try:
                # Record start time
                start_time = time.time()
                
                # API call
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional bias analyzer, skilled at analyzing bias in text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Record end time
                end_time = time.time()
                
                # Extract response content
                evaluation = result.choices[0].message.content
                
                # Parse evaluation results
                parsed_result = self._parse_evaluation(evaluation)
                
                # Add metadata
                parsed_result.update({
                    "topic": topic,
                    "raw_evaluation": evaluation,
                    "duration": end_time - start_time,
                    "timestamp": time.time()
                })
                
                return parsed_result
                
            except Exception as e:
                error_msg = str(e)
                full_error = f"Bias evaluation error: {error_msg}\n{traceback.format_exc()}"
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
                
                # 如果所有尝试都失败，返回错误信息
                return {
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "topic": topic,
                    "timestamp": time.time()
                } 