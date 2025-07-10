# -*- coding: utf-8 -*-
"""
单智能体回答模块
处理单个智能体对问题的回答
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import traceback
from datetime import datetime
import uuid

from config import topics, settings
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SingleAgentExperiment:
    """
    单智能体实验类
    处理单个智能体回答争议性问题的实验
    """
    
    def __init__(self, agent: BaseAgent, experiment_id: Optional[str] = None):
        """
        初始化单智能体实验
        
        Args:
            agent: 智能体实例
            experiment_id: 实验ID
        """
        self.agent = agent
        self.experiment_id = experiment_id or f"single_{agent.agent_id}_{str(uuid.uuid4())[:6]}"
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def run_experiment(self, topic: str, prompt_template: str = None) -> Dict[str, Any]:
        """
        运行单智能体实验
        
        Args:
            topic: 话题
            prompt_template: 提示词模板，如果为None则使用默认模板
            
        Returns:
            Dict[str, Any]: 实验结果
        """
        # 清除之前的历史记录
        self.agent.clear_history()
        
        # 准备提示词
        if prompt_template is None:
            prompt_template = topics.SINGLE_AGENT_PROMPT
        
        prompt = prompt_template.format(topic=topic)
        
        logger.info(f"运行单智能体实验: {self.experiment_id}, 话题: {topic}, 智能体: {self.agent.name}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 生成回复
            response = await self.agent.generate_response(prompt)
            
            # 记录结束时间
            end_time = time.time()
            
            # 记录结果
            result = {
                "experiment_id": self.experiment_id,
                "agent_id": self.agent.agent_id,
                "agent_name": self.agent.name,
                "model": self.agent.model,
                "topic": topic,
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "duration": end_time - start_time,
                "type": "single_agent"
            }
            
            self.results.append(result)
            
            # 保存结果
            self._save_result(result)
            
            logger.info(f"实验完成: {self.experiment_id}, 话题: {topic}")
            
            return result
            
        except Exception as e:
            error_msg = f"实验出错: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # 记录错误结果
            error_result = {
                "experiment_id": self.experiment_id,
                "agent_id": self.agent.agent_id,
                "agent_name": self.agent.name,
                "model": self.agent.model,
                "topic": topic,
                "prompt": prompt,
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "type": "single_agent_error"
            }
            
            self.results.append(error_result)
            
            # 保存错误结果
            self._save_result(error_result)
            
            return error_result
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        保存实验结果到文件
        
        Args:
            result: 实验结果
        """
        # 创建保存目录
        save_dir = os.path.join(
            settings.DATA_RAW_DIR, 
            f"single_{self.timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存结果
        result_file = os.path.join(
            save_dir,
            f"{result.get('topic', 'unknown')}_{self.agent.agent_id}.json"
        )
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    def get_results(self) -> List[Dict[str, Any]]:
        """
        获取所有实验结果
        
        Returns:
            List[Dict[str, Any]]: 实验结果列表
        """
        return self.results 