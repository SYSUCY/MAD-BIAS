# -*- coding: utf-8 -*-
"""
多智能体辩论模块
处理多个智能体之间的辩论
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
import random

from config import topics, settings
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MultiAgentDebate:
    """
    多智能体辩论类
    处理多个智能体之间的辩论
    """
    
    def __init__(self, 
                 agents: List[BaseAgent], 
                 experiment_id: Optional[str] = None,
                 max_rounds: int = 3):
        """
        初始化多智能体辩论
        
        Args:
            agents: 智能体列表
            experiment_id: 实验ID
            max_rounds: 最大辩论轮次
        """
        self.agents = agents
        self.experiment_id = experiment_id or f"debate_{str(uuid.uuid4())[:8]}"
        self.max_rounds = max_rounds
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.responses = {}  # 存储每个智能体在每轮的回复
        
    async def run_debate(self, topic: str) -> Dict[str, Any]:
        """
        运行多智能体辩论
        
        Args:
            topic: 辩论话题
            
        Returns:
            Dict[str, Any]: 辩论结果
        """
        logger.info(f"开始多智能体辩论: {self.experiment_id}, 话题: {topic}, 轮次: {self.max_rounds}")
        
        # 清除所有智能体的历史
        for agent in self.agents:
            agent.clear_history()
        
        # 初始化结果
        debate_result = {
            "experiment_id": self.experiment_id,
            "topic": topic,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "model": agent.model
                } for agent in self.agents
            ],
            "max_rounds": self.max_rounds,
            "rounds": [],
            "timestamp": datetime.now().isoformat(),
            "type": "multi_agent_debate"
        }
        
        try:
            # 初始轮 - 获取所有智能体的初始观点
            initial_round = {
                "round_num": 0,
                "responses": []
            }
            
            # 获取所有智能体的初始观点
            for agent in self.agents:
                prompt = topics.DEBATE_INITIAL_PROMPT.format(topic=topic)
                
                # 记录开始时间
                start_time = time.time()
                
                # 生成回复
                response = await agent.generate_response(prompt)
                
                # 记录结束时间
                end_time = time.time()
                
                # 存储回复
                self.responses[agent.agent_id] = [response]
                
                # 添加到轮次结果
                initial_round["responses"].append({
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "response": response,
                    "prompt": prompt,
                    "duration": end_time - start_time
                })
            
            # 添加初始轮到结果
            debate_result["rounds"].append(initial_round)
            
            # 进行多轮辩论
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"辩论轮次 {round_num}/{self.max_rounds}")
                
                current_round = {
                    "round_num": round_num,
                    "responses": []
                }
                
                # 随机化发言顺序
                random_order = list(range(len(self.agents)))
                random.shuffle(random_order)
                
                # 每个智能体依次发言
                for idx in random_order:
                    agent = self.agents[idx]
                    
                    # 准备其他智能体的上一轮回复
                    previous_responses = ""
                    for other_idx, other_agent in enumerate(self.agents):
                        if other_agent.agent_id != agent.agent_id:
                            prev_response = self.responses[other_agent.agent_id][-1]
                            previous_responses += f"参与者 {other_agent.name} 的观点:\n{prev_response}\n\n"
                    
                    # 准备提示词
                    prompt = topics.DEBATE_ROUND_PROMPT.format(
                        topic=topic,
                        round_num=round_num,
                        previous_responses=previous_responses
                    )
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 生成回复
                    response = await agent.generate_response(prompt)
                    
                    # 记录结束时间
                    end_time = time.time()
                    
                    # 存储回复
                    self.responses[agent.agent_id].append(response)
                    
                    # 添加到轮次结果
                    current_round["responses"].append({
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "response": response,
                        "prompt": prompt,
                        "duration": end_time - start_time
                    })
                
                # 添加当前轮到结果
                debate_result["rounds"].append(current_round)
            
            # 最终总结轮
            final_round = {
                "round_num": self.max_rounds + 1,
                "responses": []
            }
            
            # 每个智能体进行总结
            for agent in self.agents:
                # 准备提示词
                prompt = topics.DEBATE_SUMMARY_PROMPT.format(
                    topic=topic,
                    rounds=self.max_rounds
                )
                
                # 记录开始时间
                start_time = time.time()
                
                # 生成回复
                response = await agent.generate_response(prompt)
                
                # 记录结束时间
                end_time = time.time()
                
                # 添加到轮次结果
                final_round["responses"].append({
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "response": response,
                    "prompt": prompt,
                    "duration": end_time - start_time
                })
            
            # 添加总结轮到结果
            debate_result["rounds"].append(final_round)
            
            logger.info(f"辩论完成: {self.experiment_id}, 话题: {topic}")
            
            # 添加到结果列表并保存
            self.results.append(debate_result)
            self._save_result(debate_result)
            
            return debate_result
            
        except Exception as e:
            error_msg = f"辩论出错: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # 添加错误信息到结果
            debate_result["error"] = str(e)
            debate_result["error_traceback"] = traceback.format_exc()
            debate_result["type"] = "multi_agent_debate_error"
            
            # 添加到结果列表并保存
            self.results.append(debate_result)
            self._save_result(debate_result)
            
            return debate_result
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        保存辩论结果到文件
        
        Args:
            result: 辩论结果
        """
        # 创建保存目录
        save_dir = os.path.join(
            settings.DATA_RAW_DIR, 
            f"debate_{self.timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存结果
        result_file = os.path.join(
            save_dir,
            f"{result['topic']}_{len(self.agents)}agents_{self.max_rounds}rounds.json"
        )
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        获取所有辩论结果
        
        Returns:
            List[Dict[str, Any]]: 辩论结果列表
        """
        return self.results 