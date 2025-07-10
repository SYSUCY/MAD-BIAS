# -*- coding: utf-8 -*-
"""
实验控制器模块
控制实验的执行流程和参数设置
"""

import os
import json
import time
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import uuid
import pandas as pd

from config import topics, models, settings
from src.agents.model_agent import OpenAIAgent
from src.debate.single_agent import SingleAgentExperiment
from src.debate.multi_agent import MultiAgentDebate
from src.evaluation.bias_detector import BiasDetector

logger = logging.getLogger(__name__)

class ExperimentController:
    """
    实验控制器
    控制实验的执行流程和参数设置
    """
    
    def __init__(self, experiment_id: Optional[str] = None):
        """
        初始化实验控制器
        
        Args:
            experiment_id: 实验ID
        """
        self.experiment_id = experiment_id or f"exp_{str(uuid.uuid4())[:8]}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.config = settings.EXPERIMENT_SETTINGS
        self.bias_detector = BiasDetector()
        
    async def run_single_agent_experiments(self, 
                                          model_names: List[str], 
                                          selected_topics: List[str],
                                          topic_count: int = 5) -> List[Dict[str, Any]]:
        """
        运行单智能体实验
        
        Args:
            model_names: 要测试的模型名称列表
            selected_topics: 可选话题列表
            topic_count: 测试的话题数量
            
        Returns:
            List[Dict[str, Any]]: 实验结果列表
        """
        results = []
        
        # 如果没有提供话题，随机选择
        if not selected_topics:
            selected_topics = random.sample(topics.TOPICS, min(topic_count, len(topics.TOPICS)))
            
        # 限制话题数量
        if len(selected_topics) > topic_count:
            selected_topics = selected_topics[:topic_count]
            
        logger.info(f"开始单智能体实验: {self.experiment_id}")
        logger.info(f"测试模型: {model_names}")
        logger.info(f"测试话题: {selected_topics}")
        
        # 创建智能体
        agents = [OpenAIAgent(model_name=model) for model in model_names]
        
        # 对每个智能体运行实验
        for agent in agents:
            experiment = SingleAgentExperiment(agent)
            
            # 对每个话题测试
            for topic in selected_topics:
                # 对每个话题进行多次重复实验
                for rep in range(self.config["num_repeats"]):
                    try:
                        # 运行实验
                        result = await experiment.run_experiment(topic)
                        
                        # 评估偏见
                        bias_evaluation = await self.bias_detector.evaluate_bias(
                            result["response"], 
                            result["topic"]
                        )
                        
                        # 添加偏见评估结果
                        result["bias_evaluation"] = bias_evaluation
                        
                        # 添加到结果列表
                        results.append(result)
                        
                        # 添加适当的延迟以避免API限制
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"实验错误 - 模型: {agent.model}, 话题: {topic}, 错误: {str(e)}")
        
        # 保存汇总结果
        self._save_summary_results(results, "single_agent")
        
        # 添加到控制器结果
        self.results.extend(results)
        
        return results
    
    async def run_multi_agent_experiments(self,
                                         agent_counts: List[int],
                                         debate_rounds: List[int],
                                         selected_topics: List[str],
                                         homogeneous: bool = True,
                                         topic_count: int = 5) -> List[Dict[str, Any]]:
        """
        运行多智能体辩论实验
        
        Args:
            agent_counts: 智能体数量列表
            debate_rounds: 辩论轮次列表
            selected_topics: 可选话题列表
            homogeneous: 是否使用同质化智能体
            topic_count: 测试的话题数量
            
        Returns:
            List[Dict[str, Any]]: 实验结果列表
        """
        results = []
        
        # 如果没有提供话题，随机选择
        if not selected_topics:
            selected_topics = random.sample(topics.TOPICS, min(topic_count, len(topics.TOPICS)))
            
        # 限制话题数量
        if len(selected_topics) > topic_count:
            selected_topics = selected_topics[:topic_count]
            
        logger.info(f"开始多智能体辩论实验: {self.experiment_id}")
        logger.info(f"智能体数量: {agent_counts}")
        logger.info(f"辩论轮次: {debate_rounds}")
        logger.info(f"同质化: {homogeneous}")
        logger.info(f"测试话题: {selected_topics}")
        
        # 选择模型组合
        if homogeneous:
            model_combinations = models.MODEL_COMBINATIONS["homogeneous"]
            # 使用默认模型进行同质化组合
            default_model = "gpt-3.5-turbo"
            model_list = model_combinations.get(default_model, [default_model] * 4)
        else:
            model_combinations = models.MODEL_COMBINATIONS["heterogeneous"]
            # 使用默认混合组合
            model_list = model_combinations.get("mix-1", ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "deepseek-chat"])
        
        # 对每个智能体数量进行实验
        for agent_count in agent_counts:
            # 限制智能体数量
            models_for_debate = model_list[:agent_count]
            
            # 创建智能体
            agents = [OpenAIAgent(model_name=model) for model in models_for_debate]
            
            # 对每个辩论轮次进行实验
            for rounds in debate_rounds:
                # 创建辩论实验
                debate = MultiAgentDebate(agents, max_rounds=rounds)
                
                # 对每个话题测试
                for topic in selected_topics:
                    # 对每个话题进行多次重复实验
                    for rep in range(self.config["num_repeats"]):
                        try:
                            # 运行辩论
                            result = await debate.run_debate(topic)
                            
                            # 评估每个智能体在每轮的偏见
                            evaluations = {}
                            
                            # 对每个智能体评估
                            for agent_data in result["agents"]:
                                agent_id = agent_data["agent_id"]
                                evaluations[agent_id] = []
                                
                                # 评估每轮的回复
                                for round_data in result["rounds"]:
                                    for response_data in round_data["responses"]:
                                        if response_data["agent_id"] == agent_id:
                                            # 评估偏见
                                            bias_evaluation = await self.bias_detector.evaluate_bias(
                                                response_data["response"], 
                                                result["topic"]
                                            )
                                            
                                            # 添加轮次信息
                                            bias_evaluation["round_num"] = round_data["round_num"]
                                            
                                            # 添加到评估结果
                                            evaluations[agent_id].append(bias_evaluation)
                                            
                                            # 添加适当的延迟以避免API限制
                                            await asyncio.sleep(0.5)
                            
                            # 添加偏见评估结果
                            result["evaluations"] = evaluations
                            
                            # 添加到结果列表
                            results.append(result)
                            
                        except Exception as e:
                            logger.error(f"辩论实验错误 - 智能体数量: {agent_count}, 轮次: {rounds}, 话题: {topic}, 错误: {str(e)}")
        
        # 保存汇总结果
        self._save_summary_results(results, "multi_agent")
        
        # 添加到控制器结果
        self.results.extend(results)
        
        return results
        
    def _save_summary_results(self, results: List[Dict[str, Any]], experiment_type: str) -> None:
        """
        保存汇总结果
        
        Args:
            results: 实验结果列表
            experiment_type: 实验类型
        """
        # 创建保存目录
        save_dir = os.path.join(settings.DATA_PROCESSED_DIR, f"{experiment_type}_{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存完整结果
        full_results_path = os.path.join(save_dir, f"full_results.json")
        with open(full_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 提取简要结果进行CSV保存
        if experiment_type == "single_agent":
            # 处理单智能体实验
            summary_data = []
            
            for result in results:
                if "bias_evaluation" not in result:
                    continue
                    
                summary_data.append({
                    "experiment_id": result.get("experiment_id"),
                    "agent_id": result.get("agent_id"),
                    "model": result.get("model"),
                    "topic": result.get("topic"),
                    "bias_strength": result["bias_evaluation"].get("bias_strength"),
                    "bias_type": result["bias_evaluation"].get("bias_type"),
                    "polarization": result["bias_evaluation"].get("polarization"),
                    "evidence_usage": result["bias_evaluation"].get("evidence_usage"),
                    "emotional_language": result["bias_evaluation"].get("emotional_language"),
                    "timestamp": result.get("timestamp")
                })
                
        else:
            # 处理多智能体实验
            summary_data = []
            
            for result in results:
                if "evaluations" not in result:
                    continue
                
                # 提取每个智能体的评估结果
                for agent_data in result["agents"]:
                    agent_id = agent_data["agent_id"]
                    
                    # 获取该智能体的评估结果
                    agent_evals = result["evaluations"].get(agent_id, [])
                    
                    # 提取初始和最终评估
                    if not agent_evals:
                        continue
                        
                    # 初始评估
                    initial_eval = agent_evals[0]
                    
                    # 最终评估
                    final_eval = agent_evals[-1]
                    
                    summary_data.append({
                        "experiment_id": result.get("experiment_id"),
                        "agent_id": agent_id,
                        "model": agent_data.get("model"),
                        "topic": result.get("topic"),
                        "agent_count": len(result["agents"]),
                        "debate_rounds": result.get("max_rounds"),
                        "initial_bias_strength": initial_eval.get("bias_strength"),
                        "final_bias_strength": final_eval.get("bias_strength"),
                        "initial_polarization": initial_eval.get("polarization"),
                        "final_polarization": final_eval.get("polarization"),
                        "bias_strength_change": final_eval.get("bias_strength") - initial_eval.get("bias_strength") if final_eval.get("bias_strength") and initial_eval.get("bias_strength") else None,
                        "polarization_change": final_eval.get("polarization") - initial_eval.get("polarization") if final_eval.get("polarization") and initial_eval.get("polarization") else None,
                        "timestamp": result.get("timestamp")
                    })
        
        # 保存为CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(save_dir, f"summary_results.csv")
            df.to_csv(csv_path, index=False) 