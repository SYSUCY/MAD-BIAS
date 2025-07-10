# -*- coding: utf-8 -*-
"""
实验运行器模块
运行实验的主要流程
"""

import os
import json
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional

from config import topics, models, settings
from src.experiment.controller import ExperimentController

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "experiment.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    实验运行器
    运行实验的主要流程
    """
    
    def __init__(self):
        """初始化实验运行器"""
        self.controller = ExperimentController()
        
    async def run_baseline_experiments(self, models_to_test: List[str] = None, topics_to_test: List[str] = None) -> None:
        """
        运行基准实验（单智能体）
        
        Args:
            models_to_test: 要测试的模型列表
            topics_to_test: 要测试的话题列表
        """
        logger.info("开始运行基准实验（单智能体）")
        
        # 如果未指定模型，使用所有可用模型
        if not models_to_test:
            models_to_test = list(models.AVAILABLE_MODELS.keys())
            
        # 如果未指定话题，使用默认话题
        if not topics_to_test:
            topics_to_test = topics.TOPICS[:settings.EXPERIMENT_SETTINGS["topics_per_experiment"]]
        
        await self.controller.run_single_agent_experiments(
            model_names=models_to_test,
            selected_topics=topics_to_test
        )
        
        logger.info("基准实验（单智能体）完成")
    
    async def run_debate_experiments(self, 
                                    agent_counts: List[int] = None, 
                                    debate_rounds: List[int] = None,
                                    topics_to_test: List[str] = None,
                                    run_homogeneous: bool = True,
                                    run_heterogeneous: bool = True) -> None:
        """
        运行辩论实验（多智能体）
        
        Args:
            agent_counts: 智能体数量列表
            debate_rounds: 辩论轮次列表
            topics_to_test: 要测试的话题列表
            run_homogeneous: 是否运行同质化实验
            run_heterogeneous: 是否运行异质化实验
        """
        logger.info("开始运行辩论实验（多智能体）")
        
        # 如果未指定智能体数量，使用默认设置
        if not agent_counts:
            agent_counts = settings.EXPERIMENT_SETTINGS["agent_counts"]
            # 确保至少有2个智能体
            agent_counts = [count for count in agent_counts if count >= 2]
            
        # 如果未指定辩论轮次，使用默认设置
        if not debate_rounds:
            debate_rounds = settings.EXPERIMENT_SETTINGS["debate_rounds"]
            
        # 如果未指定话题，使用默认话题
        if not topics_to_test:
            topics_to_test = topics.TOPICS[:settings.EXPERIMENT_SETTINGS["topics_per_experiment"]]
            
        # 运行同质化实验
        if run_homogeneous:
            logger.info("运行同质化辩论实验")
            await self.controller.run_multi_agent_experiments(
                agent_counts=agent_counts,
                debate_rounds=debate_rounds,
                selected_topics=topics_to_test,
                homogeneous=True
            )
            
        # 运行异质化实验
        if run_heterogeneous:
            logger.info("运行异质化辩论实验")
            await self.controller.run_multi_agent_experiments(
                agent_counts=agent_counts,
                debate_rounds=debate_rounds,
                selected_topics=topics_to_test,
                homogeneous=False
            )
            
        logger.info("辩论实验（多智能体）完成")
    
    async def run_all_experiments(self, 
                                models_to_test: List[str] = None,
                                agent_counts: List[int] = None,
                                debate_rounds: List[int] = None,
                                topics_to_test: List[str] = None) -> None:
        """
        运行所有实验
        
        Args:
            models_to_test: 要测试的模型列表
            agent_counts: 智能体数量列表
            debate_rounds: 辩论轮次列表
            topics_to_test: 要测试的话题列表
        """
        logger.info("开始运行所有实验")
        
        # 1. 运行基准实验（单智能体）
        await self.run_baseline_experiments(
            models_to_test=models_to_test,
            topics_to_test=topics_to_test
        )
        
        # 2. 运行同质化辩论实验
        await self.run_debate_experiments(
            agent_counts=agent_counts,
            debate_rounds=debate_rounds,
            topics_to_test=topics_to_test,
            run_homogeneous=True,
            run_heterogeneous=False
        )
        
        # 3. 运行异质化辩论实验
        await self.run_debate_experiments(
            agent_counts=agent_counts,
            debate_rounds=debate_rounds,
            topics_to_test=topics_to_test,
            run_homogeneous=False,
            run_heterogeneous=True
        )
        
        logger.info("所有实验完成")

async def main(args):
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    runner = ExperimentRunner()
    
    if args.experiment_type == "baseline":
        # 运行基准实验
        await runner.run_baseline_experiments(
            models_to_test=args.models,
            topics_to_test=args.topics
        )
    elif args.experiment_type == "debate":
        # 运行辩论实验
        await runner.run_debate_experiments(
            agent_counts=args.agent_counts,
            debate_rounds=args.rounds,
            topics_to_test=args.topics,
            run_homogeneous=not args.no_homogeneous,
            run_heterogeneous=not args.no_heterogeneous
        )
    elif args.experiment_type == "all":
        # 运行所有实验
        await runner.run_all_experiments(
            models_to_test=args.models,
            agent_counts=args.agent_counts,
            debate_rounds=args.rounds,
            topics_to_test=args.topics
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行多智能体辩论中的偏见实验")
    
    # 实验类型
    parser.add_argument(
        "--experiment_type", 
        type=str, 
        choices=["baseline", "debate", "all"],
        default="all",
        help="要运行的实验类型"
    )
    
    # 模型选择
    parser.add_argument(
        "--models", 
        type=str,
        nargs="+",
        help="要测试的模型列表"
    )
    
    # 智能体数量
    parser.add_argument(
        "--agent_counts", 
        type=int,
        nargs="+",
        help="辩论中使用的智能体数量列表"
    )
    
    # 辩论轮次
    parser.add_argument(
        "--rounds", 
        type=int,
        nargs="+",
        help="辩论轮次列表"
    )
    
    # 话题选择
    parser.add_argument(
        "--topics", 
        type=str,
        nargs="+",
        help="要测试的话题列表"
    )
    
    # 同质化和异质化选项
    parser.add_argument(
        "--no_homogeneous", 
        action="store_true",
        help="跳过同质化实验"
    )
    
    parser.add_argument(
        "--no_heterogeneous", 
        action="store_true",
        help="跳过异质化实验"
    )
    
    args = parser.parse_args()
    
    # 运行主函数
    asyncio.run(main(args)) 