# -*- coding: utf-8 -*-
"""
项目启动脚本
用于便捷运行实验
"""

import os
import sys
import argparse
import asyncio
import logging

from config import settings
from src.experiment.runner import ExperimentRunner

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "mad_bias.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MAD-bias: 多智能体辩论中的偏见研究")
    
    # 实验类型
    parser.add_argument(
        "--experiment", 
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
        default=None,
        help="要测试的模型列表"
    )
    
    # 智能体数量
    parser.add_argument(
        "--agents", 
        type=int,
        nargs="+",
        default=None,
        help="辩论中使用的智能体数量列表"
    )
    
    # 辩论轮次
    parser.add_argument(
        "--rounds", 
        type=int,
        nargs="+",
        default=None,
        help="辩论轮次列表"
    )
    
    # 话题选择
    parser.add_argument(
        "--topics", 
        type=str,
        nargs="+",
        default=None,
        help="要测试的话题列表"
    )
    
    # 话题数量
    parser.add_argument(
        "--topic_count", 
        type=int,
        default=settings.EXPERIMENT_SETTINGS["topics_per_experiment"],
        help="要测试的话题数量"
    )
    
    args = parser.parse_args()
    
    # 初始化实验运行器
    runner = ExperimentRunner()
    
    try:
        if args.experiment == "baseline":
            # 运行基准实验
            await runner.run_baseline_experiments(
                models_to_test=args.models,
                topics_to_test=args.topics
            )
            
        elif args.experiment == "debate":
            # 运行辩论实验
            await runner.run_debate_experiments(
                agent_counts=args.agents,
                debate_rounds=args.rounds,
                topics_to_test=args.topics
            )
            
        elif args.experiment == "all":
            # 运行所有实验
            await runner.run_all_experiments(
                models_to_test=args.models,
                agent_counts=args.agents,
                debate_rounds=args.rounds,
                topics_to_test=args.topics
            )
        
        logger.info("实验执行完成")
        
    except Exception as e:
        logger.error(f"实验执行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 