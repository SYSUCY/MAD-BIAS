# -*- coding: utf-8 -*-
"""
运行实验脚本
用于启动和执行实验
"""

import os
import sys
import argparse
import asyncio
import logging

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import topics, models, settings
from src.experiment.runner import ExperimentRunner

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

async def main():
    """主函数"""
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
        default=None,
        help="要测试的模型列表"
    )
    
    # 智能体数量
    parser.add_argument(
        "--agent_counts", 
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
    
    # API密钥（可选）
    parser.add_argument(
        "--api_key", 
        type=str,
        default=None,
        help="OpenAI API密钥，如果不提供则使用环境变量"
    )
    
    args = parser.parse_args()
    
    # 如果提供了API密钥，则设置环境变量
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # 初始化实验运行器
    runner = ExperimentRunner()
    
    try:
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
        
        logger.info("实验执行完成")
        
    except Exception as e:
        logger.error(f"实验执行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 