# -*- coding: utf-8 -*-
"""
生成可视化图表脚本
用于处理实验数据并生成可视化图表
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.visualization.data_process import DataProcessor
from src.visualization.charts import BiasVisualization

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "visualization.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def generate_single_agent_visualizations(data_dir=None, output_dir=None):
    """
    生成单智能体实验的可视化图表
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
    """
    logger.info("开始生成单智能体实验可视化图表")
    
    # 初始化数据处理器和可视化类
    processor = DataProcessor(data_dir)
    visualizer = BiasVisualization(output_dir)
    
    # 加载单智能体实验数据
    data = processor.load_single_agent_data()
    if data.empty:
        logger.error("未找到有效的单智能体实验数据")
        return
    
    # 准备模型偏见数据
    model_data = processor.prepare_bias_model_data(data)
    if model_data:
        # 绘制模型偏见对比图
        visualizer.plot_bias_by_model(model_data["raw_data"])
        logger.info("已生成模型偏见对比图")
    
    # 准备话题偏见数据
    topic_data = processor.prepare_bias_topic_data(data)
    if topic_data:
        # 绘制话题偏见对比图
        visualizer.plot_bias_by_topic(topic_data["raw_data"])
        logger.info("已生成话题偏见对比图")
    
    logger.info("单智能体实验可视化图表生成完成")

def generate_multi_agent_visualizations(data_dir=None, output_dir=None):
    """
    生成多智能体实验的可视化图表
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
    """
    logger.info("开始生成多智能体实验可视化图表")
    
    # 初始化数据处理器和可视化类
    processor = DataProcessor(data_dir)
    visualizer = BiasVisualization(output_dir)
    
    # 加载多智能体实验数据
    data = processor.load_multi_agent_data()
    if data.empty:
        logger.error("未找到有效的多智能体实验数据")
        return
    
    # 绘制智能体数量与偏见强度关系图
    if "agent_count" in data.columns and "final_bias_strength" in data.columns:
        visualizer.plot_bias_vs_agent_count(data)
        logger.info("已生成智能体数量与偏见强度关系图")
    
    # 绘制辩论轮次与偏见强度关系图
    if "debate_rounds" in data.columns and "final_bias_strength" in data.columns:
        visualizer.plot_bias_vs_debate_rounds(data)
        logger.info("已生成辩论轮次与偏见强度关系图")
    
    # 分离同质化和异质化数据
    if "homogeneous" in data.columns:
        homo_data, hetero_data = processor.split_homo_hetero_data(data)
        if not homo_data.empty and not hetero_data.empty:
            visualizer.plot_homogeneous_vs_heterogeneous(homo_data, hetero_data)
            logger.info("已生成同质化与异质化对比图")
    
    # 绘制偏见演变图
    try:
        evolution_data = processor.prepare_bias_evolution_data(data)
        if not evolution_data.empty:
            visualizer.plot_bias_evolution(evolution_data)
            logger.info("已生成偏见演变图")
    except Exception as e:
        logger.error(f"生成偏见演变图出错: {str(e)}")
    
    logger.info("多智能体实验可视化图表生成完成")

def generate_all_visualizations(data_dir=None, output_dir=None):
    """
    生成所有实验的可视化图表
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
    """
    generate_single_agent_visualizations(data_dir, output_dir)
    generate_multi_agent_visualizations(data_dir, output_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成实验结果的可视化图表")
    
    # 可视化类型
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["single", "multi", "all"],
        default="all",
        help="要生成的可视化类型"
    )
    
    # 数据目录
    parser.add_argument(
        "--data_dir", 
        type=str,
        default=None,
        help="数据目录"
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="输出目录"
    )
    
    # 单智能体数据目录
    parser.add_argument(
        "--single_dir", 
        type=str,
        default=None,
        help="单智能体实验数据目录"
    )
    
    # 多智能体数据目录
    parser.add_argument(
        "--multi_dir", 
        type=str,
        default=None,
        help="多智能体实验数据目录"
    )
    
    args = parser.parse_args()
    
    try:
        if args.type == "single":
            generate_single_agent_visualizations(args.single_dir or args.data_dir, args.output_dir)
        elif args.type == "multi":
            generate_multi_agent_visualizations(args.multi_dir or args.data_dir, args.output_dir)
        else:
            generate_all_visualizations(args.data_dir, args.output_dir)
        
        logger.info("可视化图表生成完成")
        
    except Exception as e:
        logger.error(f"生成可视化图表出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 