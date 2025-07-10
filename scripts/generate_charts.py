# -*- coding: utf-8 -*-
"""
图表生成脚本
用于生成可视化图表
"""

import os
import sys
import argparse
import logging

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

def generate_single_agent_charts(single_agent_dir: str = None, output_dir: str = None) -> None:
    """
    生成单智能体实验图表
    
    Args:
        single_agent_dir: 单智能体实验数据目录
        output_dir: 图表输出目录
    """
    logger.info("开始生成单智能体实验图表")
    
    # 加载数据
    processor = DataProcessor()
    data = processor.load_single_agent_data(single_agent_dir)
    
    if data.empty:
        logger.error("未能加载单智能体实验数据")
        return
    
    # 准备模型偏见数据
    model_data = processor.prepare_bias_model_data(data)
    
    if not model_data:
        logger.error("处理模型偏见数据失败")
        return
    
    # 准备话题偏见数据
    topic_data = processor.prepare_bias_topic_data(data)
    
    if not topic_data:
        logger.error("处理话题偏见数据失败")
        return
    
    # 创建可视化对象
    visualizer = BiasVisualization(save_dir=output_dir)
    
    # 生成图表
    visualizer.plot_bias_by_model(model_data["raw_data"])
    visualizer.plot_bias_by_topic(topic_data["raw_data"])
    
    logger.info(f"单智能体实验图表已生成")

def generate_multi_agent_charts(multi_agent_dir: str = None, output_dir: str = None) -> None:
    """
    生成多智能体实验图表
    
    Args:
        multi_agent_dir: 多智能体实验数据目录
        output_dir: 图表输出目录
    """
    logger.info("开始生成多智能体实验图表")
    
    # 加载数据
    processor = DataProcessor()
    data = processor.load_multi_agent_data(multi_agent_dir)
    
    if data.empty:
        logger.error("未能加载多智能体实验数据")
        return
    
    # 创建可视化对象
    visualizer = BiasVisualization(save_dir=output_dir)
    
    # 生成智能体数量相关图表
    visualizer.plot_bias_vs_agent_count(data)
    
    # 生成辩论轮次相关图表
    visualizer.plot_bias_vs_debate_rounds(data)
    
    # 区分同质化和异质化数据（如果有分组信息）
    if "group_type" in data.columns:
        homo_data, hetero_data = processor.split_homo_hetero_data(data)
        visualizer.plot_homogeneous_vs_heterogeneous(homo_data, hetero_data)
    
    logger.info(f"多智能体实验图表已生成")

def generate_bias_evolution_charts(multi_agent_dir: str = None, output_dir: str = None) -> None:
    """
    生成偏见演变图表
    
    Args:
        multi_agent_dir: 多智能体实验数据目录
        output_dir: 图表输出目录
    """
    logger.info("开始生成偏见演变图表")
    
    # 加载数据
    processor = DataProcessor()
    
    # 这里我们需要完整的JSON数据，而不是摘要CSV
    # 通常我们需要读取原始JSON文件
    json_files = []
    
    if multi_agent_dir:
        json_path = os.path.join(multi_agent_dir, "full_results.json")
        if os.path.exists(json_path):
            json_files.append(json_path)
    else:
        # 查找最新的多智能体实验数据
        import glob
        multi_agent_dirs = glob.glob(os.path.join(settings.DATA_PROCESSED_DIR, "multi_agent_*"))
        if multi_agent_dirs:
            latest_dir = max(multi_agent_dirs, key=os.path.getmtime)
            json_path = os.path.join(latest_dir, "full_results.json")
            if os.path.exists(json_path):
                json_files.append(json_path)
    
    if not json_files:
        logger.error("未找到多智能体实验数据JSON文件")
        return
    
    # 加载和处理JSON数据
    import json
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            logger.error(f"加载JSON数据出错: {str(e)}")
    
    if not all_data:
        logger.error("未能加载多智能体实验JSON数据")
        return
    
    # 处理数据以获取演变信息
    evolution_data = processor.prepare_bias_evolution_data(all_data)
    
    if evolution_data.empty:
        logger.error("处理偏见演变数据失败")
        return
    
    # 创建可视化对象
    visualizer = BiasVisualization(save_dir=output_dir)
    
    # 为不同智能体数量和辩论轮次生成演变图表
    for agent_count in evolution_data["agent_count"].unique():
        for debate_rounds in evolution_data["debate_rounds"].unique():
            visualizer.plot_bias_evolution(
                evolution_data, 
                agent_count=agent_count,
                debate_rounds=debate_rounds
            )
    
    logger.info(f"偏见演变图表已生成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成可视化图表")
    
    # 图表类型
    parser.add_argument(
        "--chart_type", 
        type=str, 
        choices=["single", "multi", "evolution", "all"],
        default="all",
        help="要生成的图表类型"
    )
    
    # 数据目录
    parser.add_argument(
        "--single_agent_dir", 
        type=str,
        default=None,
        help="单智能体实验数据目录"
    )
    
    parser.add_argument(
        "--multi_agent_dir", 
        type=str,
        default=None,
        help="多智能体实验数据目录"
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=os.path.join(settings.DATA_PROCESSED_DIR, "visualization"),
        help="图表输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.chart_type in ["single", "all"]:
            generate_single_agent_charts(args.single_agent_dir, args.output_dir)
            
        if args.chart_type in ["multi", "all"]:
            generate_multi_agent_charts(args.multi_agent_dir, args.output_dir)
            
        if args.chart_type in ["evolution", "all"]:
            generate_bias_evolution_charts(args.multi_agent_dir, args.output_dir)
        
        logger.info(f"图表已生成到目录: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"生成图表出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 