# -*- coding: utf-8 -*-
"""
分析结果脚本
用于分析实验结果
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import logging

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.evaluation.metrics import BiasMetrics, DebateMetrics
from src.visualization.data_process import DataProcessor

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "analysis.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def analyze_single_agent_results(single_agent_dir: str = None) -> None:
    """
    分析单智能体实验结果
    
    Args:
        single_agent_dir: 单智能体实验数据目录
    """
    logger.info("开始分析单智能体实验结果")
    
    # 加载数据
    processor = DataProcessor()
    data = processor.load_single_agent_data(single_agent_dir)
    
    if data.empty:
        logger.error("未能加载单智能体实验数据")
        return
    
    # 基础统计分析
    logger.info(f"数据条数: {len(data)}")
    
    # 按模型分析偏见强度
    model_stats = data.groupby("model").agg({
        "bias_strength": ["count", "mean", "std", "min", "max"]
    })
    
    logger.info("各模型偏见强度统计:")
    logger.info("\n" + str(model_stats))
    
    # 按话题分析偏见强度
    topic_stats = data.groupby("topic").agg({
        "bias_strength": ["count", "mean", "std", "min", "max"]
    })
    
    # 按偏见强度均值排序
    topic_stats = topic_stats.sort_values(by=("bias_strength", "mean"), ascending=False)
    
    logger.info("各话题偏见强度统计 (Top 10):")
    logger.info("\n" + str(topic_stats.head(10)))
    
    # 保存详细分析结果
    output_dir = os.path.join(settings.DATA_PROCESSED_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    model_stats.to_csv(os.path.join(output_dir, "single_agent_model_stats.csv"))
    topic_stats.to_csv(os.path.join(output_dir, "single_agent_topic_stats.csv"))
    
    logger.info(f"单智能体实验分析结果已保存到 {output_dir}")

def analyze_multi_agent_results(multi_agent_dir: str = None) -> None:
    """
    分析多智能体实验结果
    
    Args:
        multi_agent_dir: 多智能体实验数据目录
    """
    logger.info("开始分析多智能体实验结果")
    
    # 加载数据
    processor = DataProcessor()
    data = processor.load_multi_agent_data(multi_agent_dir)
    
    if data.empty:
        logger.error("未能加载多智能体实验数据")
        return
    
    # 基础统计分析
    logger.info(f"数据条数: {len(data)}")
    
    # 计算偏见变化
    data["bias_change_ratio"] = np.where(
        data["initial_bias_strength"] > 0,
        (data["final_bias_strength"] - data["initial_bias_strength"]) / data["initial_bias_strength"],
        0
    )
    
    # 按智能体数量分析
    agent_count_stats = data.groupby("agent_count").agg({
        "final_bias_strength": ["count", "mean", "std"],
        "bias_strength_change": ["mean", "std"],
        "bias_change_ratio": ["mean", "std"]
    })
    
    logger.info("各智能体数量的偏见统计:")
    logger.info("\n" + str(agent_count_stats))
    
    # 按辩论轮次分析
    rounds_stats = data.groupby("debate_rounds").agg({
        "final_bias_strength": ["count", "mean", "std"],
        "bias_strength_change": ["mean", "std"],
        "bias_change_ratio": ["mean", "std"]
    })
    
    logger.info("各辩论轮次的偏见统计:")
    logger.info("\n" + str(rounds_stats))
    
    # 检验主假设：计算初始和最终偏见强度的差异显著性
    ttest_result = stats.ttest_rel(
        data["initial_bias_strength"].dropna(),
        data["final_bias_strength"].dropna()
    )
    
    logger.info(f"初始和最终偏见强度的配对t检验: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f}")
    
    if ttest_result.pvalue < 0.05:
        if data["final_bias_strength"].mean() > data["initial_bias_strength"].mean():
            logger.info("结论：多智能体辩论显著增强了偏见强度 (p < 0.05)")
        else:
            logger.info("结论：多智能体辩论显著降低了偏见强度 (p < 0.05)")
    else:
        logger.info("结论：多智能体辩论对偏见强度的影响不显著 (p >= 0.05)")
    
    # 保存详细分析结果
    output_dir = os.path.join(settings.DATA_PROCESSED_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    agent_count_stats.to_csv(os.path.join(output_dir, "multi_agent_count_stats.csv"))
    rounds_stats.to_csv(os.path.join(output_dir, "multi_agent_rounds_stats.csv"))
    
    # 保存假设检验结果
    with open(os.path.join(output_dir, "hypothesis_test_results.txt"), "w") as f:
        f.write(f"初始和最终偏见强度的配对t检验:\n")
        f.write(f"t值: {ttest_result.statistic:.4f}\n")
        f.write(f"p值: {ttest_result.pvalue:.4f}\n")
        f.write(f"初始偏见强度均值: {data['initial_bias_strength'].mean():.4f}\n")
        f.write(f"最终偏见强度均值: {data['final_bias_strength'].mean():.4f}\n")
        f.write(f"偏见强度变化均值: {data['bias_strength_change'].mean():.4f}\n")
        f.write(f"偏见强度变化比例均值: {data['bias_change_ratio'].mean():.4f}\n")
    
    logger.info(f"多智能体实验分析结果已保存到 {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析实验结果")
    
    # 分析类型
    parser.add_argument(
        "--analysis_type", 
        type=str, 
        choices=["single", "multi", "all"],
        default="all",
        help="要分析的实验类型"
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
    
    args = parser.parse_args()
    
    try:
        if args.analysis_type in ["single", "all"]:
            analyze_single_agent_results(args.single_agent_dir)
            
        if args.analysis_type in ["multi", "all"]:
            analyze_multi_agent_results(args.multi_agent_dir)
        
        logger.info("分析完成")
        
    except Exception as e:
        logger.error(f"分析出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 