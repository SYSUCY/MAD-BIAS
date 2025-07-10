# -*- coding: utf-8 -*-
"""
MAD-bias 论文实验运行脚本
专门针对论文中提到的三个研究问题设计实验
"""

import os
import sys
import time
import logging
import subprocess
import datetime
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings, topics

# 创建日志目录
os.makedirs(os.path.join(settings.LOG_DIR, "paper_experiments"), exist_ok=True)

# 配置日志
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(settings.LOG_DIR, "paper_experiments", f"run_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """
    运行命令并记录日志
    
    Args:
        cmd: 要运行的命令
        description: 命令描述
    
    Returns:
        bool: 是否成功
    """
    logger.info(f"开始: {description}")
    logger.info(f"运行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        logger.info(f"完成: {description} (耗时: {duration:.2f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"失败: {description} (耗时: {duration:.2f}秒)")
        logger.error(f"错误信息: {e.stderr}")
        return False

def run_rq1_experiments():
    """
    RQ1: 偏见如何随辩论轮次演变，是否支持群体极化理论？
    实验：对比不同辩论轮次下的偏见强度变化
    """
    logger.info("开始 RQ1 实验: 偏见演变与群体极化")
    
    # 选择一组有争议的话题
    selected_topics = ["Abortion", "Gun Control", "Immigration", "Death Penalty", "Animal Testing"]
    topics_str = " ".join([f'"{topic}"' for topic in selected_topics])
    
    # 运行不同轮次的实验
    for rounds in [1, 3, 5]:
        cmd = [sys.executable, "run.py", "--experiment", "debate", 
               "--agents", "3", 
               "--rounds", str(rounds), 
               "--topics"] + selected_topics
        
        description = f"RQ1 - {rounds}轮辩论实验"
        if not run_command(cmd, description):
            logger.error(f"RQ1 - {rounds}轮辩论实验失败")
            return False
    
    logger.info("RQ1 实验完成")
    return True

def run_rq2_experiments():
    """
    RQ2: 智能体组合构成（同质化vs异质化）如何影响偏见动态？
    实验：对比同质化和异质化智能体组合的偏见表现
    """
    logger.info("开始 RQ2 实验: 同质化vs异质化智能体组合")
    
    # 选择一组有争议的话题
    selected_topics = ["Abortion", "Gun Control", "Immigration", "Genetic Engineering", "Nuclear Energy"]
    
    # 同质化实验
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "2", "3", 
           "--rounds", "3", 
           "--topics"] + selected_topics + ["--models", "gpt-4.1-nano"]
    
    if not run_command(cmd, "RQ2 - 同质化智能体实验"):
        logger.error("RQ2 - 同质化智能体实验失败")
        return False
    
    # 异质化实验
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "2", "3", 
           "--rounds", "3", 
           "--topics"] + selected_topics + ["--models", "gpt-4.1-nano", "deepseek-chat"]
    
    if not run_command(cmd, "RQ2 - 异质化智能体实验"):
        logger.error("RQ2 - 异质化智能体实验失败")
        return False
    
    logger.info("RQ2 实验完成")
    return True

def run_rq3_experiments():
    """
    RQ3: 哪些语言特征与多智能体辩论中的偏见相关联？
    实验：收集足够的辩论数据，用于后续语言特征分析
    """
    logger.info("开始 RQ3 实验: 语言特征与偏见关联")
    
    # 选择更多样化的话题，确保覆盖不同类型的语言特征
    selected_topics = [
        "Abortion", "Gun Control", "Immigration", 
        "Freedom of Speech", "Cultural Appropriation", 
        "Gender Identity in Sports", "Drug Legalization"
    ]
    
    # 运行多轮辩论以收集丰富的语言样本
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "3", 
           "--rounds", "5", 
           "--topics"] + selected_topics
    
    if not run_command(cmd, "RQ3 - 语言特征收集实验"):
        logger.error("RQ3 - 语言特征收集实验失败")
        return False
    
    logger.info("RQ3 实验完成")
    return True

def generate_visualizations():
    """生成可视化和报告"""
    logger.info("生成可视化和报告")
    
    cmd = [sys.executable, "scripts/generate_all.py", "--open_report"]
    if not run_command(cmd, "生成可视化和报告"):
        logger.error("可视化生成失败")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MAD-bias 论文实验运行脚本")
    
    parser.add_argument(
        "--rq", 
        type=int,
        choices=[1, 2, 3, 0],
        default=0,
        help="要运行的研究问题实验 (1, 2, 3), 0表示全部"
    )
    
    args = parser.parse_args()
    
    logger.info("MAD-bias 论文实验批处理脚本开始执行")
    
    success = True
    
    if args.rq == 0 or args.rq == 1:
        success = run_rq1_experiments() and success
    
    if args.rq == 0 or args.rq == 2:
        success = run_rq2_experiments() and success
    
    if args.rq == 0 or args.rq == 3:
        success = run_rq3_experiments() and success
    
    if success:
        success = generate_visualizations() and success
    
    if success:
        logger.info("所有论文实验完成！")
    else:
        logger.error("论文实验执行出错，请检查日志")
    
    logger.info(f"日志文件保存在: {log_file}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 