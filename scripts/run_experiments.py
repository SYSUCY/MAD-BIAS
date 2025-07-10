# -*- coding: utf-8 -*-
"""
MAD-bias 实验运行脚本
按顺序运行所有必要的实验，并记录日志
"""

import os
import sys
import time
import logging
import subprocess
import datetime
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings

# 创建日志目录
os.makedirs(os.path.join(settings.LOG_DIR, "experiments"), exist_ok=True)

# 配置日志
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(settings.LOG_DIR, "experiments", f"run_{timestamp}.log")

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

def main():
    """主函数"""
    logger.info("MAD-bias 实验批处理脚本开始执行")
    
    # 1. 单智能体基线实验 - 所有模型，多个话题
    cmd = [sys.executable, "run.py", "--experiment", "baseline", 
           "--models", "gpt-4.1-nano", "deepseek-chat", 
           "--topic_count", "10"]
    if not run_command(cmd, "单智能体基线实验 (所有模型)"):
        logger.error("单智能体实验失败，终止执行")
        return False
    
    # 2. 多智能体同质化实验 - 2个智能体，不同轮次
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "2", 
           "--rounds", "1", "3", "5", 
           "--topic_count", "8"]
    if not run_command(cmd, "多智能体同质化实验 (2个智能体)"):
        logger.error("2智能体实验失败，终止执行")
        return False
    
    # 3. 多智能体同质化实验 - 3个智能体，不同轮次
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "3", 
           "--rounds", "1", "3", "5", 
           "--topic_count", "8"]
    if not run_command(cmd, "多智能体同质化实验 (3个智能体)"):
        logger.error("3智能体实验失败，终止执行")
        return False
    
    # 4. 多智能体异质化实验
    cmd = [sys.executable, "run.py", "--experiment", "debate", 
           "--agents", "2", "3", 
           "--rounds", "3", "5", 
           "--topic_count", "6", 
           "--models", "gpt-4.1-nano", "deepseek-chat"]
    if not run_command(cmd, "多智能体异质化实验"):
        logger.error("异质化实验失败，终止执行")
        return False
    
    # 5. 生成可视化和报告
    cmd = [sys.executable, "scripts/generate_all.py", "--open_report"]
    if not run_command(cmd, "生成可视化和报告"):
        logger.error("可视化生成失败，终止执行")
        return False
    
    logger.info("所有实验完成！")
    logger.info(f"日志文件保存在: {log_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 