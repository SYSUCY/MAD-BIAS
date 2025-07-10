# -*- coding: utf-8 -*-
"""
一键生成可视化和报告脚本
整合所有操作，方便用户一键生成所有可视化和报告
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "generate_all.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_script(script_path, args=None):
    """
    运行指定的脚本
    
    Args:
        script_path: 脚本路径
        args: 命令行参数
    
    Returns:
        bool: 是否成功
    """
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    logger.info(f"运行脚本: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"脚本运行成功: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"脚本运行失败: {script_path}")
        logger.error(f"错误信息: {e.stderr}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一键生成可视化和报告")
    
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
    
    # 报告输出路径
    parser.add_argument(
        "--report_path", 
        type=str,
        default=None,
        help="报告输出路径"
    )
    
    # 是否打开报告
    parser.add_argument(
        "--open_report", 
        action="store_true",
        help="生成后自动打开报告"
    )
    
    args = parser.parse_args()
    
    try:
        # 步骤1：生成可视化图表
        vis_script = os.path.join(os.path.dirname(__file__), "generate_visualizations.py")
        vis_args = []
        
        if args.data_dir:
            vis_args.extend(["--data_dir", args.data_dir])
        if args.output_dir:
            vis_args.extend(["--output_dir", args.output_dir])
        if args.single_dir:
            vis_args.extend(["--single_dir", args.single_dir])
        if args.multi_dir:
            vis_args.extend(["--multi_dir", args.multi_dir])
        
        if not run_script(vis_script, vis_args):
            logger.error("生成可视化图表失败，终止后续操作")
            sys.exit(1)
        
        # 步骤2：生成报告
        report_script = os.path.join(os.path.dirname(__file__), "generate_report.py")
        report_args = []
        
        if args.report_path:
            report_args.extend(["--output", args.report_path])
        if args.single_dir:
            report_args.extend(["--single_dir", args.single_dir])
        if args.multi_dir:
            report_args.extend(["--multi_dir", args.multi_dir])
        if args.output_dir:
            report_args.extend(["--vis_dir", args.output_dir])
        
        if not run_script(report_script, report_args):
            logger.error("生成报告失败")
            sys.exit(1)
        
        # 确定报告路径
        if args.report_path:
            report_path = args.report_path
        else:
            report_path = os.path.join(settings.DATA_PROCESSED_DIR, "report.html")
        
        logger.info(f"所有操作完成，报告已保存到: {report_path}")
        
        # 自动打开报告
        if args.open_report and os.path.exists(report_path):
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
                logger.info("已自动打开报告")
            except Exception as e:
                logger.error(f"自动打开报告失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"生成过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 