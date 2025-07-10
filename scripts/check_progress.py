# -*- coding: utf-8 -*-
"""
实验进度检查脚本
用于检查实验进度和数据质量
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def scan_experiment_directories():
    """扫描实验目录，获取所有实验数据"""
    data_dir = settings.DATA_PROCESSED_DIR
    
    # 获取所有实验目录
    single_agent_dirs = list(data_dir.glob("single_agent_*"))
    multi_agent_dirs = list(data_dir.glob("multi_agent_*"))
    
    # 按时间排序
    single_agent_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    multi_agent_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    return {
        "single_agent": single_agent_dirs,
        "multi_agent": multi_agent_dirs
    }

def analyze_experiment_data(exp_dir):
    """分析单个实验目录的数据"""
    summary_file = exp_dir / "summary_results.csv"
    full_results_file = exp_dir / "full_results.json"
    
    results = {
        "directory": exp_dir.name,
        "created_time": datetime.fromtimestamp(os.path.getctime(exp_dir)).strftime("%Y-%m-%d %H:%M:%S"),
        "has_summary": summary_file.exists(),
        "has_full_results": full_results_file.exists(),
    }
    
    # 分析摘要数据
    if results["has_summary"]:
        try:
            df = pd.read_csv(summary_file)
            results["summary_rows"] = len(df)
            results["summary_size"] = format_size(os.path.getsize(summary_file))
            
            # 分析数据内容
            if "bias_score" in df.columns:
                results["avg_bias_score"] = df["bias_score"].mean()
                results["min_bias_score"] = df["bias_score"].min()
                results["max_bias_score"] = df["bias_score"].max()
            
            # 检查实验类型
            if "agent_count" in df.columns:
                results["experiment_type"] = "multi_agent"
                results["agent_counts"] = df["agent_count"].unique().tolist()
                if "debate_round" in df.columns:
                    results["debate_rounds"] = df["debate_round"].unique().tolist()
            else:
                results["experiment_type"] = "single_agent"
            
            # 检查话题
            if "topic" in df.columns:
                results["topics"] = df["topic"].unique().tolist()
                results["topic_count"] = len(results["topics"])
            
        except Exception as e:
            results["summary_error"] = str(e)
    
    # 分析完整结果数据
    if results["has_full_results"]:
        results["full_results_size"] = format_size(os.path.getsize(full_results_file))
        
        try:
            with open(full_results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results["experiment_count"] = len(data)
                
                # 检查是否有完整的实验数据
                complete_exps = [exp for exp in data if exp.get("completed", False)]
                results["completed_experiments"] = len(complete_exps)
                results["completion_rate"] = results["completed_experiments"] / results["experiment_count"] if results["experiment_count"] > 0 else 0
                
        except Exception as e:
            results["full_results_error"] = str(e)
    
    return results

def print_experiment_summary(exp_data):
    """打印实验摘要信息"""
    print(f"\n{'=' * 80}")
    print(f"实验目录: {exp_data['directory']}")
    print(f"创建时间: {exp_data['created_time']}")
    
    if exp_data.get("experiment_type"):
        print(f"实验类型: {exp_data['experiment_type']}")
    
    if exp_data.get("topics"):
        print(f"话题数量: {exp_data['topic_count']}")
        print(f"话题列表: {', '.join(exp_data['topics'])}")
    
    if exp_data.get("agent_counts"):
        print(f"智能体数量: {exp_data['agent_counts']}")
    
    if exp_data.get("debate_rounds"):
        print(f"辩论轮次: {exp_data['debate_rounds']}")
    
    if exp_data.get("avg_bias_score") is not None:
        print(f"平均偏见分数: {exp_data['avg_bias_score']:.2f} (范围: {exp_data['min_bias_score']:.2f}-{exp_data['max_bias_score']:.2f})")
    
    if exp_data.get("completed_experiments") is not None:
        print(f"完成率: {exp_data['completed_experiments']}/{exp_data['experiment_count']} ({exp_data['completion_rate']*100:.1f}%)")
    
    if exp_data.get("summary_size"):
        print(f"摘要文件大小: {exp_data['summary_size']}")
    
    if exp_data.get("full_results_size"):
        print(f"完整结果文件大小: {exp_data['full_results_size']}")
    
    print(f"{'=' * 80}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检查MAD-bias实验进度和数据质量")
    
    parser.add_argument(
        "--type", 
        type=str,
        choices=["single", "multi", "all"],
        default="all",
        help="要检查的实验类型"
    )
    
    parser.add_argument(
        "--count", 
        type=int,
        default=3,
        help="要显示的最新实验数量"
    )
    
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="是否显示详细信息"
    )
    
    args = parser.parse_args()
    
    # 扫描实验目录
    experiment_dirs = scan_experiment_directories()
    
    # 确定要分析的目录
    dirs_to_analyze = []
    if args.type in ["single", "all"]:
        dirs_to_analyze.extend(experiment_dirs["single_agent"][:args.count])
    if args.type in ["multi", "all"]:
        dirs_to_analyze.extend(experiment_dirs["multi_agent"][:args.count])
    
    # 分析并打印结果
    print(f"\n{'=' * 80}")
    print(f"MAD-bias 实验进度检查")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    
    print(f"\n单智能体实验总数: {len(experiment_dirs['single_agent'])}")
    print(f"多智能体实验总数: {len(experiment_dirs['multi_agent'])}")
    
    # 分析每个目录
    for exp_dir in dirs_to_analyze:
        exp_data = analyze_experiment_data(exp_dir)
        print_experiment_summary(exp_data)
    
    print(f"\n{'=' * 80}")
    print("检查完成!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 