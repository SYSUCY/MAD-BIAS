# -*- coding: utf-8 -*-
"""
数据处理模块
处理和准备可视化数据
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import glob

from config import settings

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    数据处理器
    处理实验结果数据，准备用于可视化
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or settings.DATA_PROCESSED_DIR
        
    def load_single_agent_data(self, specific_dir: Optional[str] = None) -> pd.DataFrame:
        """
        加载单智能体实验数据
        
        Args:
            specific_dir: 指定的数据目录，如果为None则搜索最新的目录
            
        Returns:
            pd.DataFrame: 单智能体实验数据
        """
        # 确定数据目录
        if specific_dir:
            csv_path = os.path.join(specific_dir, "summary_results.csv")
        else:
            # 查找最新的single_agent目录
            single_agent_dirs = glob.glob(os.path.join(self.data_dir, "single_agent_*"))
            if not single_agent_dirs:
                logger.error("未找到单智能体实验数据目录")
                return pd.DataFrame()
                
            # 按修改时间排序
            latest_dir = max(single_agent_dirs, key=os.path.getmtime)
            csv_path = os.path.join(latest_dir, "summary_results.csv")
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            logger.error(f"未找到单智能体实验数据文件: {csv_path}")
            return pd.DataFrame()
            
        try:
            # 加载数据
            data = pd.read_csv(csv_path)
            logger.info(f"已加载单智能体实验数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载单智能体实验数据出错: {str(e)}")
            return pd.DataFrame()
    
    def load_multi_agent_data(self, specific_dir: Optional[str] = None) -> pd.DataFrame:
        """
        加载多智能体实验数据
        
        Args:
            specific_dir: 指定的数据目录，如果为None则搜索最新的目录
            
        Returns:
            pd.DataFrame: 多智能体实验数据
        """
        # 确定数据目录
        if specific_dir:
            csv_path = os.path.join(specific_dir, "summary_results.csv")
        else:
            # 查找最新的multi_agent目录
            multi_agent_dirs = glob.glob(os.path.join(self.data_dir, "multi_agent_*"))
            if not multi_agent_dirs:
                logger.error("未找到多智能体实验数据目录")
                return pd.DataFrame()
                
            # 按修改时间排序
            latest_dir = max(multi_agent_dirs, key=os.path.getmtime)
            csv_path = os.path.join(latest_dir, "summary_results.csv")
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            logger.error(f"未找到多智能体实验数据文件: {csv_path}")
            return pd.DataFrame()
            
        try:
            # 加载数据
            data = pd.read_csv(csv_path)
            logger.info(f"已加载多智能体实验数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载多智能体实验数据出错: {str(e)}")
            return pd.DataFrame()
    
    def prepare_bias_model_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备模型偏见数据
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 检查必要列是否存在
        required_cols = ["model", "bias_strength"]
        if not all(col in data.columns for col in required_cols):
            logger.error(f"数据缺少必要列: {required_cols}")
            return pd.DataFrame()
        
        # 筛选有效数据
        valid_data = data.dropna(subset=["bias_strength", "model"])
        
        # 按模型分组并计算统计量
        model_stats = valid_data.groupby("model").agg({
            "bias_strength": ["mean", "median", "std", "count"]
        }).reset_index()
        
        # 展平多级列名
        model_stats.columns = ["_".join(col).strip("_") for col in model_stats.columns.values]
        
        return {
            "raw_data": valid_data,
            "stats": model_stats
        }
    
    def prepare_bias_topic_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        准备话题偏见数据
        
        Args:
            data: 原始数据
            
        Returns:
            Dict[str, Any]: 处理后的数据
        """
        # 检查必要列是否存在
        required_cols = ["topic", "bias_strength"]
        if not all(col in data.columns for col in required_cols):
            logger.error(f"数据缺少必要列: {required_cols}")
            return {}
        
        # 筛选有效数据
        valid_data = data.dropna(subset=["bias_strength", "topic"])
        
        # 按话题分组并计算统计量
        topic_stats = valid_data.groupby("topic").agg({
            "bias_strength": ["mean", "median", "std", "count"]
        }).reset_index()
        
        # 展平多级列名
        topic_stats.columns = ["_".join(col).strip("_") for col in topic_stats.columns.values]
        
        # 按平均偏见强度排序
        topic_stats = topic_stats.sort_values(by="bias_strength_mean", ascending=False)
        
        return {
            "raw_data": valid_data,
            "stats": topic_stats
        }
    
    def prepare_bias_evolution_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备偏见演变数据
        
        Args:
            data: 原始数据，应包含完整的评估结果
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 检查是否包含完整的评估数据
        if "evaluations" not in data.columns and not isinstance(data, pd.DataFrame):
            logger.error("数据不包含完整的评估结果，需要加载完整JSON数据")
            return pd.DataFrame()
            
        # 以下代码假设data是完整的JSON数据加载后的DataFrame
        # 实际应用中需要调整以适应实际数据结构
        
        # 创建用于可视化的数据框
        evolution_data = []
        
        # 处理每个实验结果
        for _, row in data.iterrows():
            experiment_id = row.get("experiment_id")
            topic = row.get("topic")
            agent_count = row.get("agent_count")
            debate_rounds = row.get("debate_rounds")
            
            # 获取评估结果
            evaluations = row.get("evaluations", {})
            
            # 处理每个智能体的评估
            for agent_id, evals in evaluations.items():
                # 处理每轮的评估
                for eval_data in evals:
                    round_num = eval_data.get("round_num")
                    bias_strength = eval_data.get("bias_strength")
                    polarization = eval_data.get("polarization")
                    
                    # 添加到结果列表
                    if round_num is not None and bias_strength is not None:
                        evolution_data.append({
                            "experiment_id": experiment_id,
                            "topic": topic,
                            "agent_count": agent_count,
                            "debate_rounds": debate_rounds,
                            "agent_id": agent_id,
                            "round_num": round_num,
                            "bias_strength": bias_strength,
                            "polarization": polarization
                        })
        
        # 转换为DataFrame
        if evolution_data:
            return pd.DataFrame(evolution_data)
        else:
            logger.warning("未找到有效的偏见演变数据")
            return pd.DataFrame()
    
    def split_homo_hetero_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分离同质化和异质化数据
        
        Args:
            data: 原始数据，应包含组类型信息
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 同质化和异质化数据
        """
        # 检查必要列是否存在
        if "group_type" not in data.columns:
            logger.error("数据不包含组类型信息")
            return pd.DataFrame(), pd.DataFrame()
        
        # 分离数据
        homo_data = data[data["group_type"] == "homogeneous"]
        hetero_data = data[data["group_type"] == "heterogeneous"]
        
        return homo_data, hetero_data 