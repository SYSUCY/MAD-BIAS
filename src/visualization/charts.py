# -*- coding: utf-8 -*-
"""
可视化图表生成模块
生成实验结果的可视化图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from config import settings

logger = logging.getLogger(__name__)

class BiasVisualization:
    """
    偏见可视化类
    生成偏见相关的可视化图表
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化可视化类
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir or os.path.join(settings.DATA_PROCESSED_DIR, "visualization")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 应用可视化设置
        self.fig_size = settings.VISUALIZATION_SETTINGS["figure_size"]
        self.dpi = settings.VISUALIZATION_SETTINGS["dpi"]
        self.font_size = settings.VISUALIZATION_SETTINGS["font_size"]
        self.color_palette = settings.VISUALIZATION_SETTINGS["color_palette"]
        self.save_format = settings.VISUALIZATION_SETTINGS["save_format"]
        
        # 设置Seaborn样式
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = self.fig_size
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["font.size"] = self.font_size
        
    def plot_bias_by_model(self, data: pd.DataFrame, save_name: str = "bias_by_model") -> None:
        """
        绘制不同模型的偏见强度对比图
        
        Args:
            data: 数据框，包含模型和偏见强度信息
            save_name: 保存的文件名
        """
        plt.figure(figsize=self.fig_size)
        
        # 绘制箱线图
        ax = sns.boxplot(x="model", y="bias_strength", data=data, palette=self.color_palette)
        
        # 添加均值点
        means = data.groupby("model")["bias_strength"].mean()
        ax.scatter(
            x=range(len(means)), 
            y=means.values, 
            color='red', 
            marker='o', 
            s=50,
            label="Mean"
        )
        
        # 设置标题和标签
        ax.set_title("Bias Strength by Model", fontsize=self.font_size+4)
        ax.set_xlabel("Model", fontsize=self.font_size+2)
        ax.set_ylabel("Bias Strength", fontsize=self.font_size+2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close()
        
    def plot_bias_by_topic(self, data: pd.DataFrame, save_name: str = "bias_by_topic") -> None:
        """
        绘制不同话题的偏见强度对比图
        
        Args:
            data: 数据框，包含话题和偏见强度信息
            save_name: 保存的文件名
        """
        # 计算每个话题的平均偏见强度并排序
        topic_bias = data.groupby("topic")["bias_strength"].mean().sort_values(ascending=False)
        top_topics = topic_bias.head(10).index.tolist()
        
        # 筛选数据
        plot_data = data[data["topic"].isin(top_topics)]
        
        plt.figure(figsize=self.fig_size)
        
        # 绘制箱线图
        ax = sns.boxplot(
            x="topic", 
            y="bias_strength", 
            data=plot_data, 
            palette=self.color_palette,
            order=top_topics
        )
        
        # 添加均值点
        means = plot_data.groupby("topic")["bias_strength"].mean()
        means = means.reindex(top_topics)  # 确保顺序一致
        
        ax.scatter(
            x=range(len(means)), 
            y=means.values, 
            color='red', 
            marker='o', 
            s=50,
            label="Mean"
        )
        
        # 设置标题和标签
        ax.set_title("Bias Strength by Topic (Top 10)", fontsize=self.font_size+4)
        ax.set_xlabel("Topic", fontsize=self.font_size+2)
        ax.set_ylabel("Bias Strength", fontsize=self.font_size+2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close()
        
    def plot_bias_vs_agent_count(self, data: pd.DataFrame, save_name: str = "bias_vs_agent_count") -> None:
        """
        绘制智能体数量与偏见强度关系图
        
        Args:
            data: 数据框，包含智能体数量和偏见强度信息
            save_name: 保存的文件名
        """
        plt.figure(figsize=self.fig_size)
        
        # 绘制箱线图
        ax = sns.boxplot(
            x="agent_count", 
            y="final_bias_strength", 
            data=data, 
            palette=self.color_palette
        )
        
        # 添加均值点
        means = data.groupby("agent_count")["final_bias_strength"].mean()
        ax.scatter(
            x=range(len(means)), 
            y=means.values, 
            color='red', 
            marker='o', 
            s=50,
            label="Mean"
        )
        
        # 添加均值连线
        sns.pointplot(
            x="agent_count", 
            y="final_bias_strength", 
            data=data, 
            color="red", 
            markers="", 
            linestyles="-", 
            scale=0.7,
            errorbar=None,
            ax=ax
        )
        
        # 设置标题和标签
        ax.set_title("Bias Strength vs Number of Agents", fontsize=self.font_size+4)
        ax.set_xlabel("Number of Agents", fontsize=self.font_size+2)
        ax.set_ylabel("Final Bias Strength", fontsize=self.font_size+2)
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close()
        
    def plot_bias_vs_debate_rounds(self, data: pd.DataFrame, save_name: str = "bias_vs_debate_rounds") -> None:
        """
        绘制辩论轮次与偏见强度关系图
        
        Args:
            data: 数据框，包含辩论轮次和偏见强度信息
            save_name: 保存的文件名
        """
        plt.figure(figsize=self.fig_size)
        
        # 绘制箱线图
        ax = sns.boxplot(
            x="debate_rounds", 
            y="final_bias_strength", 
            data=data, 
            palette=self.color_palette
        )
        
        # 添加均值点
        means = data.groupby("debate_rounds")["final_bias_strength"].mean()
        ax.scatter(
            x=range(len(means)), 
            y=means.values, 
            color='red', 
            marker='o', 
            s=50,
            label="Mean"
        )
        
        # 添加均值连线
        sns.pointplot(
            x="debate_rounds", 
            y="final_bias_strength", 
            data=data, 
            color="red", 
            markers="", 
            linestyles="-", 
            scale=0.7,
            errorbar=None,
            ax=ax
        )
        
        # 设置标题和标签
        ax.set_title("Bias Strength vs Debate Rounds", fontsize=self.font_size+4)
        ax.set_xlabel("Number of Debate Rounds", fontsize=self.font_size+2)
        ax.set_ylabel("Final Bias Strength", fontsize=self.font_size+2)
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close()
        
    def plot_bias_evolution(self, 
                           data: pd.DataFrame, 
                           agent_count: int = 4, 
                           debate_rounds: int = 5,
                           save_name: Optional[str] = None) -> None:
        """
        绘制偏见随辩论轮次的演变
        
        Args:
            data: 数据框，包含轮次和偏见强度信息
            agent_count: 智能体数量过滤
            debate_rounds: 辩论轮次过滤
            save_name: 保存的文件名
        """
        # 筛选数据
        filtered_data = data[
            (data["agent_count"] == agent_count) & 
            (data["debate_rounds"] == debate_rounds)
        ]
        
        if filtered_data.empty:
            logger.warning(f"没有找到符合条件的数据: agent_count={agent_count}, debate_rounds={debate_rounds}")
            return
            
        plt.figure(figsize=self.fig_size)
        
        # 重塑数据结构：需要包含每个轮次的偏见强度数据
        
        # 简化版：假设我们已经有了重塑后的数据，轮次为x轴，不同智能体的偏见强度为y值
        # 实际应用中需要根据实际数据结构进行重塑
        
        # 示例绘图
        ax = sns.lineplot(
            x="round_num", 
            y="bias_strength", 
            hue="agent_id",
            data=filtered_data,  # 这里应该是重塑后的数据
            palette=self.color_palette,
            markers=True,
            dashes=False
        )
        
        # 设置标题和标签
        ax.set_title(f"Evolution of Bias Strength During Debate (Agents={agent_count}, Rounds={debate_rounds})", 
                    fontsize=self.font_size+4)
        ax.set_xlabel("Debate Round", fontsize=self.font_size+2)
        ax.set_ylabel("Bias Strength", fontsize=self.font_size+2)
        
        # 保存图表
        plt.tight_layout()
        if not save_name:
            save_name = f"bias_evolution_a{agent_count}_r{debate_rounds}"
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close()
        
    def plot_homogeneous_vs_heterogeneous(self, 
                                         homo_data: pd.DataFrame, 
                                         hetero_data: pd.DataFrame,
                                         save_name: str = "homo_vs_hetero") -> None:
        """
        比较同质化和异质化智能体组合的偏见强度
        
        Args:
            homo_data: 同质化智能体组合数据
            hetero_data: 异质化智能体组合数据
            save_name: 保存的文件名
        """
        plt.figure(figsize=self.fig_size)
        
        # 合并数据并添加类型列
        homo_data = homo_data.copy()
        homo_data["group_type"] = "Homogeneous"
        
        hetero_data = hetero_data.copy()
        hetero_data["group_type"] = "Heterogeneous"
        
        combined_data = pd.concat([homo_data, hetero_data])
        
        # 绘制箱线图
        ax = sns.boxplot(
            x="agent_count", 
            y="final_bias_strength", 
            hue="group_type",
            data=combined_data, 
            palette=self.color_palette
        )
        
        # 设置标题和标签
        ax.set_title("Homogeneous vs Heterogeneous Agent Groups", fontsize=self.font_size+4)
        ax.set_xlabel("Number of Agents", fontsize=self.font_size+2)
        ax.set_ylabel("Final Bias Strength", fontsize=self.font_size+2)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            dpi=self.dpi
        )
        plt.close() 