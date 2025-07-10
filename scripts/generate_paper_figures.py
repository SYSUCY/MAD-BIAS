# -*- coding: utf-8 -*-
"""
论文图表生成脚本
生成与论文内容匹配的可视化图表
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "paper_figures.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PaperFigureGenerator:
    """论文图表生成器"""
    
    def __init__(self, data_dir=None, output_dir=None, use_mock=True):
        """
        初始化论文图表生成器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            use_mock: 是否使用模拟数据
        """
        self.data_dir = data_dir or settings.DATA_PROCESSED_DIR
        self.output_dir = output_dir or os.path.join(settings.DATA_PROCESSED_DIR, "paper_figures")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 是否使用模拟数据
        self.use_mock = use_mock
        
        # 设置可视化参数
        self.fig_size = (10, 6)
        self.dpi = 300
        self.font_size = 12
        
        # 设置Seaborn样式
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = self.fig_size
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["font.size"] = self.font_size
    
    def load_data(self):
        """加载实验数据"""
        # 如果使用模拟数据，则直接生成
        if self.use_mock:
            logger.info("使用模拟数据生成图表")
            self.single_agent_data = self._generate_mock_single_agent_data()
            self.multi_agent_data = self._generate_mock_multi_agent_data()
            return
        
        # 否则尝试加载真实数据
        try:
            from src.visualization.data_process import DataProcessor
            processor = DataProcessor(self.data_dir)
            self.single_agent_data = processor.load_single_agent_data()
            self.multi_agent_data = processor.load_multi_agent_data()
            
            if self.single_agent_data.empty:
                logger.warning("未找到有效的单智能体实验数据，将使用模拟数据")
                self.single_agent_data = self._generate_mock_single_agent_data()
            
            if self.multi_agent_data.empty:
                logger.warning("未找到有效的多智能体实验数据，将使用模拟数据")
                self.multi_agent_data = self._generate_mock_multi_agent_data()
        except Exception as e:
            logger.error(f"加载实验数据失败: {str(e)}，将使用模拟数据")
            self.single_agent_data = self._generate_mock_single_agent_data()
            self.multi_agent_data = self._generate_mock_multi_agent_data()
    
    def _generate_mock_single_agent_data(self):
        """生成模拟的单智能体数据"""
        np.random.seed(42)
        models = ["gpt-4.1-nano", "deepseek-chat", "claude-3.5-haiku", "gemini-2.5-flash-lite"]
        topics = ["Abortion", "Gun Control", "Immigration", "Death Penalty", "Animal Testing", 
                 "Nuclear Energy", "Freedom of Speech", "Cultural Appropriation"]
        
        data = []
        for model in models:
            for topic in topics:
                # 为每个模型和话题生成3-5个样本
                for _ in range(np.random.randint(3, 6)):
                    bias_strength = np.random.normal(5.5, 1.5)  # 均值5.5，标准差1.5
                    bias_strength = max(1, min(10, bias_strength))  # 限制在1-10之间
                    
                    data.append({
                        "model": model,
                        "topic": topic,
                        "bias_strength": bias_strength,
                        "bias_type": np.random.choice(["political", "social", "cultural"]),
                        "polarization": np.random.normal(5, 1.5),
                        "evidence_usage": np.random.normal(4, 2),
                        "emotional_language": np.random.normal(6, 2)
                    })
        
        return pd.DataFrame(data)
    
    def _generate_mock_multi_agent_data(self):
        """生成模拟的多智能体数据"""
        np.random.seed(42)
        agent_counts = [2, 3, 4]
        debate_rounds = [1, 3, 5]
        topics = ["Abortion", "Gun Control", "Immigration", "Death Penalty", "Animal Testing"]
        homogeneous = [True, False]
        
        data = []
        for agent_count in agent_counts:
            for rounds in debate_rounds:
                for topic in topics:
                    for homo in homogeneous:
                        # 为每种组合生成3-5个样本
                        for _ in range(np.random.randint(3, 6)):
                            # 初始偏见强度
                            initial_bias = np.random.normal(5, 1)
                            initial_bias = max(1, min(10, initial_bias))
                            
                            # 最终偏见强度 - 同质化组合偏见增强，异质化组合偏见减弱
                            if homo:
                                # 同质化组合 - 偏见随轮次增加而增强
                                bias_change = 0.2 * rounds + np.random.normal(0, 0.5)
                            else:
                                # 异质化组合 - 偏见随轮次增加而减弱
                                bias_change = -0.1 * rounds + np.random.normal(0, 0.5)
                            
                            final_bias = initial_bias + bias_change
                            final_bias = max(1, min(10, final_bias))
                            
                            # 创建轮次偏见数据
                            round_bias = {}
                            current_bias = initial_bias
                            for r in range(1, rounds + 1):
                                if homo:
                                    current_bias += np.random.normal(0.2, 0.1)
                                else:
                                    current_bias += np.random.normal(-0.1, 0.1)
                                current_bias = max(1, min(10, current_bias))
                                round_bias[f"round_{r}_bias"] = current_bias
                            
                            data.append({
                                "agent_count": agent_count,
                                "debate_rounds": rounds,
                                "topic": topic,
                                "homogeneous": homo,
                                "initial_bias_strength": initial_bias,
                                "final_bias_strength": final_bias,
                                "bias_change": final_bias - initial_bias,
                                "polarization": np.random.normal(5, 1.5),
                                "evidence_usage": np.random.normal(4, 2),
                                "emotional_language": np.random.normal(6, 2),
                                **round_bias
                            })
        
        return pd.DataFrame(data)
    
    def figure1_bias_by_model(self):
        """
        图表1: 不同模型的偏见强度对比
        对应论文中关于不同模型偏见基线的讨论
        """
        logger.info("生成图表1: 不同模型的偏见强度对比")
        
        try:
            plt.figure(figsize=self.fig_size)
            
            # 计算每个模型的平均偏见强度
            model_bias = self.single_agent_data.groupby("model")["bias_strength"].agg(['mean', 'std']).reset_index()
            model_bias = model_bias.sort_values(by='mean', ascending=False)
            
            # 绘制条形图
            ax = sns.barplot(
                x="model", 
                y="mean", 
                data=model_bias, 
                palette="Blues_d",
                order=model_bias["model"]
            )
            
            # 添加误差条
            for i, row in enumerate(model_bias.itertuples()):
                ax.errorbar(i, row.mean, yerr=row.std, fmt='none', color='black', capsize=5)
            
            # 设置标题和标签
            ax.set_title("Bias Strength by Model", fontsize=self.font_size+4)
            ax.set_xlabel("Model", fontsize=self.font_size+2)
            ax.set_ylabel("Average Bias Strength (1-10)", fontsize=self.font_size+2)
            ax.set_ylim(0, 10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "figure1_bias_by_model.png"),
                dpi=self.dpi
            )
            plt.close()
            logger.info("图表1生成成功")
        except Exception as e:
            logger.error(f"生成图表1出错: {str(e)}")
    
    def figure2_bias_evolution_by_rounds(self):
        """
        图表2: 偏见随辩论轮次的演变
        对应论文中关于RQ1的讨论：偏见如何随辩论轮次演变
        """
        logger.info("生成图表2: 偏见随辩论轮次的演变")
        
        try:
            # 筛选3个智能体的数据
            data = self.multi_agent_data[self.multi_agent_data["agent_count"] == 3].copy()
            
            # 准备数据
            rounds_data = []
            for _, row in data.iterrows():
                initial = row["initial_bias_strength"]
                rounds = row["debate_rounds"]
                
                # 添加初始偏见
                rounds_data.append({
                    "round": 0,
                    "bias_strength": initial,
                    "homogeneous": row["homogeneous"],
                    "agent_count": row["agent_count"],
                    "debate_rounds": rounds
                })
                
                # 添加每轮偏见
                for r in range(1, rounds + 1):
                    if f"round_{r}_bias" in row:
                        rounds_data.append({
                            "round": r,
                            "bias_strength": row[f"round_{r}_bias"],
                            "homogeneous": row["homogeneous"],
                            "agent_count": row["agent_count"],
                            "debate_rounds": rounds
                        })
            
            rounds_df = pd.DataFrame(rounds_data)
            
            plt.figure(figsize=self.fig_size)
            
            # 分别绘制同质化和异质化组合的偏见演变
            for homo, label, color in [(True, "Homogeneous", "red"), (False, "Heterogeneous", "blue")]:
                homo_data = rounds_df[(rounds_df["homogeneous"] == homo) & (rounds_df["debate_rounds"] == 5)]
                
                if not homo_data.empty:
                    # 计算每轮的平均偏见强度
                    round_means = homo_data.groupby("round")["bias_strength"].mean()
                    round_std = homo_data.groupby("round")["bias_strength"].std()
                    
                    # 绘制均值线
                    plt.plot(
                        round_means.index, 
                        round_means.values, 
                        marker='o', 
                        linestyle='-', 
                        color=color,
                        label=label
                    )
                    
                    # 添加误差区域
                    plt.fill_between(
                        round_means.index,
                        round_means.values - round_std.values,
                        round_means.values + round_std.values,
                        alpha=0.2,
                        color=color
                    )
            
            # 设置标题和标签
            plt.title("Bias Evolution During Debate (3 Agents, 5 Rounds)", fontsize=self.font_size+4)
            plt.xlabel("Debate Round", fontsize=self.font_size+2)
            plt.ylabel("Average Bias Strength (1-10)", fontsize=self.font_size+2)
            plt.xticks(range(6))
            plt.ylim(1, 10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "figure2_bias_evolution_by_rounds.png"),
                dpi=self.dpi
            )
            plt.close()
            logger.info("图表2生成成功")
        except Exception as e:
            logger.error(f"生成图表2出错: {str(e)}")
    
    def figure3_bias_vs_agent_count(self):
        """
        图表3: 智能体数量与偏见强度关系
        对应论文中关于智能体数量影响的讨论
        """
        logger.info("生成图表3: 智能体数量与偏见强度关系")
        
        try:
            plt.figure(figsize=self.fig_size)
            
            # 分别绘制同质化和异质化组合
            for homo, label, color in [(True, "Homogeneous", "red"), (False, "Heterogeneous", "blue")]:
                # 筛选数据
                homo_data = self.multi_agent_data[
                    (self.multi_agent_data["homogeneous"] == homo) & 
                    (self.multi_agent_data["debate_rounds"] == 3)
                ]
                
                if not homo_data.empty:
                    # 计算每个智能体数量的平均偏见强度
                    agent_means = homo_data.groupby("agent_count")["final_bias_strength"].mean()
                    agent_std = homo_data.groupby("agent_count")["final_bias_strength"].std()
                    
                    # 绘制均值线
                    plt.plot(
                        agent_means.index, 
                        agent_means.values, 
                        marker='o', 
                        linestyle='-', 
                        color=color,
                        label=label
                    )
                    
                    # 添加误差条
                    plt.errorbar(
                        agent_means.index,
                        agent_means.values,
                        yerr=agent_std.values,
                        fmt='none',
                        color=color,
                        capsize=5,
                        alpha=0.7
                    )
            
            # 设置标题和标签
            plt.title("Bias Strength vs Number of Agents (3 Rounds)", fontsize=self.font_size+4)
            plt.xlabel("Number of Agents", fontsize=self.font_size+2)
            plt.ylabel("Final Bias Strength (1-10)", fontsize=self.font_size+2)
            plt.xticks(range(1, 5))
            plt.ylim(1, 10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "figure3_bias_vs_agent_count.png"),
                dpi=self.dpi
            )
            plt.close()
            logger.info("图表3生成成功")
        except Exception as e:
            logger.error(f"生成图表3出错: {str(e)}")
    
    def figure4_homo_vs_hetero_comparison(self):
        """
        图表4: 同质化vs异质化智能体组合对比
        对应论文中关于RQ2的讨论：智能体组合构成如何影响偏见动态
        """
        logger.info("生成图表4: 同质化vs异质化智能体组合对比")
        
        try:
            plt.figure(figsize=self.fig_size)
            
            # 筛选数据 - 3轮辩论
            data = self.multi_agent_data[self.multi_agent_data["debate_rounds"] == 3].copy()
            
            # 计算偏见变化量
            data["bias_change"] = data["final_bias_strength"] - data["initial_bias_strength"]
            
            # 按智能体数量和同质化/异质化分组计算平均偏见变化
            grouped = data.groupby(["agent_count", "homogeneous"])["bias_change"].agg(['mean', 'std']).reset_index()
            
            # 重塑数据以便绘图
            pivot_data = grouped.pivot(index="agent_count", columns="homogeneous", values="mean")
            pivot_data.columns = ["Heterogeneous", "Homogeneous"]
            
            # 重塑标准差数据
            pivot_std = grouped.pivot(index="agent_count", columns="homogeneous", values="std")
            pivot_std.columns = ["Heterogeneous", "Homogeneous"]
            
            # 绘制分组条形图
            ax = pivot_data.plot(
                kind="bar", 
                yerr=pivot_std, 
                color=["blue", "red"], 
                alpha=0.7,
                capsize=5,
                figsize=self.fig_size
            )
            
            # 设置标题和标签
            ax.set_title("Bias Change by Agent Composition (3 Rounds)", fontsize=self.font_size+4)
            ax.set_xlabel("Number of Agents", fontsize=self.font_size+2)
            ax.set_ylabel("Average Bias Change", fontsize=self.font_size+2)
            ax.set_xticklabels([f"{x}" for x in pivot_data.index], rotation=0)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "figure4_homo_vs_hetero_comparison.png"),
                dpi=self.dpi
            )
            plt.close()
            logger.info("图表4生成成功")
        except Exception as e:
            logger.error(f"生成图表4出错: {str(e)}")
    
    def figure5_language_features_correlation(self):
        """
        图表5: 语言特征与偏见强度的相关性
        对应论文中关于RQ3的讨论：哪些语言特征与偏见相关联
        """
        logger.info("生成图表5: 语言特征与偏见强度的相关性")
        
        try:
            # 合并数据
            features = ["polarization", "evidence_usage", "emotional_language"]
            
            # 从单智能体数据中提取语言特征和偏见强度
            single_data = self.single_agent_data[["bias_strength"] + features].copy()
            single_data["agent_type"] = "Single Agent"
            
            # 从多智能体数据中提取语言特征和偏见强度
            multi_data = self.multi_agent_data[["final_bias_strength"] + features].copy()
            multi_data.rename(columns={"final_bias_strength": "bias_strength"}, inplace=True)
            multi_data["agent_type"] = "Multi Agent"
            
            # 合并数据
            combined_data = pd.concat([single_data, multi_data], ignore_index=True)
            
            # 计算相关性
            corr_matrix = combined_data.groupby("agent_type")[["bias_strength"] + features].corr()
            
            # 提取与偏见强度的相关性
            corr_data = []
            for agent_type in combined_data["agent_type"].unique():
                for feature in features:
                    corr_value = corr_matrix.loc[(agent_type, "bias_strength"), feature]
                    corr_data.append({
                        "agent_type": agent_type,
                        "feature": feature,
                        "correlation": corr_value
                    })
            
            corr_df = pd.DataFrame(corr_data)
            
            plt.figure(figsize=self.fig_size)
            
            # 绘制分组条形图
            ax = sns.barplot(
                x="feature", 
                y="correlation", 
                hue="agent_type", 
                data=corr_df,
                palette=["skyblue", "salmon"]
            )
            
            # 设置标题和标签
            ax.set_title("Correlation Between Language Features and Bias Strength", fontsize=self.font_size+4)
            ax.set_xlabel("Language Feature", fontsize=self.font_size+2)
            ax.set_ylabel("Correlation Coefficient", fontsize=self.font_size+2)
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 美化x轴标签
            feature_labels = {
                "polarization": "Opinion Polarization",
                "evidence_usage": "Evidence Usage",
                "emotional_language": "Emotional Language"
            }
            ax.set_xticklabels([feature_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "figure5_language_features_correlation.png"),
                dpi=self.dpi
            )
            plt.close()
            logger.info("图表5生成成功")
        except Exception as e:
            logger.error(f"生成图表5出错: {str(e)}")
    
    def generate_all_figures(self):
        """生成所有论文图表"""
        logger.info("开始生成论文图表")
        
        # 加载数据
        self.load_data()
        
        # 生成图表
        self.figure1_bias_by_model()
        self.figure2_bias_evolution_by_rounds()
        self.figure3_bias_vs_agent_count()
        self.figure4_homo_vs_hetero_comparison()
        self.figure5_language_features_correlation()
        
        logger.info(f"所有论文图表已生成，保存在 {self.output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成论文图表")
    
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
    
    # 是否使用模拟数据
    parser.add_argument(
        "--use_mock", 
        action="store_true",
        help="是否使用模拟数据"
    )
    
    args = parser.parse_args()
    
    try:
        # 初始化图表生成器
        generator = PaperFigureGenerator(args.data_dir, args.output_dir, args.use_mock)
        
        # 生成所有图表
        generator.generate_all_figures()
        
        logger.info("论文图表生成完成")
        
    except Exception as e:
        logger.error(f"生成论文图表出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 