# -*- coding: utf-8 -*-
"""
评估指标模块
计算实验相关的定量指标
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from config import settings

logger = logging.getLogger(__name__)

class BiasMetrics:
    """
    偏见相关的评估指标
    """
    
    @staticmethod
    def calculate_bias_strength(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算偏见强度相关指标
        
        Args:
            evaluations: 偏见评估结果列表
            
        Returns:
            Dict[str, Any]: 偏见强度指标
        """
        # 提取偏见强度值
        bias_strengths = [
            eval_result.get("bias_strength") 
            for eval_result in evaluations 
            if eval_result.get("bias_strength") is not None
        ]
        
        if not bias_strengths:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0
            }
        
        return {
            "mean": np.mean(bias_strengths),
            "median": np.median(bias_strengths),
            "std": np.std(bias_strengths),
            "min": np.min(bias_strengths),
            "max": np.max(bias_strengths),
            "count": len(bias_strengths)
        }
    
    @staticmethod
    def calculate_polarization(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算极化程度相关指标
        
        Args:
            evaluations: 偏见评估结果列表
            
        Returns:
            Dict[str, Any]: 极化程度指标
        """
        # 提取极化程度值
        polarizations = [
            eval_result.get("polarization") 
            for eval_result in evaluations 
            if eval_result.get("polarization") is not None
        ]
        
        if not polarizations:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0
            }
        
        return {
            "mean": np.mean(polarizations),
            "median": np.median(polarizations),
            "std": np.std(polarizations),
            "min": np.min(polarizations),
            "max": np.max(polarizations),
            "count": len(polarizations)
        }
    
    @staticmethod
    def calculate_emotional_language(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算情绪化语言程度相关指标
        
        Args:
            evaluations: 偏见评估结果列表
            
        Returns:
            Dict[str, Any]: 情绪化语言指标
        """
        # 提取情绪化语言程度值
        emotional_langs = [
            eval_result.get("emotional_language") 
            for eval_result in evaluations 
            if eval_result.get("emotional_language") is not None
        ]
        
        if not emotional_langs:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0
            }
        
        return {
            "mean": np.mean(emotional_langs),
            "median": np.median(emotional_langs),
            "std": np.std(emotional_langs),
            "min": np.min(emotional_langs),
            "max": np.max(emotional_langs),
            "count": len(emotional_langs)
        }
    
    @staticmethod
    def calculate_bias_type_distribution(evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        计算偏见类型分布
        
        Args:
            evaluations: 偏见评估结果列表
            
        Returns:
            Dict[str, int]: 偏见类型分布
        """
        # 提取偏见类型
        bias_types = [
            eval_result.get("bias_type") 
            for eval_result in evaluations 
            if eval_result.get("bias_type")
        ]
        
        # 统计各类型数量
        type_counts = {}
        for bias_type in bias_types:
            # 处理可能包含多个类型的情况（如"政治偏见，社会偏见"）
            for btype in bias_type.split("，"):
                btype = btype.strip()
                if btype:
                    type_counts[btype] = type_counts.get(btype, 0) + 1
        
        return type_counts
    
    @staticmethod
    def calculate_consensus_index(evaluations: List[Dict[str, Any]]) -> float:
        """
        计算共识指数（衡量评估结果的一致性）
        
        Args:
            evaluations: 偏见评估结果列表
            
        Returns:
            float: 共识指数（0-1之间，1表示完全一致）
        """
        if len(evaluations) <= 1:
            return 1.0
        
        # 提取偏见强度和极化程度
        bias_strengths = [
            eval_result.get("bias_strength") 
            for eval_result in evaluations 
            if eval_result.get("bias_strength") is not None
        ]
        
        polarizations = [
            eval_result.get("polarization") 
            for eval_result in evaluations 
            if eval_result.get("polarization") is not None
        ]
        
        if not bias_strengths or not polarizations:
            return 0.0
        
        # 计算标准差
        bs_std = np.std(bias_strengths)
        pol_std = np.std(polarizations)
        
        # 计算最大可能标准差（对于1-10的评分范围）
        max_std = (10 - 1) / 2
        
        # 计算共识指数
        bs_consensus = 1 - min(bs_std / max_std, 1)
        pol_consensus = 1 - min(pol_std / max_std, 1)
        
        # 取平均
        return (bs_consensus + pol_consensus) / 2

class DebateMetrics:
    """
    辩论相关的评估指标
    """
    
    @staticmethod
    def calculate_opinion_change(debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算辩论过程中的观点变化
        
        Args:
            debate_result: 辩论实验结果
            
        Returns:
            Dict[str, Any]: 观点变化指标
        """
        # 获取辩论轮次和智能体
        rounds = debate_result.get("rounds", [])
        if len(rounds) < 2:  # 至少需要初始轮和一轮辩论
            return {"error": "辩论轮次不足"}
        
        agents = debate_result.get("agents", [])
        agent_ids = [agent["agent_id"] for agent in agents]
        
        # 使用偏见检测器的评估结果
        evaluations = debate_result.get("evaluations", {})
        if not evaluations:
            return {"error": "缺少偏见评估结果"}
        
        # 计算每个智能体的观点变化
        changes = {}
        for agent_id in agent_ids:
            agent_evals = evaluations.get(agent_id, [])
            if len(agent_evals) < 2:
                changes[agent_id] = {
                    "bias_strength_change": None,
                    "polarization_change": None
                }
                continue
            
            # 计算初始和最终偏见强度的变化
            initial_bs = agent_evals[0].get("bias_strength")
            final_bs = agent_evals[-1].get("bias_strength")
            
            # 计算初始和最终极化程度的变化
            initial_pol = agent_evals[0].get("polarization")
            final_pol = agent_evals[-1].get("polarization")
            
            if initial_bs is not None and final_bs is not None:
                bs_change = final_bs - initial_bs
            else:
                bs_change = None
                
            if initial_pol is not None and final_pol is not None:
                pol_change = final_pol - initial_pol
            else:
                pol_change = None
            
            changes[agent_id] = {
                "bias_strength_change": bs_change,
                "polarization_change": pol_change,
                "initial_bias_strength": initial_bs,
                "final_bias_strength": final_bs,
                "initial_polarization": initial_pol,
                "final_polarization": final_pol
            }
        
        # 计算整体变化统计
        bs_changes = [
            change["bias_strength_change"] 
            for change in changes.values() 
            if change["bias_strength_change"] is not None
        ]
        
        pol_changes = [
            change["polarization_change"] 
            for change in changes.values() 
            if change["polarization_change"] is not None
        ]
        
        # 计算平均变化
        overall_stats = {
            "mean_bias_strength_change": np.mean(bs_changes) if bs_changes else None,
            "mean_polarization_change": np.mean(pol_changes) if pol_changes else None,
            "agent_changes": changes
        }
        
        return overall_stats
    
    @staticmethod
    def calculate_group_polarization(debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算群体极化效应
        
        Args:
            debate_result: 辩论实验结果
            
        Returns:
            Dict[str, Any]: 群体极化指标
        """
        # 获取辩论轮次和智能体
        rounds = debate_result.get("rounds", [])
        if len(rounds) < 2:  # 至少需要初始轮和一轮辩论
            return {"error": "辩论轮次不足"}
        
        agents = debate_result.get("agents", [])
        agent_ids = [agent["agent_id"] for agent in agents]
        
        # 使用偏见检测器的评估结果
        evaluations = debate_result.get("evaluations", {})
        if not evaluations:
            return {"error": "缺少偏见评估结果"}
        
        # 计算初始轮和最终轮的评估结果
        initial_evals = []
        final_evals = []
        
        for agent_id in agent_ids:
            agent_evals = evaluations.get(agent_id, [])
            if len(agent_evals) >= 2:
                initial_evals.append(agent_evals[0])
                final_evals.append(agent_evals[-1])
        
        if not initial_evals or not final_evals:
            return {"error": "无法获取完整的评估结果"}
        
        # 计算初始和最终的偏见强度和极化程度
        initial_bs = [
            e.get("bias_strength") for e in initial_evals 
            if e.get("bias_strength") is not None
        ]
        final_bs = [
            e.get("bias_strength") for e in final_evals 
            if e.get("bias_strength") is not None
        ]
        
        initial_pol = [
            e.get("polarization") for e in initial_evals 
            if e.get("polarization") is not None
        ]
        final_pol = [
            e.get("polarization") for e in final_evals 
            if e.get("polarization") is not None
        ]
        
        # 计算统计指标
        result = {
            "initial_bias_strength_mean": np.mean(initial_bs) if initial_bs else None,
            "final_bias_strength_mean": np.mean(final_bs) if final_bs else None,
            "initial_polarization_mean": np.mean(initial_pol) if initial_pol else None,
            "final_polarization_mean": np.mean(final_pol) if final_pol else None,
            
            "initial_bias_strength_std": np.std(initial_bs) if len(initial_bs) > 1 else 0,
            "final_bias_strength_std": np.std(final_bs) if len(final_bs) > 1 else 0,
            "initial_polarization_std": np.std(initial_pol) if len(initial_pol) > 1 else 0,
            "final_polarization_std": np.std(final_pol) if len(final_pol) > 1 else 0,
        }
        
        # 计算极化效应（最终值与初始值的差异）
        if result["initial_bias_strength_mean"] is not None and result["final_bias_strength_mean"] is not None:
            result["bias_strength_polarization"] = result["final_bias_strength_mean"] - result["initial_bias_strength_mean"]
        else:
            result["bias_strength_polarization"] = None
            
        if result["initial_polarization_mean"] is not None and result["final_polarization_mean"] is not None:
            result["opinion_polarization"] = result["final_polarization_mean"] - result["initial_polarization_mean"]
        else:
            result["opinion_polarization"] = None
            
        # 计算标准差变化（分歧程度的变化）
        if result["initial_bias_strength_std"] is not None and result["final_bias_strength_std"] is not None:
            result["bias_strength_std_change"] = result["final_bias_strength_std"] - result["initial_bias_strength_std"]
        else:
            result["bias_strength_std_change"] = None
            
        if result["initial_polarization_std"] is not None and result["final_polarization_std"] is not None:
            result["polarization_std_change"] = result["final_polarization_std"] - result["initial_polarization_std"]
        else:
            result["polarization_std_change"] = None
        
        return result 