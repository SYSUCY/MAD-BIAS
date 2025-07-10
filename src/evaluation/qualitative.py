# -*- coding: utf-8 -*-
"""
定性分析模块
用于进行内容分析和文本特征提取
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import Counter
import nltk
import pandas as pd

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from config import settings

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """
    内容分析器
    分析回复内容的文本特征和语言模式
    """
    
    def __init__(self):
        """初始化内容分析器"""
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon')
            self.sia = SentimentIntensityAnalyzer()
            
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本内容
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        if not text:
            return {
                "error": "空文本"
            }
        
        result = {}
        
        # 1. 基础文本统计
        result["text_length"] = len(text)
        
        # 分句和分词
        sentences = sent_tokenize(text)
        result["sentence_count"] = len(sentences)
        
        words = word_tokenize(text)
        result["word_count"] = len(words)
        
        if result["sentence_count"] > 0:
            result["avg_sentence_length"] = result["word_count"] / result["sentence_count"]
        else:
            result["avg_sentence_length"] = 0
            
        # 2. 情感分析
        sentiment = self.sia.polarity_scores(text)
        result["sentiment"] = {
            "negative": sentiment["neg"],
            "neutral": sentiment["neu"],
            "positive": sentiment["pos"],
            "compound": sentiment["compound"]
        }
        
        # 3. 词频分析（去除停用词）
        words_no_stop = [word.lower() for word in words if word.isalpha() and word.lower() not in self.stop_words]
        word_freq = Counter(words_no_stop)
        result["top_words"] = dict(word_freq.most_common(20))
        
        # 4. 句式分析
        question_sentences = len([s for s in sentences if s.strip().endswith('?')])
        exclamation_sentences = len([s for s in sentences if s.strip().endswith('!')])
        
        result["sentence_types"] = {
            "question": question_sentences,
            "exclamation": exclamation_sentences,
            "statement": result["sentence_count"] - question_sentences - exclamation_sentences
        }
        
        return result
    
    def extract_argument_features(self, text: str) -> Dict[str, Any]:
        """
        提取论证特征
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict[str, Any]: 论证特征
        """
        result = {}
        
        # 1. 论证指标词汇
        reasoning_indicators = [
            "because", "since", "therefore", "thus", "hence", "consequently",
            "as a result", "so", "it follows that", "for this reason",
            "due to", "given that", "suggests that", "indicates that"
        ]
        
        counter_indicators = [
            "however", "but", "nevertheless", "although", "though", "despite",
            "on the other hand", "conversely", "in contrast", "alternatively",
            "contrary to", "yet", "while", "whereas"
        ]
        
        evidence_indicators = [
            "research", "study", "evidence", "data", "according to", "statistic",
            "survey", "experiment", "finding", "report", "analysis", "expert",
            "demonstrate", "show", "prove", "confirm", "verify", "document"
        ]
        
        opinion_indicators = [
            "believe", "think", "feel", "opinion", "view", "perspective",
            "suggest", "consider", "argue", "claim", "assert", "maintain",
            "contend", "insist", "propose", "hold that", "in my opinion"
        ]
        
        # 计数各类指标词汇
        text_lower = text.lower()
        
        reasoning_count = sum(text_lower.count(" " + indicator + " ") for indicator in reasoning_indicators)
        counter_count = sum(text_lower.count(" " + indicator + " ") for indicator in counter_indicators)
        evidence_count = sum(text_lower.count(" " + indicator + " ") for indicator in evidence_indicators)
        opinion_count = sum(text_lower.count(" " + indicator + " ") for indicator in opinion_indicators)
        
        result["argument_indicators"] = {
            "reasoning": reasoning_count,
            "counter_argument": counter_count,
            "evidence": evidence_count,
            "opinion": opinion_count
        }
        
        # 2. 论证结构特征
        sentences = sent_tokenize(text)
        
        # 检测条件语句 (if-then)
        condition_sentences = sum(1 for s in sentences if re.search(r'\bif\b.*?\bthen\b', s.lower()))
        
        # 检测比较语句
        comparison_sentences = sum(1 for s in sentences if re.search(r'\bmore\b|\bless\b|\bbetter\b|\bworse\b|\bgreater\b|\bfewer\b', s.lower()))
        
        result["argument_structures"] = {
            "conditional": condition_sentences,
            "comparative": comparison_sentences
        }
        
        # 3. 认知复杂度估计（简单指标）
        long_sentences = sum(1 for s in sentences if len(word_tokenize(s)) > 20)
        complex_words = sum(1 for w in word_tokenize(text) if len(w) > 8)
        
        if result.get("sentence_count", 0) > 0:
            result["cognitive_complexity"] = {
                "long_sentence_ratio": long_sentences / len(sentences),
                "complex_word_ratio": complex_words / len(word_tokenize(text)) if len(word_tokenize(text)) > 0 else 0
            }
        else:
            result["cognitive_complexity"] = {
                "long_sentence_ratio": 0,
                "complex_word_ratio": 0
            }
        
        return result

class BiasLanguageDetector:
    """
    偏见语言检测器
    识别文本中可能表现偏见的语言特征
    """
    
    def __init__(self):
        """初始化偏见语言检测器"""
        # 偏见表达模式
        self.bias_patterns = {
            # 过度概括
            "generalization": [
                r"\ball\b.*?\bare\b", r"\bevery\b.*?\bis\b", r"\balways\b",
                r"\bnever\b", r"\bno one\b", r"\beveryone\b"
            ],
            
            # 情绪性语言
            "emotional": [
                r"\bterrible\b", r"\bawful\b", r"\bhorrible\b", r"\bdeplorable\b",
                r"\bwonderful\b", r"\bexcellent\b", r"\boutstanding\b", r"\bperfect\b",
                r"\bdisgust(ing)?\b", r"\bshock(ing)?\b", r"\bappall(ing)?\b"
            ],
            
            # 绝对化表达
            "absolutist": [
                r"\babsolutely\b", r"\bcompletely\b", r"\btotally\b",
                r"\bentirely\b", r"\bwithout\s+doubt\b", r"\bundeniably\b"
            ],
            
            # 标签化语言
            "labeling": [
                r"\bradical\b", r"\bextremist\b", r"\bfanatic\b",
                r"\bliberal\b", r"\bconservative\b", r"\bleftist\b", r"\brightist\b"
            ]
        }
    
    def detect_bias_language(self, text: str) -> Dict[str, Any]:
        """
        检测文本中的偏见语言
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict[str, Any]: 偏见语言分析结果
        """
        result = {
            "bias_language_count": {k: 0 for k in self.bias_patterns}
        }
        
        # 检测各类偏见表达
        text_lower = text.lower()
        
        for category, patterns in self.bias_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.finditer(pattern, text_lower))
            
            # 去重（可能有重叠）
            unique_matches = set()
            for match in matches:
                unique_matches.add(match.group())
            
            result["bias_language_count"][category] = len(unique_matches)
            
            # 存储匹配到的表达
            if unique_matches:
                if "bias_language_examples" not in result:
                    result["bias_language_examples"] = {}
                result["bias_language_examples"][category] = list(unique_matches)
        
        # 计算总偏见表达数量
        result["total_bias_expressions"] = sum(result["bias_language_count"].values())
        
        # 计算每千词偏见表达密度
        word_count = len(word_tokenize(text))
        if word_count > 0:
            result["bias_expressions_per_1k_words"] = (result["total_bias_expressions"] / word_count) * 1000
        else:
            result["bias_expressions_per_1k_words"] = 0
        
        return result 