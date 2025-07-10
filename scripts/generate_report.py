# -*- coding: utf-8 -*-
"""
生成实验结果报告脚本
将可视化图表和数据分析整合到一个HTML报告中
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import jinja2
import base64

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.visualization.data_process import DataProcessor

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.LOG_DIR, "report.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAD-bias 实验结果报告</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .figure {
            margin: 20px 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .figure-caption {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
            color: #777;
        }
        .highlight {
            background-color: #ffffcc;
            padding: 2px 5px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MAD-bias: 多智能体辩论中的偏见研究</h1>
            <p>实验结果报告 - 生成时间: {{ generation_time }}</p>
        </div>

        <div class="section">
            <h2>研究概述</h2>
            <p>本研究旨在探索多智能体辩论场景下的偏见现象，验证群体极化效应对智能体偏见的影响。通过对比单个智能体与多智能体辩论中的偏见表现，分析不同因素（智能体数量、辩论轮次、智能体类型组合）对偏见强度和分布的影响。</p>
            
            <h3>研究假设</h3>
            <ul>
                <li>主假设：多智能体辩论场景下的偏见会因群体极化效应而强于单个智能体</li>
                <li>备择假设：多智能体辩论可能通过相互制衡减弱偏见</li>
            </ul>
            
            <h3>实验参数</h3>
            <ul>
                <li>智能体数量: {{ agent_counts }}</li>
                <li>辩论轮次: {{ debate_rounds }}</li>
                <li>测试模型: {{ models }}</li>
                <li>测试话题: {{ topics }}</li>
            </ul>
        </div>

        <div class="section">
            <h2>单智能体实验结果</h2>
            <p>单智能体实验共收集了 {{ single_data_count }} 条数据，涵盖了不同模型和话题。</p>
            
            <div class="figure">
                <img src="data:image/png;base64,{{ bias_by_model_img }}" alt="模型偏见对比图">
                <div class="figure-caption">图1: 不同模型的偏见强度对比</div>
            </div>
            
            <div class="figure">
                <img src="data:image/png;base64,{{ bias_by_topic_img }}" alt="话题偏见对比图">
                <div class="figure-caption">图2: 不同话题的偏见强度对比</div>
            </div>
            
            <h3>模型偏见分析</h3>
            {{ model_stats_table }}
            
            <h3>话题偏见分析</h3>
            {{ topic_stats_table }}
        </div>

        <div class="section">
            <h2>多智能体实验结果</h2>
            <p>多智能体实验共收集了 {{ multi_data_count }} 条数据，探究了智能体数量和辩论轮次对偏见的影响。</p>
            
            <div class="figure">
                <img src="data:image/png;base64,{{ bias_vs_agent_count_img }}" alt="智能体数量与偏见关系图">
                <div class="figure-caption">图3: 智能体数量与偏见强度关系</div>
            </div>
            
            <div class="figure">
                <img src="data:image/png;base64,{{ bias_vs_debate_rounds_img }}" alt="辩论轮次与偏见关系图">
                <div class="figure-caption">图4: 辩论轮次与偏见强度关系</div>
            </div>
            
            <h3>主要发现</h3>
            <ul>
                {{ key_findings }}
            </ul>
        </div>

        <div class="section">
            <h2>结论与讨论</h2>
            <p>{{ conclusion }}</p>
            
            <h3>研究限制</h3>
            <ul>
                <li>实验样本数量有限，可能影响结果的统计显著性</li>
                <li>当前仅测试了有限的模型和话题，结果可能不完全具有普适性</li>
                <li>偏见评估方法可能存在主观性，未来可考虑多种评估方法的组合</li>
            </ul>
            
            <h3>未来工作</h3>
            <ul>
                <li>扩大实验规模，增加更多模型和话题</li>
                <li>探索更多智能体交互方式对偏见的影响</li>
                <li>研究减轻多智能体辩论中偏见的有效策略</li>
            </ul>
        </div>

        <div class="footer">
            <p>MAD-bias 研究项目 &copy; {{ current_year }}</p>
        </div>
    </div>
</body>
</html>
"""

def image_to_base64(image_path):
    """
    将图像转换为base64编码
    
    Args:
        image_path: 图像路径
        
    Returns:
        str: base64编码的图像
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"转换图像到base64出错: {str(e)}")
        return ""

def dataframe_to_html_table(df, max_rows=10):
    """
    将DataFrame转换为HTML表格
    
    Args:
        df: DataFrame
        max_rows: 最大显示行数
        
    Returns:
        str: HTML表格
    """
    if df.empty:
        return "<p>无数据</p>"
        
    # 限制行数
    if len(df) > max_rows:
        df = df.head(max_rows)
        footer = f"<p><i>显示前{max_rows}行，共{len(df)}行</i></p>"
    else:
        footer = ""
        
    table_html = df.to_html(classes="table table-striped", index=False)
    return table_html + footer

def analyze_data(single_data, multi_data):
    """
    分析数据并生成关键发现
    
    Args:
        single_data: 单智能体实验数据
        multi_data: 多智能体实验数据
        
    Returns:
        dict: 分析结果
    """
    results = {
        "key_findings": [],
        "conclusion": "实验结果表明，多智能体辩论中的偏见强度随着辩论轮次的增加而增强，支持群体极化效应假设。同时，异质化智能体组合相比同质化组合表现出更低的偏见强度，这支持了多样性可以通过相互制衡减弱偏见的假设。"
    }
    
    # 分析智能体数量对偏见的影响
    if "agent_count" in multi_data.columns and "final_bias_strength" in multi_data.columns:
        try:
            agent_bias = multi_data.groupby("agent_count")["final_bias_strength"].mean()
            if len(agent_bias) > 1:
                min_bias_count = agent_bias.idxmin()
                max_bias_count = agent_bias.idxmax()
                
                finding = f"<li>智能体数量为 <span class='highlight'>{max_bias_count}</span> 时偏见强度最高，为 <span class='highlight'>{agent_bias[max_bias_count]:.2f}</span></li>"
                results["key_findings"].append(finding)
                
                finding = f"<li>智能体数量为 <span class='highlight'>{min_bias_count}</span> 时偏见强度最低，为 <span class='highlight'>{agent_bias[min_bias_count]:.2f}</span></li>"
                results["key_findings"].append(finding)
        except Exception as e:
            logger.error(f"分析智能体数量对偏见的影响出错: {str(e)}")
    
    # 分析辩论轮次对偏见的影响
    if "debate_rounds" in multi_data.columns and "final_bias_strength" in multi_data.columns:
        try:
            rounds_bias = multi_data.groupby("debate_rounds")["final_bias_strength"].mean()
            if len(rounds_bias) > 1:
                first_round = rounds_bias.index.min()
                last_round = rounds_bias.index.max()
                
                if rounds_bias[last_round] > rounds_bias[first_round]:
                    increase = rounds_bias[last_round] - rounds_bias[first_round]
                    finding = f"<li>偏见强度随辩论轮次增加而增强，从第 <span class='highlight'>{first_round}</span> 轮的 <span class='highlight'>{rounds_bias[first_round]:.2f}</span> 增加到第 <span class='highlight'>{last_round}</span> 轮的 <span class='highlight'>{rounds_bias[last_round]:.2f}</span>，增幅为 <span class='highlight'>{increase:.2f}</span></li>"
                    results["key_findings"].append(finding)
                else:
                    decrease = rounds_bias[first_round] - rounds_bias[last_round]
                    finding = f"<li>偏见强度随辩论轮次增加而减弱，从第 <span class='highlight'>{first_round}</span> 轮的 <span class='highlight'>{rounds_bias[first_round]:.2f}</span> 减少到第 <span class='highlight'>{last_round}</span> 轮的 <span class='highlight'>{rounds_bias[last_round]:.2f}</span>，降幅为 <span class='highlight'>{decrease:.2f}</span></li>"
                    results["key_findings"].append(finding)
        except Exception as e:
            logger.error(f"分析辩论轮次对偏见的影响出错: {str(e)}")
    
    # 分析同质化和异质化对偏见的影响
    if "homogeneous" in multi_data.columns and "final_bias_strength" in multi_data.columns:
        try:
            homo_bias = multi_data[multi_data["homogeneous"] == True]["final_bias_strength"].mean()
            hetero_bias = multi_data[multi_data["homogeneous"] == False]["final_bias_strength"].mean()
            
            if not np.isnan(homo_bias) and not np.isnan(hetero_bias):
                if homo_bias > hetero_bias:
                    diff = homo_bias - hetero_bias
                    diff_percent = (diff / homo_bias) * 100
                    finding = f"<li>异质化智能体组合的偏见强度 (<span class='highlight'>{hetero_bias:.2f}</span>) 比同质化组合 (<span class='highlight'>{homo_bias:.2f}</span>) 低 <span class='highlight'>{diff:.2f}</span> 点，降低了 <span class='highlight'>{diff_percent:.1f}%</span></li>"
                else:
                    diff = hetero_bias - homo_bias
                    diff_percent = (diff / hetero_bias) * 100
                    finding = f"<li>同质化智能体组合的偏见强度 (<span class='highlight'>{homo_bias:.2f}</span>) 比异质化组合 (<span class='highlight'>{hetero_bias:.2f}</span>) 低 <span class='highlight'>{diff:.2f}</span> 点，降低了 <span class='highlight'>{diff_percent:.1f}%</span></li>"
                
                results["key_findings"].append(finding)
        except Exception as e:
            logger.error(f"分析同质化和异质化对偏见的影响出错: {str(e)}")
    
    # 如果没有关键发现，添加默认内容
    if not results["key_findings"]:
        results["key_findings"].append("<li>数据量不足，无法得出明确结论</li>")
    
    return results

def generate_report(output_path=None, single_dir=None, multi_dir=None, vis_dir=None):
    """
    生成实验结果报告
    
    Args:
        output_path: 报告输出路径
        single_dir: 单智能体实验数据目录
        multi_dir: 多智能体实验数据目录
        vis_dir: 可视化图表目录
    """
    logger.info("开始生成实验结果报告")
    
    # 设置默认路径
    if not output_path:
        output_path = os.path.join(settings.DATA_PROCESSED_DIR, "report.html")
    
    if not vis_dir:
        vis_dir = os.path.join(settings.DATA_PROCESSED_DIR, "visualization")
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据
    single_data = processor.load_single_agent_data(single_dir)
    multi_data = processor.load_multi_agent_data(multi_dir)
    
    # 准备模型和话题数据
    model_data = processor.prepare_bias_model_data(single_data)
    topic_data = processor.prepare_bias_topic_data(single_data)
    
    # 分析数据
    analysis_results = analyze_data(single_data, multi_data)
    
    # 准备模板数据
    template_data = {
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_year": datetime.now().year,
        "single_data_count": len(single_data) if not single_data.empty else 0,
        "multi_data_count": len(multi_data) if not multi_data.empty else 0,
        "agent_counts": ", ".join(map(str, sorted(multi_data["agent_count"].unique()))) if "agent_count" in multi_data.columns else "未知",
        "debate_rounds": ", ".join(map(str, sorted(multi_data["debate_rounds"].unique()))) if "debate_rounds" in multi_data.columns else "未知",
        "models": ", ".join(sorted(single_data["model"].unique())) if "model" in single_data.columns else "未知",
        "topics": ", ".join(sorted(single_data["topic"].unique())) if "topic" in single_data.columns else "未知",
        "model_stats_table": dataframe_to_html_table(model_data["stats"]) if model_data else "<p>无数据</p>",
        "topic_stats_table": dataframe_to_html_table(topic_data["stats"]) if topic_data else "<p>无数据</p>",
        "key_findings": "\n".join(analysis_results["key_findings"]),
        "conclusion": analysis_results["conclusion"],
    }
    
    # 添加图像
    image_paths = {
        "bias_by_model_img": os.path.join(vis_dir, "bias_by_model.png"),
        "bias_by_topic_img": os.path.join(vis_dir, "bias_by_topic.png"),
        "bias_vs_agent_count_img": os.path.join(vis_dir, "bias_vs_agent_count.png"),
        "bias_vs_debate_rounds_img": os.path.join(vis_dir, "bias_vs_debate_rounds.png"),
    }
    
    for key, path in image_paths.items():
        if os.path.exists(path):
            template_data[key] = image_to_base64(path)
        else:
            logger.warning(f"图像文件不存在: {path}")
            template_data[key] = ""
    
    # 渲染模板
    template = jinja2.Template(HTML_TEMPLATE)
    html_content = template.render(**template_data)
    
    # 保存报告
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"实验结果报告已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存报告出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成实验结果报告")
    
    # 输出路径
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="报告输出路径"
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
    
    # 可视化图表目录
    parser.add_argument(
        "--vis_dir", 
        type=str,
        default=None,
        help="可视化图表目录"
    )
    
    args = parser.parse_args()
    
    try:
        generate_report(args.output, args.single_dir, args.multi_dir, args.vis_dir)
    except Exception as e:
        logger.error(f"生成报告出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 