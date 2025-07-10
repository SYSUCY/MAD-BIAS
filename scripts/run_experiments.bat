@echo off
echo MAD-bias 实验批处理脚本
echo ===========================

REM 设置时间戳
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"
echo 实验开始时间: %timestamp%

REM 创建日志目录
if not exist logs\experiments mkdir logs\experiments

REM 1. 单智能体基线实验 - 所有模型，多个话题
echo 步骤 1/5: 运行单智能体基线实验 (所有模型)
python run.py --experiment baseline --models gpt-4.1-nano deepseek-chat --topic_count 10 > logs\experiments\baseline_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo 单智能体实验失败，请检查日志
    exit /b %errorlevel%
)
echo 单智能体基线实验完成

REM 2. 多智能体同质化实验 - 2个智能体，不同轮次
echo 步骤 2/5: 运行多智能体同质化实验 (2个智能体)
python run.py --experiment debate --agents 2 --rounds 1 3 5 --topic_count 8 > logs\experiments\debate_2agents_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo 2智能体实验失败，请检查日志
    exit /b %errorlevel%
)
echo 2智能体同质化实验完成

REM 3. 多智能体同质化实验 - 3个智能体，不同轮次
echo 步骤 3/5: 运行多智能体同质化实验 (3个智能体)
python run.py --experiment debate --agents 3 --rounds 1 3 5 --topic_count 8 > logs\experiments\debate_3agents_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo 3智能体实验失败，请检查日志
    exit /b %errorlevel%
)
echo 3智能体同质化实验完成

REM 4. 多智能体异质化实验
echo 步骤 4/5: 运行多智能体异质化实验
python run.py --experiment debate --agents 2 3 --rounds 3 5 --topic_count 6 --models gpt-4.1-nano deepseek-chat > logs\experiments\debate_hetero_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo 异质化实验失败，请检查日志
    exit /b %errorlevel%
)
echo 异质化实验完成

REM 5. 生成可视化和报告
echo 步骤 5/5: 生成可视化和报告
python scripts/generate_all.py --open_report > logs\experiments\visualization_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo 可视化生成失败，请检查日志
    exit /b %errorlevel%
)

echo ===========================
echo 所有实验完成！
echo 请查看生成的报告和可视化结果
echo 日志文件保存在 logs\experiments 目录下 