"""
main_professional.py — 投行研究院专业晨报生成器

整合所有模块，生成符合头部券商标准的 Morning Briefing
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

# 导入现有模块
from modules.data_fetcher import fetch_market_data, fetch_news
from modules.chart_features import extract_chart_features, features_to_prompt_block
from modules.llm_client import get_client

# 导入新增的专业模块
from modules.macro_calendar import fetch_macro_data
from modules.sector_news import fetch_sector_data
from modules.market_movers import fetch_movers_data
from modules.risk_radar import fetch_risk_data
from modules.report_template import (
    PROFESSIONAL_SYSTEM_PROMPT,
    format_professional_report,
    get_llm_prompt_for_professional_report,
)

# 导入图表生成（复用现有的）
from main import create_charts, _configure_console_output


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    default_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    parser = argparse.ArgumentParser(
        description="投行研究院专业晨报生成器 - Morning Briefing Generator"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=default_date,
        help="报告日期 (YYYY-MM-DD)，默认为昨天",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports_professional",
        help="输出目录，默认为 reports_professional",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="跳过图表生成（加快测试速度）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式，保存中间数据",
    )
    
    return parser.parse_args()


def fetch_all_data(report_date: str, debug: bool = False) -> Dict:
    """
    获取所有数据源
    
    Args:
        report_date: YYYY-MM-DD 格式
        debug: 是否保存调试数据
        
    Returns:
        包含所有数据的字典
    """
    print(f"\n{'='*60}")
    print(f"📊 投行研究院晨报数据采集 | {report_date}")
    print(f"{'='*60}\n")
    
    all_data = {}
    
    # 1. 市场数据（价格、图表）
    print("📈 [1/6] 获取市场数据...")
    try:
        market_data = fetch_market_data(report_date)
        all_data['market'] = market_data
        print(f"   ✓ 市场数据获取完成")
    except Exception as e:
        print(f"   ✗ 市场数据获取失败: {e}")
        all_data['market'] = {'summary': {}, 'timeseries': [], 'meta': {}}
    
    # 2. 宏观日历
    print("📅 [2/6] 获取宏观经济日历...")
    try:
        macro_data = fetch_macro_data(report_date)
        all_data['macro'] = macro_data
        print(f"   ✓ 宏观日历获取完成")
    except Exception as e:
        print(f"   ✗ 宏观日历获取失败: {e}")
        all_data['macro'] = {'calendar': {}, 'central_bank_events': []}
    
    # 3. 行业新闻
    print("📰 [3/6] 获取行业与个股新闻...")
    try:
        sector_data = fetch_sector_data(report_date)
        all_data['sector'] = sector_data
        print(f"   ✓ 行业新闻获取完成")
    except Exception as e:
        print(f"   ✗ 行业新闻获取失败: {e}")
        all_data['sector'] = {'sector_news': {}, 'earnings_calendar': [], 'analyst_changes': []}
    
    # 4. 市场异动
    print("💹 [4/6] 获取盘前异动与资金流向...")
    try:
        movers_data = fetch_movers_data(report_date)
        all_data['movers'] = movers_data
        print(f"   ✓ 市场异动获取完成")
    except Exception as e:
        print(f"   ✗ 市场异动获取失败: {e}")
        all_data['movers'] = {'premarket_movers': {}, 'etf_flows': []}
    
    # 5. 风险监控
    print("⚠️  [5/6] 获取风险雷达数据...")
    try:
        # 从市场数据中提取当前价格
        current_prices = _extract_current_prices(all_data.get('market', {}))
        risk_data = fetch_risk_data(current_prices)
        all_data['risk'] = risk_data
        print(f"   ✓ 风险雷达获取完成")
    except Exception as e:
        print(f"   ✗ 风险雷达获取失败: {e}")
        all_data['risk'] = {'geopolitical_risks': [], 'upcoming_events': []}
    
    # 6. 新闻标题
    print("📡 [6/6] 获取新闻标题...")
    try:
        news_headlines = fetch_news(max_per_feed=10)
        all_data['news'] = news_headlines
        print(f"   ✓ 新闻标题获取完成 ({len(news_headlines)} 条)")
    except Exception as e:
        print(f"   ✗ 新闻标题获取失败: {e}")
        all_data['news'] = []
    
    # 保存调试数据
    if debug:
        debug_file = f"debug_data_{report_date}.json"
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 调试数据已保存: {debug_file}")
        except Exception as e:
            print(f"\n⚠️  调试数据保存失败: {e}")
    
    print(f"\n{'='*60}")
    print("✅ 数据采集完成")
    print(f"{'='*60}\n")
    
    return all_data


def _extract_current_prices(market_data: Dict) -> Dict:
    """从市场数据中提取当前价格"""
    prices = {}
    
    summary = market_data.get('summary', {})
    
    # 提取主要指数价格
    if 'Equities' in summary:
        equities = summary['Equities']
        if 'S&P 500' in equities and isinstance(equities['S&P 500'], dict):
            prices['SPX'] = equities['S&P 500'].get('Price', 0)
        if 'Nasdaq 100' in equities and isinstance(equities['Nasdaq 100'], dict):
            prices['NDX'] = equities['Nasdaq 100'].get('Price', 0)
    
    # 提取外汇
    if 'FX' in summary:
        fx = summary['FX']
        if 'DXY' in fx and isinstance(fx['DXY'], dict):
            prices['DXY'] = fx['DXY'].get('Price', 0)
    
    # 提取利率
    if 'Rates' in summary:
        rates = summary['Rates']
        if '10Y Treasury' in rates and isinstance(rates['10Y Treasury'], dict):
            prices['US10Y'] = rates['10Y Treasury'].get('Price', 0)
    
    return prices


def generate_llm_analysis(
    report_date: str,
    all_data: Dict,
    chart_features_block: str,
) -> str:
    """
    使用 LLM 生成专业分析
    
    Args:
        report_date: 报告日期
        all_data: 所有采集的数据
        chart_features_block: 图表特征文本
        
    Returns:
        LLM 生成的分析文本
    """
    print("🤖 正在生成 AI 分析...")
    
    max_retries = 3
    retry_delay = 5  # 初始延迟5秒
    
    for attempt in range(max_retries):
        try:
            client = get_client()
            
            # 构建 prompt
            user_prompt = get_llm_prompt_for_professional_report(
                date=report_date,
                market_summary=all_data.get('market', {}).get('summary', {}),
                chart_features=chart_features_block,
                news_headlines=all_data.get('news', []),
                macro_calendar=all_data.get('macro', {}),
            )
            
            # 调用 LLM
            model_name = os.getenv("LLM_MODEL", "MiniMax-M2.7")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": PROFESSIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            
            analysis = response.choices[0].message.content
            print("   ✓ AI 分析生成完成")
            
            return analysis
            
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是服务器负载问题
            if '529' in error_msg or 'overloaded' in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"   ⚠️ 服务器负载高，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"   ✗ AI 分析生成失败（服务器负载过高）: {e}")
                    return f"*AI 分析暂时不可用（服务器负载过高），请稍后重试。错误: {error_msg[:200]}*"
            else:
                print(f"   ✗ AI 分析生成失败: {e}")
                return f"*AI 分析生成失败: {error_msg[:200]}*"
    
    return "*AI 分析生成失败: 达到最大重试次数*"


def main():
    """主函数"""
    _configure_console_output()
    args = parse_args()
    
    report_date = args.date
    output_dir = args.output_dir
    
    print(f"\n{'#'*60}")
    print(f"#  投行研究院晨报生成器")
    print(f"#  Morning Briefing Generator")
    print(f"#")
    print(f"#  报告日期: {report_date}")
    print(f"#  输出目录: {output_dir}")
    print(f"{'#'*60}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据采集
    all_data = fetch_all_data(report_date, debug=args.debug)
    
    # 2. 生成图表
    charts_section = ""
    if not args.skip_charts:
        print("📊 正在生成图表...")
        try:
            market_data = all_data.get('market', {})
            charts_section = create_charts(
                report_date=report_date,
                market_data_dict=market_data,
                output_dir=output_dir,
                chart_label=report_date,
            )
            print("   ✓ 图表生成完成")
        except Exception as e:
            print(f"   ✗ 图表生成失败: {e}")
            charts_section = "\n*(图表生成失败)*\n"
    else:
        print("⏭️  跳过图表生成")
        charts_section = "\n*(图表已跳过)*\n"
    
    # 3. 提取图表特征
    print("🔬 正在提取图表特征...")
    try:
        timeseries_data = all_data.get('market', {}).get('timeseries', [])
        chart_features = extract_chart_features(timeseries_data, tz="Asia/Shanghai")
        chart_features_block = features_to_prompt_block(chart_features)
        
        # 保存图表特征
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        feat_path = os.path.join(chart_dir, f"features_{report_date}.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(chart_features, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"   ✓ 图表特征提取完成")
    except Exception as e:
        print(f"   ✗ 图表特征提取失败: {e}")
        chart_features_block = "[图表特征提取失败]"
    
    # 4. 生成 LLM 分析
    llm_analysis = generate_llm_analysis(report_date, all_data, chart_features_block)
    
    # 5. 组装专业晨报
    print("📝 正在组装晨报...")
    try:
        final_report = format_professional_report(
            date=report_date,
            market_data=all_data.get('market', {}),
            macro_data=all_data.get('macro', {}),
            sector_data=all_data.get('sector', {}),
            movers_data=all_data.get('movers', {}),
            risk_data=all_data.get('risk', {}),
            llm_analysis=llm_analysis,
            charts_section=charts_section,
        )
        print("   ✓ 晨报组装完成")
    except Exception as e:
        print(f"   ✗ 晨报组装失败: {e}")
        sys.exit(1)
    
    # 6. 保存报告
    output_file = os.path.join(output_dir, f"{report_date}_morning_briefing.md")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\n{'='*60}")
        print(f"✅ 晨报生成成功！")
        print(f"📄 报告路径: {output_file}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n❌ 报告保存失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
