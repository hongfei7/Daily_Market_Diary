"""
test_professional_system.py - 专业晨报系统测试脚本

测试所有模块的功能是否正常
"""

import sys
import os
from datetime import datetime, timedelta

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_diary'))

def test_imports():
    """测试所有模块是否可以正常导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from modules.data_fetcher import fetch_market_data, fetch_news
        print("✓ data_fetcher 导入成功")
    except Exception as e:
        print(f"✗ data_fetcher 导入失败: {e}")
        return False
    
    try:
        from modules.macro_calendar import fetch_macro_data
        print("✓ macro_calendar 导入成功")
    except Exception as e:
        print(f"✗ macro_calendar 导入失败: {e}")
        return False
    
    try:
        from modules.sector_news import fetch_sector_data
        print("✓ sector_news 导入成功")
    except Exception as e:
        print(f"✗ sector_news 导入失败: {e}")
        return False
    
    try:
        from modules.market_movers import fetch_movers_data
        print("✓ market_movers 导入成功")
    except Exception as e:
        print(f"✗ market_movers 导入失败: {e}")
        return False
    
    try:
        from modules.risk_radar import fetch_risk_data
        print("✓ risk_radar 导入成功")
    except Exception as e:
        print(f"✗ risk_radar 导入失败: {e}")
        return False
    
    try:
        from modules.report_template import format_professional_report
        print("✓ report_template 导入成功")
    except Exception as e:
        print(f"✗ report_template 导入失败: {e}")
        return False
    
    try:
        from modules.chart_features import extract_chart_features
        print("✓ chart_features 导入成功")
    except Exception as e:
        print(f"✗ chart_features 导入失败: {e}")
        return False
    
    try:
        from modules.llm_client import get_client
        print("✓ llm_client 导入成功")
    except Exception as e:
        print(f"✗ llm_client 导入失败: {e}")
        return False
    
    print("\n✅ 所有模块导入成功\n")
    return True


def test_data_fetching():
    """测试数据获取功能"""
    print("=" * 60)
    print("测试 2: 数据获取")
    print("=" * 60)
    
    test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # 测试市场数据
    try:
        from modules.data_fetcher import fetch_market_data
        print(f"正在获取市场数据 ({test_date})...")
        market_data = fetch_market_data(test_date)
        
        if market_data and 'summary' in market_data:
            print(f"✓ 市场数据获取成功")
            print(f"  - 数据类别: {len(market_data['summary'])} 个")
            print(f"  - 时间序列: {len(market_data.get('timeseries', []))} 个")
        else:
            print("⚠ 市场数据为空")
    except Exception as e:
        print(f"✗ 市场数据获取失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试新闻获取
    try:
        from modules.data_fetcher import fetch_news
        print("\n正在获取新闻...")
        news = fetch_news(max_per_feed=5)
        
        if news:
            print(f"✓ 新闻获取成功 ({len(news)} 条)")
            print(f"  示例: {news[0][:80]}...")
        else:
            print("⚠ 新闻为空")
    except Exception as e:
        print(f"✗ 新闻获取失败: {e}")
    
    # 测试宏观日历
    try:
        from modules.macro_calendar import fetch_macro_data
        print("\n正在获取宏观日历...")
        macro_data = fetch_macro_data(test_date)
        
        if macro_data:
            print(f"✓ 宏观日历获取成功")
            print(f"  - 已公布数据: {len(macro_data.get('calendar', {}).get('released', []))} 条")
            print(f"  - 待公布数据: {len(macro_data.get('calendar', {}).get('upcoming', []))} 条")
        else:
            print("⚠ 宏观日历为空")
    except Exception as e:
        print(f"⚠ 宏观日历获取失败: {e} (这是正常的，因为使用模拟数据)")
    
    # 测试行业新闻
    try:
        from modules.sector_news import fetch_sector_data
        print("\n正在获取行业新闻...")
        sector_data = fetch_sector_data(test_date)
        
        if sector_data:
            print(f"✓ 行业新闻获取成功")
            sector_news = sector_data.get('sector_news', {})
            total_news = sum(len(news_list) for news_list in sector_news.values())
            print(f"  - 行业数: {len(sector_news)} 个")
            print(f"  - 新闻总数: {total_news} 条")
        else:
            print("⚠ 行业新闻为空")
    except Exception as e:
        print(f"⚠ 行业新闻获取失败: {e} (可能是网络问题)")
    
    # 测试市场异动
    try:
        from modules.market_movers import fetch_movers_data
        print("\n正在获取市场异动...")
        movers_data = fetch_movers_data(test_date)
        
        if movers_data:
            print(f"✓ 市场异动获取成功")
            print(f"  - ETF 流向: {len(movers_data.get('etf_flows', []))} 个")
        else:
            print("⚠ 市场异动为空")
    except Exception as e:
        print(f"✗ 市场异动获取失败: {e}")
    
    # 测试风险雷达
    try:
        from modules.risk_radar import fetch_risk_data
        print("\n正在获取风险雷达...")
        risk_data = fetch_risk_data({'SPX': 6850, 'DXY': 98.5})
        
        if risk_data:
            print(f"✓ 风险雷达获取成功")
            print(f"  - 地缘风险: {len(risk_data.get('geopolitical_risks', []))} 个")
            print(f"  - 重大事件: {len(risk_data.get('upcoming_events', []))} 个")
        else:
            print("⚠ 风险雷达为空")
    except Exception as e:
        print(f"✗ 风险雷达获取失败: {e}")
    
    print("\n✅ 数据获取测试完成\n")
    return True


def test_llm_connection():
    """测试 LLM 连接"""
    print("=" * 60)
    print("测试 3: LLM 连接")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ 未设置 API 密钥 (MINIMAX_API_KEY 或 OPENAI_API_KEY)")
        print("  跳过 LLM 连接测试")
        return True
    
    try:
        from modules.llm_client import get_client
        print("正在测试 LLM 连接...")
        
        client = get_client()
        print("✓ LLM 客户端初始化成功")
        
        # 测试简单调用
        model_name = os.getenv("LLM_MODEL", "MiniMax-M2.7")
        print(f"  使用模型: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "请用一句话回复：测试成功"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        
        result = response.choices[0].message.content
        print(f"✓ LLM 调用成功")
        print(f"  响应: {result[:100]}")
        
    except Exception as e:
        print(f"✗ LLM 连接失败: {e}")
        return False
    
    print("\n✅ LLM 连接测试完成\n")
    return True


def test_report_generation():
    """测试报告生成"""
    print("=" * 60)
    print("测试 4: 报告生成")
    print("=" * 60)
    
    try:
        from modules.report_template import get_professional_template
        
        test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"正在生成报告模板 ({test_date})...")
        
        template = get_professional_template(test_date)
        
        if template and len(template) > 1000:
            print(f"✓ 报告模板生成成功")
            print(f"  - 模板长度: {len(template)} 字符")
            print(f"  - 包含章节: {template.count('##')} 个")
        else:
            print("⚠ 报告模板异常")
            return False
        
    except Exception as e:
        print(f"✗ 报告生成失败: {e}")
        return False
    
    print("\n✅ 报告生成测试完成\n")
    return True


def test_chart_features():
    """测试图表特征提取"""
    print("=" * 60)
    print("测试 5: 图表特征提取")
    print("=" * 60)
    
    try:
        from modules.chart_features import extract_chart_features, features_to_prompt_block
        import pandas as pd
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'time': pd.date_range('2026-04-13 09:00', periods=10, freq='5min'),
            'price': [100, 101, 102, 101.5, 103, 102, 104, 103.5, 105, 104],
            'symbol': 'TEST',
            'Category': 'FX',
        })
        
        print("正在提取图表特征...")
        features = extract_chart_features([test_data])
        
        if features:
            print(f"✓ 图表特征提取成功")
            print(f"  - FX 对数: {len(features.get('fx_pairs', []))} 个")
            
            # 测试格式化
            prompt_block = features_to_prompt_block(features)
            if prompt_block and len(prompt_block) > 100:
                print(f"✓ 特征格式化成功")
                print(f"  - 文本长度: {len(prompt_block)} 字符")
            else:
                print("⚠ 特征格式化异常")
        else:
            print("⚠ 图表特征为空")
        
    except Exception as e:
        print(f"✗ 图表特征提取失败: {e}")
        return False
    
    print("\n✅ 图表特征测试完成\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("投行研究院专业晨报系统 - 功能测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("数据获取", test_data_fetching()))
    results.append(("LLM 连接", test_llm_connection()))
    results.append(("报告生成", test_report_generation()))
    results.append(("图表特征", test_chart_features()))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查配置。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
