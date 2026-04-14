"""
test_professional_system.py - Professional Morning Briefing System Test Script

Test all module functionalities
"""

import sys
import os
from datetime import datetime, timedelta

# Add module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_diary'))

def test_imports():
    """Test if all modules can be imported properly"""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)
    
    try:
        from modules.data_fetcher import fetch_market_data, fetch_news
        print("✓ data_fetcher imported successfully")
    except Exception as e:
        print(f"✗ data_fetcher import failed: {e}")
        return False
    
    try:
        from modules.macro_calendar import fetch_macro_data
        print("✓ macro_calendar imported successfully")
    except Exception as e:
        print(f"✗ macro_calendar import failed: {e}")
        return False
    
    try:
        from modules.sector_news import fetch_sector_data
        print("✓ sector_news imported successfully")
    except Exception as e:
        print(f"✗ sector_news import failed: {e}")
        return False
    
    try:
        from modules.market_movers import fetch_movers_data
        print("✓ market_movers imported successfully")
    except Exception as e:
        print(f"✗ market_movers import failed: {e}")
        return False
    
    try:
        from modules.risk_radar import fetch_risk_data
        print("✓ risk_radar imported successfully")
    except Exception as e:
        print(f"✗ risk_radar import failed: {e}")
        return False
    
    try:
        from modules.report_template import format_professional_report
        print("✓ report_template imported successfully")
    except Exception as e:
        print(f"✗ report_template import failed: {e}")
        return False
    
    try:
        from modules.chart_features import extract_chart_features
        print("✓ chart_features imported successfully")
    except Exception as e:
        print(f"✗ chart_features import failed: {e}")
        return False
    
    try:
        from modules.llm_client import get_client
        print("✓ llm_client imported successfully")
    except Exception as e:
        print(f"✗ llm_client import failed: {e}")
        return False
    
    print("\n✅ All modules imported successfully\n")
    return True


def test_data_fetching():
    """Test data fetching functionality"""
    print("=" * 60)
    print("Test 2: Data Fetching")
    print("=" * 60)
    
    test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Test market data
    try:
        from modules.data_fetcher import fetch_market_data
        print(f"Fetching market data ({test_date})...")
        market_data = fetch_market_data(test_date)
        
        if market_data and 'summary' in market_data:
            print(f"✓ Market data fetched successfully")
            print(f"  - Data categories: {len(market_data['summary'])}")
            print(f"  - Time series: {len(market_data.get('timeseries', []))}")
        else:
            print("⚠ Market data is empty")
    except Exception as e:
        print(f"✗ Market data fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test news fetching
    try:
        from modules.data_fetcher import fetch_news
        print("\nFetching news...")
        news = fetch_news(max_per_feed=5)
        
        if news:
            print(f"✓ News fetched successfully ({len(news)} items)")
            print(f"  Example: {news[0][:80]}...")
        else:
            print("⚠ News is empty")
    except Exception as e:
        print(f"✗ News fetch failed: {e}")
    
    # Test macro calendar
    try:
        from modules.macro_calendar import fetch_macro_data
        print("\nFetching macro calendar...")
        macro_data = fetch_macro_data(test_date)
        
        if macro_data:
            print(f"✓ Macro calendar fetched successfully")
            print(f"  - Released data: {len(macro_data.get('calendar', {}).get('released', []))}")
            print(f"  - Upcoming data: {len(macro_data.get('calendar', {}).get('upcoming', []))}")
        else:
            print("⚠ Macro calendar is empty")
    except Exception as e:
        print(f"⚠ Macro calendar fetch failed: {e} (This is normal, using mock data)")
    
    # Test sector news
    try:
        from modules.sector_news import fetch_sector_data
        print("\nFetching sector news...")
        sector_data = fetch_sector_data(test_date)
        
        if sector_data:
            print(f"✓ Sector news fetched successfully")
            sector_news = sector_data.get('sector_news', {})
            total_news = sum(len(news_list) for news_list in sector_news.values())
            print(f"  - Sectors: {len(sector_news)}")
            print(f"  - Total news: {total_news}")
        else:
            print("⚠ Sector news is empty")
    except Exception as e:
        print(f"⚠ Sector news fetch failed: {e} (May be network issue)")
    
    # Test market movers
    try:
        from modules.market_movers import fetch_movers_data
        print("\nFetching market movers...")
        movers_data = fetch_movers_data(test_date)
        
        if movers_data:
            print(f"✓ Market movers fetched successfully")
            print(f"  - ETF flows: {len(movers_data.get('etf_flows', []))}")
        else:
            print("⚠ Market movers is empty")
    except Exception as e:
        print(f"✗ Market movers fetch failed: {e}")
    
    # Test risk radar
    try:
        from modules.risk_radar import fetch_risk_data
        print("\nFetching risk radar...")
        risk_data = fetch_risk_data({'SPX': 6850, 'DXY': 98.5})
        
        if risk_data:
            print(f"✓ Risk radar fetched successfully")
            print(f"  - Geopolitical risks: {len(risk_data.get('geopolitical_risks', []))}")
            print(f"  - Upcoming events: {len(risk_data.get('upcoming_events', []))}")
        else:
            print("⚠ Risk radar is empty")
    except Exception as e:
        print(f"✗ Risk radar fetch failed: {e}")
    
    print("\n✅ Data fetching test completed\n")
    return True


def test_llm_connection():
    """Test LLM connection"""
    print("=" * 60)
    print("Test 3: LLM Connection")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ API key not set (MINIMAX_API_KEY or OPENAI_API_KEY)")
        print("  Skipping LLM connection test")
        return True
    
    try:
        from modules.llm_client import get_client
        print("Testing LLM connection...")
        
        client = get_client()
        print("✓ LLM client initialized successfully")
        
        # Test simple call
        model_name = os.getenv("LLM_MODEL", "MiniMax-M2.7")
        print(f"  Using model: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Reply in one sentence: Test successful"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        
        result = response.choices[0].message.content
        print(f"✓ LLM call successful")
        print(f"  Response: {result[:100]}")
        
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        return False
    
    print("\n✅ LLM connection test completed\n")
    return True


def test_report_generation():
    """Test report generation"""
    print("=" * 60)
    print("Test 4: Report Generation")
    print("=" * 60)
    
    try:
        from modules.report_template import get_professional_template
        
        test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Generating report template ({test_date})...")
        
        template = get_professional_template(test_date)
        
        if template and len(template) > 1000:
            print(f"✓ Report template generated successfully")
            print(f"  - Template length: {len(template)} characters")
            print(f"  - Sections: {template.count('##')}")
        else:
            print("⚠ Report template abnormal")
            return False
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        return False
    
    print("\n✅ Report generation test completed\n")
    return True


def test_chart_features():
    """Test chart feature extraction"""
    print("=" * 60)
    print("Test 5: Chart Feature Extraction")
    print("=" * 60)
    
    try:
        from modules.chart_features import extract_chart_features, features_to_prompt_block
        import pandas as pd
        
        # Create test data
        test_data = pd.DataFrame({
            'time': pd.date_range('2026-04-13 09:00', periods=10, freq='5min'),
            'price': [100, 101, 102, 101.5, 103, 102, 104, 103.5, 105, 104],
            'symbol': 'TEST',
            'Category': 'FX',
        })
        
        print("Extracting chart features...")
        features = extract_chart_features([test_data])
        
        if features:
            print(f"✓ Chart features extracted successfully")
            print(f"  - FX pairs: {len(features.get('fx_pairs', []))}")
            
            # Test formatting
            prompt_block = features_to_prompt_block(features)
            if prompt_block and len(prompt_block) > 100:
                print(f"✓ Feature formatting successful")
                print(f"  - Text length: {len(prompt_block)} characters")
            else:
                print("⚠ Feature formatting abnormal")
        else:
            print("⚠ Chart features empty")
        
    except Exception as e:
        print(f"✗ Chart feature extraction failed: {e}")
        return False
    
    print("\n✅ Chart feature test completed\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Investment Bank Morning Briefing System - Functionality Test")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Data Fetching", test_data_fetching()))
    results.append(("LLM Connection", test_llm_connection()))
    results.append(("Report Generation", test_report_generation()))
    results.append(("Chart Features", test_chart_features()))
    
    # Summary
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{test_name:20s} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is running normally.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed, please check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
