"""
test_github_actions.py - GitHub Actions 环境测试脚本

简化版测试，只测试关键功能
"""

import sys
import os

def test_basic_imports():
    """测试基础模块导入"""
    print("=" * 60)
    print("测试 1: 基础模块导入")
    print("=" * 60)
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✓ openai")
    except ImportError as e:
        print(f"✗ openai: {e}")
        return False
    
    try:
        import yfinance
        print("✓ yfinance")
    except ImportError as e:
        print(f"✗ yfinance: {e}")
        return False
    
    print("\n✅ 基础模块导入成功\n")
    return True


def test_project_modules():
    """测试项目模块导入"""
    print("=" * 60)
    print("测试 2: 项目模块导入")
    print("=" * 60)
    
    # 添加模块路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_diary'))
    
    modules = [
        'modules.data_fetcher',
        'modules.chart_features',
        'modules.llm_client',
        'modules.macro_calendar',
        'modules.sector_news',
        'modules.market_movers',
        'modules.risk_radar',
        'modules.report_template',
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n⚠️ {len(failed)} 个模块导入失败")
        return False
    
    print("\n✅ 项目模块导入成功\n")
    return True


def test_api_key():
    """测试 API 密钥配置"""
    print("=" * 60)
    print("测试 3: API 密钥配置")
    print("=" * 60)
    
    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("✗ 未设置 API 密钥")
        print("  请设置 MINIMAX_API_KEY 或 OPENAI_API_KEY 环境变量")
        return False
    
    print(f"✓ API 密钥已设置 (长度: {len(api_key)})")
    
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        print(f"✓ Base URL: {base_url}")
    
    model = os.getenv("LLM_MODEL")
    if model:
        print(f"✓ Model: {model}")
    
    print("\n✅ API 配置正常\n")
    return True


def test_llm_client():
    """测试 LLM 客户端"""
    print("=" * 60)
    print("测试 4: LLM 客户端")
    print("=" * 60)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'market_diary'))
    
    try:
        from modules.llm_client import get_client
        
        client = get_client()
        print("✓ LLM 客户端初始化成功")
        
        # 简单测试调用
        model_name = os.getenv("LLM_MODEL", "MiniMax-M2.7")
        print(f"  使用模型: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "回复'测试成功'"}
            ],
            max_tokens=20,
            temperature=0.7,
        )
        
        result = response.choices[0].message.content
        print(f"✓ LLM 调用成功")
        print(f"  响应: {result[:50]}")
        
        print("\n✅ LLM 客户端测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ LLM 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("GitHub Actions 环境测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("基础模块导入", test_basic_imports()))
    results.append(("项目模块导入", test_project_modules()))
    results.append(("API 密钥配置", test_api_key()))
    
    # 只有前面都通过才测试 LLM
    if all(r[1] for r in results):
        results.append(("LLM 客户端", test_llm_client()))
    else:
        print("⏭️  跳过 LLM 测试（前置条件未满足）\n")
    
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
        print("\n🎉 所有测试通过！可以运行晨报生成器。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查配置。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
