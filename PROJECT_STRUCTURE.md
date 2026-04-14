# 项目结构说明

## 📁 完整目录结构

```
market_diary_project/
│
├── 📄 README_PROFESSIONAL.md          # 专业版完整文档
├── 📄 QUICK_START.md                  # 5分钟快速开始指南
├── 📄 UPGRADE_GUIDE.md                # 从学生版升级指南
├── 📄 PROJECT_STRUCTURE.md            # 本文件：项目结构说明
│
├── 🔧 .env.example                    # 环境变量配置模板
├── 🔧 config_example.json             # 系统配置文件示例
│
├── 🚀 run_morning_briefing.sh         # Linux/Mac 启动脚本
├── 🚀 run_morning_briefing.bat        # Windows 启动脚本
├── 🧪 test_professional_system.py     # 系统功能测试脚本
│
├── 📂 market_diary/                   # 核心代码目录
│   │
│   ├── 📄 main_professional.py        # ✨ 专业版主程序（新）
│   ├── 📄 main.py                     # 原学生版主程序（保留）
│   ├── 📄 requirements.txt            # Python 依赖列表
│   │
│   └── 📂 modules/                    # 功能模块目录
│       │
│       ├── 📄 __init__.py
│       │
│       ├── 📊 data_fetcher.py         # 市场数据获取（原有）
│       ├── 📊 chart_features.py       # 图表特征提取（原有）
│       ├── 📊 llm_client.py           # LLM 客户端（原有）
│       │
│       ├── 📅 macro_calendar.py       # ✨ 宏观经济日历（新）
│       ├── 📰 sector_news.py          # ✨ 行业新闻聚合（新）
│       ├── 💹 market_movers.py        # ✨ 市场异动分析（新）
│       ├── ⚠️ risk_radar.py           # ✨ 风险雷达监控（新）
│       └── 📝 report_template.py      # ✨ 专业报告模板（新）
│
├── 📂 reports/                        # 学生版报告输出目录
│   ├── 2026-04-13.md
│   ├── 2026-04-14.md
│   └── 📂 charts/                     # 图表文件
│       ├── fx_2026-04-13.png
│       ├── multi_2026-04-13.png
│       └── ...
│
├── 📂 reports_professional/           # ✨ 专业版报告输出目录（新）
│   ├── 2026-04-13_morning_briefing.md
│   ├── 2026-04-14_morning_briefing.md
│   └── 📂 charts/                     # 图表文件
│       ├── fx_2026-04-13.png
│       ├── features_2026-04-13.json   # 图表特征数据
│       └── ...
│
└── 📂 logs/                           # 日志目录（可选）
    └── morning_briefing.log
```

## 📦 核心模块说明

### 1. 主程序

#### `main_professional.py` ✨ 新增
专业版主程序，整合所有模块生成投行级别晨报。

**主要功能：**
- 数据采集协调
- 模块调用管理
- 报告组装
- 错误处理

**使用方法：**
```bash
python market_diary/main_professional.py --date 2026-04-13
```

#### `main.py`
原学生版主程序，保留用于对比和向后兼容。

### 2. 数据模块

#### `data_fetcher.py`
市场数据获取模块（原有，已优化）

**功能：**
- Yahoo Finance 数据获取
- 多资产类别支持（股票、外汇、商品、债券）
- 日内数据和日线数据
- 数据清洗和标准化

**主要函数：**
```python
fetch_market_data(date: str) -> Dict
fetch_news(max_per_feed: int) -> List[str]
```

#### `macro_calendar.py` ✨ 新增
宏观经济日历模块

**功能：**
- 经济数据发布日历
- 已公布数据与预期偏差分析
- 央行事件和官员讲话
- 重要性评级

**主要类：**
```python
class MacroCalendar:
    def fetch_economic_calendar(date: str) -> Dict
    def fetch_central_bank_events(date: str) -> List[Dict]
```

**数据源接入点：**
- Trading Economics API
- Investing.com
- Bloomberg Calendar

#### `sector_news.py` ✨ 新增
行业新闻聚合模块

**功能：**
- 按行业分类新闻（8大行业）
- 新闻重要性评分
- 财报日历
- 分析师评级追踪

**主要类：**
```python
class SectorNewsAggregator:
    def fetch_sector_news(max_per_sector: int) -> Dict
    def fetch_earnings_calendar(date: str) -> List[Dict]
    def fetch_analyst_changes(date: str) -> List[Dict]
```

**支持的行业：**
- Technology（科技）
- Financials（金融）
- Healthcare（医疗）
- Energy（能源）
- Consumer（消费）
- Industrials（工业）
- Materials（材料）
- Real Estate（房地产）

#### `market_movers.py` ✨ 新增
市场异动分析模块

**功能：**
- 盘前涨跌幅榜
- ETF 资金流向（13个主要 ETF）
- 期权市场异常活跃标的
- A股大宗交易和龙虎榜

**主要类：**
```python
class MarketMoversAnalyzer:
    def fetch_premarket_movers(top_n: int) -> Dict
    def fetch_etf_flows(date: str) -> List[Dict]
    def fetch_unusual_options(date: str) -> List[Dict]
```

**监控的 ETF：**
- SPY, QQQ, IWM（美股）
- EEM, FXI（新兴市场）
- GLD, SLV（贵金属）
- USO（原油）
- TLT, HYG, LQD（债券）

#### `risk_radar.py` ✨ 新增
风险雷达监控模块

**功能：**
- 地缘政治风险监控
- 重大事件日历（FOMC、期权到期、IPO、解禁）
- 技术面关键支撑/阻力位
- 市场情绪指标

**主要类：**
```python
class RiskRadar:
    def fetch_geopolitical_risks() -> List[Dict]
    def fetch_upcoming_events(days_ahead: int) -> List[Dict]
    def fetch_technical_levels(current_prices: Dict) -> Dict
    def fetch_sentiment_indicators() -> Dict
```

**监控的情绪指标：**
- AAII Bull/Bear Sentiment
- CNN Fear & Greed Index
- Put/Call Ratio
- VIX Term Structure

### 3. 分析模块

#### `chart_features.py`
图表特征提取模块（原有）

**功能：**
- 从时间序列数据提取数值特征
- 转折点检测
- 相关性计算
- 格式化为 LLM 可读文本

**主要函数：**
```python
extract_chart_features(timeseries_list: List[pd.DataFrame]) -> Dict
features_to_prompt_block(features: Dict) -> str
```

#### `llm_client.py`
LLM 客户端模块（原有，已优化）

**功能：**
- OpenAI / MiniMax API 调用
- Prompt 管理
- 错误处理和重试

**主要函数：**
```python
get_client() -> OpenAI
generate_report(date: str, market_summary: Dict, ...) -> str
```

**支持的 LLM：**
- MiniMax M2.7（默认）
- OpenAI GPT-4 / GPT-4-turbo
- 其他 OpenAI 兼容 API

### 4. 报告模块

#### `report_template.py` ✨ 新增
专业报告模板模块

**功能：**
- 投行级别报告模板
- 专业 System Prompt
- 报告组装逻辑
- LLM Prompt 生成

**主要函数：**
```python
get_professional_template(date: str) -> str
format_professional_report(...) -> str
get_llm_prompt_for_professional_report(...) -> str
```

**报告章节：**
1. Executive Summary
2. Market Snapshot
3. Macro Calendar
4. Central Bank Watch
5. Sector & Stock News
6. Pre-market Movers
7. Risk Radar
8. Technical Levels
9. Trading Strategy
10. Chart Analysis
11. Tomorrow's Focus

## 🔄 数据流程

```
┌─────────────────────────────────────────────────────────────┐
│                    main_professional.py                      │
│                        (主程序)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         1. 数据采集阶段                  │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│  data_fetcher    │                    │  macro_calendar  │
│  (市场数据)      │                    │  (宏观日历)      │
└──────────────────┘                    └──────────────────┘
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│  sector_news     │                    │  market_movers   │
│  (行业新闻)      │                    │  (市场异动)      │
└──────────────────┘                    └──────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         2. 特征提取阶段                  │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ chart_features   │
                    │ (图表特征提取)   │
                    └──────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         3. AI 分析阶段                   │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   llm_client     │
                    │   (AI 分析)      │
                    └──────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         4. 报告组装阶段                  │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ report_template  │
                    │ (报告组装)       │
                    └──────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         5. 输出阶段                      │
        └─────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  reports_professional/   │
                │  YYYY-MM-DD_morning_     │
                │  briefing.md             │
                └──────────────────────────┘
```

## 🔌 扩展点

### 1. 添加新的数据源

在 `modules/` 下创建新模块：

```python
# modules/custom_data.py
def fetch_custom_data(date: str) -> Dict:
    # 你的数据获取逻辑
    return data
```

在 `main_professional.py` 中调用：

```python
from modules.custom_data import fetch_custom_data

# 在 fetch_all_data() 函数中添加
custom_data = fetch_custom_data(report_date)
all_data['custom'] = custom_data
```

### 2. 自定义报告章节

编辑 `modules/report_template.py`：

```python
def get_professional_template(date: str) -> str:
    template = f"""
    ...
    ## 🆕 Your Custom Section
    {{{{custom_section}}}}
    ...
    """
    return template
```

### 3. 接入专业数据源

#### Bloomberg Terminal

```python
# modules/bloomberg_data.py
import blpapi

def fetch_bloomberg_data(tickers, fields):
    session = blpapi.Session()
    session.start()
    # Bloomberg API 调用
    return data
```

#### Wind 万得

```python
# modules/wind_data.py
from WindPy import w

w.start()
def fetch_wind_data(codes, fields):
    data = w.wsd(codes, fields, start_date, end_date)
    return data
```

### 4. 添加通知渠道

```python
# modules/notifications.py

def send_email(report_content, recipients):
    # 邮件发送逻辑
    pass

def send_wechat(report_summary):
    # 企业微信推送逻辑
    pass

def send_dingtalk(report_summary):
    # 钉钉推送逻辑
    pass
```

## 📊 配置文件

### `.env` 环境变量

```bash
# LLM API
MINIMAX_API_KEY=xxx
LLM_MODEL=MiniMax-M2.7

# 数据源 API
BLOOMBERG_API_KEY=xxx
WIND_API_KEY=xxx

# 通知配置
SMTP_HOST=smtp.gmail.com
SMTP_USER=xxx
SMTP_PASSWORD=xxx
```

### `config.json` 系统配置

```json
{
  "system": {
    "timezone": "Asia/Shanghai",
    "output_dir": "reports_professional"
  },
  "data_sources": {
    "market_data": {
      "provider": "yfinance",
      "fallback_days": 4
    }
  },
  "report": {
    "language": "zh-CN",
    "style": "professional"
  }
}
```

## 🧪 测试

运行完整测试：

```bash
python test_professional_system.py
```

测试单个模块：

```python
# 测试宏观日历
from modules.macro_calendar import fetch_macro_data
data = fetch_macro_data("2026-04-13")
print(data)
```

## 📝 开发规范

### 代码风格
- 遵循 PEP 8
- 使用类型提示（Type Hints）
- 函数必须有 docstring

### 提交规范
```
feat: 添加新功能
fix: 修复 bug
docs: 更新文档
refactor: 重构代码
test: 添加测试
```

### 分支管理
- `main`: 稳定版本
- `dev`: 开发版本
- `feature/*`: 新功能分支

## 📞 技术支持

- 📖 文档：`README_PROFESSIONAL.md`
- 🚀 快速开始：`QUICK_START.md`
- 🔄 升级指南：`UPGRADE_GUIDE.md`
- 🐛 问题反馈：GitHub Issues

---

**项目维护者：[Your Name]**  
**最后更新：2026-04-14**
