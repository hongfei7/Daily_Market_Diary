"""
report_template.py — 投行研究院晨报模板

专业化的 Morning Briefing 结构
"""

from datetime import datetime
from typing import Dict, List, Optional


PROFESSIONAL_SYSTEM_PROMPT = """
You are a Chief Strategy Analyst at a top-tier investment bank (Goldman Sachs/Morgan Stanley level), 
responsible for writing the daily Morning Briefing.

[TARGET AUDIENCE]
- Traders (need specific entry points, stop losses, target prices)
- Fund Managers (need allocation recommendations and risk assessments)
- Institutional Clients (need market insights and risk alerts)

[CORE PRINCIPLES]
1. Data-Driven: Every judgment must cite specific data and indicators
2. Actionable: Provide clear prices, positions, stops, targets
3. Risk-Aware: Every recommendation must have invalidation conditions and hedge plans
4. Concise & Efficient: Must be readable within 15 minutes before market open

[WRITING STANDARDS]
✅ Use: "We believe", "Recommend", "Expect", "Data shows"
❌ Avoid: "Maybe", "Perhaps", "Probably", "Seems"

✅ Specific: "SPX support at 6,850, break below targets 6,800, upside to 6,950"
❌ Vague: "SPX may decline"

✅ Quantified: "Win rate 65%, Risk/Reward 2:1, Position size 30%"
❌ Qualitative: "Risk manageable, moderate participation"

✅ Conditional: "If CPI > 0.3%, shift defensive; if < 0.2%, add tech exposure"
❌ Unconditional: "Recommend buying tech stocks"

[DATA CITATION STANDARDS]
- Price quotes: (SPX 6,875, Close) or (DXY 98.5, as of 4:00 PM ET)
- Changes: (+1.02% vs prior close) or (-4.6bp vs last Friday)
- Indicators: (VIX 19.2, below 20-day MA 21.5) or (Put/Call 0.68, 30th percentile)
- Thresholds: (1.1% from key resistance 6,950) or (1.2% from stop loss)

[OUTPUT FORMAT]
Strictly follow the provided Morning Briefing template. Do not add extra explanations or preambles.
First line must be: # Morning Briefing | {DATE}

[PROHIBITED]
- Do not use meta-narrative like "Based on provided data", "According to the above information"
- Do not use vague words like "maybe", "perhaps"
- Do not repeat data, focus on interpretation and implications
- Do not write like an academic paper, write like a trading manual
- Do not give recommendations without stop losses
- Do not give views without invalidation conditions

[EXAMPLE OUTPUT EXCERPT]

**Market Theme:** US Treasury yield decline drives tech rally, but low volume suggests weak participation

**Key Drivers:**
1. US 10Y yield down 4.6bp to 4.35% → Lowers tech discount rate → Nasdaq +1.06% | Duration: Short-term (awaiting CPI confirmation)
2. Goldman Q1 trading revenue beats by 15% → Lifts financials sentiment → Financials +1.2% | Duration: Medium-term (earnings season catalyst)
3. Crude futures curve steepens to -$2.5/bbl → Demand concerns rise → Energy -0.8% | Duration: Medium-term (watch OPEC meeting)

**Strategy Recommendations:**
- **Long:** QQQ @ 485-487 | Target 495 (+1.8%) | Stop 480 (-1.2%) | Position 30% | Win Rate 65%
  Rationale: Rate decline favors tech, MACD golden cross, RSI 55 (neutral-bullish)
  Risk: Exit on stop if Thursday CPI exceeds expectations
  
- **Short:** USO @ 78-79 | Target 75 (-4%) | Stop 81 (+3%) | Position 15% | Risk/Reward 2:1
  Rationale: Curve steepening shows weak demand, broke below 50-day MA 78.5
  Risk: Stop out if OPEC announces surprise production cut
  
- **Watch:** Await Thursday CPI data | Trigger: Core CPI > 0.3% shift defensive / < 0.2% add growth

**Risk Alert:** Thursday CPI upside risk (Core CPI est. 0.2%, if reaches 0.4% rate expectations reverse) | 
Probability 25% | Hedge: Buy VIX 20-22 call spread, cost 0.5 points
"""


def get_professional_template(date: str) -> str:
    """
    返回专业晨报的 Markdown 模板
    
    Args:
        date: YYYY-MM-DD 格式
    """
    weekday = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
    
    # 使用普通字符串，不使用 f-string，避免花括号被解析
    template = """# Morning Briefing | DATE_PLACEHOLDER (WEEKDAY_PLACEHOLDER)

---

**Investment Banking Research Institute · Strategy Research Department**

Report Time: {report_time_placeholder}  
Analyst Team: Strategy Research  
AI Model: {model_name_placeholder}

---

## I. Executive Summary

**Market Theme:** [一句话总结今日市场主题，不超过30字]

**Market Sentiment:** Risk-On / Risk-Off / Neutral | VIX: [数值] | Put/Call Ratio: [数值]

**Key Drivers:**
1. [驱动因素] → [影响] | Duration: Short/Medium/Long-term | Confidence: High/Medium/Low
2. [驱动因素] → [影响] | Duration: Short/Medium/Long-term | Confidence: High/Medium/Low
3. [驱动因素] → [影响] | Duration: Short/Medium/Long-term | Confidence: High/Medium/Low

**Strategy Recommendations:**
- **Long:** [标的] @ [入场价位] | Target [价位] (+X%) | Stop [价位] (-Y%) | Position Z% | Win Rate W%
  Rationale: [1-2句话，包含技术面+基本面]
  Risk: [失效条件]
  
- **Short:** [标的] @ [入场价位] | Target [价位] (-X%) | Stop [价位] (+Y%) | Position Z% | Risk/Reward R:1
  Rationale: [1-2句话，包含技术面+基本面]
  Risk: [失效条件]
  
- **Watch:** Wait for [具体事件/数据] | Trigger: [具体指标阈值]

**Risk Alert:** [今日最大的尾部风险] | Probability: Low/Medium/High (X%) | Hedge: [具体操作]

---

## II. Market Snapshot

### Overnight US Equities
{us_equity_summary_placeholder}

**Key Metrics:**
- Advance/Decline Ratio: [数值] (vs 20-day MA [数值])
- Volume: [数值]B (vs 20-day MA +/-X%)
- Sector Performance: [领涨板块] +X% | [领跌板块] -Y%

### European Markets
{europe_summary_placeholder}

### Asia-Pacific Markets
{asia_summary_placeholder}

**China Markets:**
- Northbound Flows: [流入/流出] [金额]B CNY (Week-to-date [金额]B)
- Margin Balance: [数值]T CNY (Daily change +/-[金额]B)

### FX & Commodities
{fx_commodity_summary_placeholder}

**Key Levels:**
- DXY: [当前] | Resistance [数值] (+X%) | Support [数值] (-Y%)
- Gold: [当前] | Resistance [数值] | Support [数值]
- Crude Oil: [当前] | Curve Structure: Contango/Backwardation

### Fixed Income
{fixed_income_summary_placeholder}

**Yield Curve:**
- 2s10s Spread: [数值]bp (vs Last Week [数值]bp) | Trend: Steepening/Flattening
- Real Yield: [数值]% | Breakeven Inflation: [数值]%

---

## III. Macro Calendar

{macro_calendar_section_placeholder}

**Key Focus:**
- [时间] | [国家] | [指标] | Forecast [数值] vs Prior [数值] | Importance: High / Medium / Low

---

## IV. Central Bank Watch

{central_bank_section_placeholder}

**Policy Expectations:**
- Federal Reserve: Next Meeting [日期] | Rate Cut Probability [X%] (CME FedWatch)
- European Central Bank: [观点]
- People's Bank of China: [观点]

---

## V. Sector & Stock News

{sector_news_section_placeholder}

---

## VI. Pre-market Movers

{market_movers_section_placeholder}

---

## VII. Risk Radar

{risk_radar_section_placeholder}

---

## VIII. Key Thresholds

{technical_levels_section_placeholder}

### Technical Levels
| Asset | Current | Key Resistance | Distance | Key Support | Distance | Breakout Prob |
|-------|---------|----------------|----------|-------------|----------|---------------|
| SPX   | [数值]  | [数值]         | +X%      | [数值]      | -Y%      | Z%            |
| NDX   | [数值]  | [数值]         | +X%      | [数值]      | -Y%      | Z%            |
| DXY   | [数值]  | [数值]         | +X%      | [数值]      | -Y%      | Z%            |

### Macro Thresholds
- **US 10Y Yield**: [当前]% | Critical Level [数值]% | Distance +/-Xbp | Implication: [描述]
- **VIX Index**: [当前] | Critical Level [数值] | Implication: [描述]
- **Credit Spread**: [当前]bp | Historical Percentile [X%] | Interpretation: [描述]

---

## IX. Trading Strategy

### Today's Key Events
1. **[时间]** | **Event:** [描述] | **Expected Impact:** Positive/Negative/Neutral | **Trade Setup:** [具体操作]
2. **[时间]** | **Event:** [描述] | **Expected Impact:** Positive/Negative/Neutral | **Trade Setup:** [具体操作]

### Recommended Trades

#### Trade #1: [标的名称]
- **Direction:** Long/Short
- **Entry:** [价格区间] or [技术条件]
- **Stop Loss:** [价格] (-X%) | Trigger: [描述]
- **Target:** [价格] (+Y%) | Partial Exits: [价格1] 50% / [价格2] 50%
- **Position Size:** [X%] | Risk Exposure: [金额]
- **Win Rate:** [X%] | Risk/Reward: [R:1]
- **Rationale:** 
  - Fundamental: [1句话]
  - Technical: [1句话，包含具体指标]
  - Catalyst: [1句话]
- **Risk:** [主要风险因素] | Invalidation: [具体指标/事件]
- **Hedge:** [可选对冲方案]

#### Trade #2: [标的名称]
[同上结构]

#### Trade #3: [标的名称]
[同上结构]

### Portfolio Risk Management
- **Total Exposure:** [X%] (Cash [Y%])
- **Max Daily Loss:** -[X%] (Triggers position reduction)
- **Correlation Risk:** [描述多头之间的相关性]
- **Hedge Ratio:** [X%] (via [工具])

---

## X. Chart Analysis

{chart_analysis_section_placeholder}

---

## XI. Tomorrow's Focus

### Economic Data Releases
- **[时间]** | [国家] | **[指标]** | Forecast [数值] vs Prior [数值] | Importance: High/Medium/Low
  Impact Scenarios: [正面/负面情景分析]

### Earnings Releases
- **[公司]** | [Pre-market/After-hours] | EPS Est. $[数值] | Revenue Est. $[数值]B
  Key Metrics: [关键指标]

### Other Catalysts
- [事件描述] | Potential Impact: [描述]

### Scenario Analysis
**Scenario 1 (Probability X%):** If [条件] → Expected [结果] → Strategy: [操作]
**Scenario 2 (Probability Y%):** If [条件] → Expected [结果] → Strategy: [操作]
**Scenario 3 (Probability Z%):** If [条件] → Expected [结果] → Strategy: [操作]

---

## Disclaimer

This report is for professional investors only and does not constitute investment advice. 
Markets involve risks and investors should exercise caution. The views and forecasts in this 
report are based on currently available information and may be adjusted as market conditions change. 
Past performance does not guarantee future results. Please make prudent decisions based on your 
own risk tolerance.

---

*Report Generated: {generation_time_placeholder}*  
*Data Sources: Bloomberg, Wind, Reuters, Internal Models*  
*For Institutional Use Only. Not for Redistribution.*

"""
    
    # 替换日期和星期
    template = template.replace('DATE_PLACEHOLDER', date)
    template = template.replace('WEEKDAY_PLACEHOLDER', weekday)
    
    return template


def format_professional_report(
    date: str,
    market_data: Dict,
    macro_data: Dict,
    sector_data: Dict,
    movers_data: Dict,
    risk_data: Dict,
    llm_analysis: str,
    charts_section: str,
    model_name: Optional[str] = None,
) -> str:
    """
    组装专业晨报
    
    Args:
        date: 报告日期
        market_data: 市场数据
        macro_data: 宏观日历数据
        sector_data: 行业新闻数据
        movers_data: 市场异动数据
        risk_data: 风险监控数据
        llm_analysis: LLM 生成的分析
        charts_section: 图表部分
        model_name: 使用的 LLM 模型名称
        
    Returns:
        完整的晨报 Markdown 文本
    """
    template = get_professional_template(date)
    
    # 填充时间戳
    now = datetime.now()
    report_time = now.strftime('%Y-%m-%d %H:%M:%S')
    
    # 获取模型名称
    if model_name is None:
        import os
        model_name = os.getenv("LLM_MODEL", "Not Specified")
    
    # 替换所有占位符（使用单花括号格式）
    report = template.replace('{report_time_placeholder}', report_time)
    report = report.replace('{generation_time_placeholder}', report_time)
    report = report.replace('{model_name_placeholder}', model_name)
    
    # 替换数据模块占位符
    report = report.replace('{macro_calendar_section_placeholder}', 
                           macro_data.get('formatted_text', '*宏观日历数据获取中...*'))
    report = report.replace('{sector_news_section_placeholder}', 
                           sector_data.get('formatted_text', '*行业新闻数据获取中...*'))
    report = report.replace('{market_movers_section_placeholder}', 
                           movers_data.get('formatted_text', '*市场异动数据获取中...*'))
    report = report.replace('{risk_radar_section_placeholder}', 
                           risk_data.get('formatted_text', '*风险雷达数据获取中...*'))
    report = report.replace('{chart_analysis_section_placeholder}', charts_section)
    
    # 替换其他占位符（如果 LLM 没有生成，则显示占位符）
    report = report.replace('{us_equity_summary_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{europe_summary_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{asia_summary_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{fx_commodity_summary_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{fixed_income_summary_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{central_bank_section_placeholder}', '*等待 AI 分析...*')
    report = report.replace('{technical_levels_section_placeholder}', '*等待 AI 分析...*')
    
    # 如果 LLM 分析成功，将其插入到报告末尾
    if llm_analysis and not llm_analysis.startswith('*AI 分析'):
        report += f"\n\n---\n\n## XII. AI Deep Analysis\n\n{llm_analysis}\n"
    else:
        report += f"\n\n---\n\n## XII. AI Deep Analysis\n\n{llm_analysis}\n"
    
    return report


def get_llm_prompt_for_professional_report(
    date: str,
    market_summary: Dict,
    chart_features: str,
    news_headlines: List[str],
    macro_calendar: Dict,
) -> str:
    """
    为专业晨报生成 LLM prompt
    
    Args:
        date: 报告日期
        market_summary: 市场数据摘要
        chart_features: 图表特征
        news_headlines: 新闻标题
        macro_calendar: 宏观日历
        
    Returns:
        LLM prompt 文本
    """
    prompt = f"""请为 {date} 撰写投行研究院晨报的核心分析部分。

## 输入数据

### 市场数据
{_format_market_summary(market_summary)}

### 图表特征
{chart_features}

### 新闻标题
{chr(10).join(news_headlines[:20])}

### 宏观日历
{_format_macro_calendar(macro_calendar)}

## 输出要求

请按以下结构输出分析：

### 1. Executive Summary
- 市场主题（一句话）
- 关键驱动因素（3个，每个一句话）
- 今日策略建议（做多/做空/观望，各一句话）
- 风险提示（一句话）

### 2. Market Snapshot 分段分析
- 隔夜美股：[2-3句话，重点是驱动因素和板块表现]
- 欧洲市场：[1-2句话]
- 亚太市场：[1-2句话，重点关注中国]
- 外汇与大宗商品：[2-3句话，重点是美元、黄金、原油]
- 固定收益：[2-3句话，重点是美债收益率曲线]

### 3. Trading Strategy
- 推荐2-3个具体交易设置，每个包括：
  - 标的、方向、入场点、止损、目标、仓位、理由、风险

### 4. Technical Levels
- 列出 SPX、NDX、DXY、US10Y 的关键支撑/阻力位

注意：
1. 使用投行专业术语
2. 给出明确观点，不要模糊表述
3. 每个判断必须有数据支撑
4. 保持简洁，总字数控制在 1500 字以内
"""
    
    return prompt


def _format_market_summary(summary: Dict) -> str:
    """格式化市场摘要为文本"""
    lines = []
    for category, items in summary.items():
        lines.append(f"**{category}**")
        for name, data in items.items():
            if isinstance(data, dict):
                lines.append(f"- {name}: {data.get('Price', 'N/A')} ({data.get('Pct Change', 'N/A')})")
            else:
                lines.append(f"- {name}: {data}")
    return "\n".join(lines)


def _format_macro_calendar(calendar: Dict) -> str:
    """格式化宏观日历为文本"""
    lines = []
    
    if calendar.get('calendar', {}).get('released'):
        lines.append("已公布数据：")
        for item in calendar['calendar']['released'][:5]:
            lines.append(f"- {item.get('indicator', 'N/A')}: {item.get('actual', 'N/A')} (预期 {item.get('forecast', 'N/A')})")
    
    if calendar.get('calendar', {}).get('upcoming'):
        lines.append("\n待公布数据：")
        for item in calendar['calendar']['upcoming'][:5]:
            lines.append(f"- {item.get('time', 'N/A')} | {item.get('indicator', 'N/A')}")
    
    return "\n".join(lines)
