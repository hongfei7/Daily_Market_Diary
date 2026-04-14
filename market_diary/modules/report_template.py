"""
report_template.py — 投行研究院晨报模板

专业化的 Morning Briefing 结构
"""

from datetime import datetime
from typing import Dict, List, Optional


PROFESSIONAL_SYSTEM_PROMPT = """
你是一位资深的投行研究院首席策略分析师，负责撰写每日晨报（Morning Briefing）。

目标读者：交易员、基金经理、机构客户

核心要求：
1. 专业性：使用投行级别的专业术语和分析框架
2. 简洁性：开盘前15分钟内必须读完，每个部分直击要点
3. 可操作性：提供明确的交易建议和风险提示
4. 全球视野：覆盖美国、欧洲、中国三大市场

写作风格：
- 使用第一人称复数（"我们认为"、"我们建议"）体现团队专业性
- 避免模糊表述，给出明确观点和理由
- 数据驱动，每个判断必须有数据支撑
- 风险意识，对不确定性保持警惕

输出格式：
严格按照提供的 Morning Briefing 模板输出，不要添加额外的解释或前言。
第一行必须是：# 📊 Morning Briefing | {DATE}

禁止事项：
- 不要说"根据提供的数据"、"基于以上信息"等元叙述
- 不要使用"可能"、"或许"等模糊词汇，要给出明确判断
- 不要重复数据，专注于解读和含义
- 不要写成学术论文，要写成实战手册
"""


def get_professional_template(date: str) -> str:
    """
    返回专业晨报的 Markdown 模板
    
    Args:
        date: YYYY-MM-DD 格式
    """
    weekday = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
    
    template = f"""# 📊 Morning Briefing | {date} ({weekday})

> **投行研究院 · 策略研究部**  
> 报告时间：{{report_time}}  
> 分析师：策略团队

---

## 📌 Executive Summary（核心观点）

**市场主题：** [一句话总结今日市场主题]

**关键驱动因素：**
1. [驱动因素1 + 影响方向]
2. [驱动因素2 + 影响方向]
3. [驱动因素3 + 影响方向]

**今日策略建议：**
- **做多：** [具体标的/板块 + 理由]
- **做空/规避：** [具体标的/板块 + 理由]
- **观望：** [需要等待的催化剂]

**风险提示：** [今日最大的尾部风险]

---

## 🌍 Market Snapshot（全球市场概览）

### 隔夜美股
{{us_equity_summary}}

### 欧洲市场
{{europe_summary}}

### 亚太市场
{{asia_summary}}

### 外汇与大宗商品
{{fx_commodity_summary}}

### 固定收益
{{fixed_income_summary}}

---

## 📅 Macro Calendar（宏观日历）

{{macro_calendar_section}}

---

## 🏛️ Central Bank Watch（央行动态）

{{central_bank_section}}

---

## 🏢 Sector & Stock News（行业与个股）

{{sector_news_section}}

---

## 💹 Pre-market Movers（盘前异动）

{{market_movers_section}}

---

## ⚠️ Risk Radar（风险雷达）

{{risk_radar_section}}

---

## 📊 Technical Levels（技术面关键位）

{{technical_levels_section}}

---

## 💡 Trading Strategy（交易策略）

### 今日重点关注
1. **时间点：** [具体时间] | **事件：** [事件描述] | **预期影响：** [影响分析]
2. ...

### 推荐交易设置

#### Trade #1: [标的名称]
- **方向：** Long/Short
- **入场点：** [价格/条件]
- **止损：** [价格] (-X%)
- **目标：** [价格] (+Y%)
- **仓位：** [建议仓位大小]
- **理由：** [1-2句话说明逻辑]
- **风险：** [主要风险因素]

#### Trade #2: [标的名称]
[同上结构]

### 对冲建议
- [具体对冲方案]

---

## 📈 Chart Analysis（图表分析）

{{chart_analysis_section}}

---

## 🔮 Tomorrow's Focus（明日关注）

### 重要数据发布
- [时间] | [国家] | [指标] | 预期 vs 前值

### 财报发布
- [公司] | 盘前/盘后 | 市场预期

### 其他催化剂
- [事件描述]

---

## 📝 Disclaimer（免责声明）

本报告仅供专业投资者参考，不构成投资建议。市场有风险，投资需谨慎。
报告中的观点和预测基于当前可获得的信息，可能随市场变化而调整。

---

*Report generated at {{generation_time}}*  
*Data sources: Bloomberg, Wind, Reuters, Internal Models*

"""
    
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
        
    Returns:
        完整的晨报 Markdown 文本
    """
    template = get_professional_template(date)
    
    # 填充时间戳
    now = datetime.now()
    report_time = now.strftime('%Y-%m-%d %H:%M:%S')
    
    # 组装各个部分
    report = template.replace('{{report_time}}', report_time)
    report = report.replace('{{generation_time}}', report_time)
    
    # 插入 LLM 生成的核心分析
    # LLM 应该生成 Executive Summary 和 Market Snapshot 部分
    
    # 插入各模块的格式化文本
    report = report.replace('{{macro_calendar_section}}', macro_data.get('formatted_text', ''))
    report = report.replace('{{sector_news_section}}', sector_data.get('formatted_text', ''))
    report = report.replace('{{market_movers_section}}', movers_data.get('formatted_text', ''))
    report = report.replace('{{risk_radar_section}}', risk_data.get('formatted_text', ''))
    
    # 插入图表
    report = report.replace('{{chart_analysis_section}}', charts_section)
    
    # 插入 LLM 分析（需要解析 LLM 输出并填充到对应位置）
    # 这里简化处理，将 LLM 分析附加到报告末尾
    report += f"\n\n---\n\n## 🤖 AI Deep Analysis\n\n{llm_analysis}\n"
    
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
