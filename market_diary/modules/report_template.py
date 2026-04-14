"""
report_template.py — 投行研究院晨报模板

专业化的 Morning Briefing 结构
"""

from datetime import datetime
from typing import Dict, List, Optional


PROFESSIONAL_SYSTEM_PROMPT = """
你是顶级投行（高盛/摩根士丹利级别）的首席策略分析师，负责撰写每日晨报（Morning Briefing）。

【目标读者】
- 交易员（需要具体入场点、止损、目标价）
- 基金经理（需要配置建议和风险评估）
- 机构客户（需要市场洞察和风险提示）

【核心原则】
1. 数据驱动：每个判断必须引用具体数据和指标
2. 可执行性：给出明确的价格、仓位、止损、目标
3. 风险意识：每个建议都要有失效条件和对冲方案
4. 简洁高效：开盘前15分钟内必须读完

【写作规范】
✅ 使用："我们认为"、"建议"、"预计"、"数据显示"
❌ 禁止："可能"、"或许"、"大概"、"似乎"

✅ 具体："SPX 6,850支撑，跌破看6,800，目标6,950"
❌ 模糊："SPX可能会下跌"

✅ 量化："胜率65%，盈亏比2:1，建议仓位30%"
❌ 定性："风险可控，适度参与"

✅ 有条件："若CPI > 0.3%，则转为防御；若 < 0.2%，则加仓科技"
❌ 无条件："建议买入科技股"

【数据引用规范】
- 引用价格：(SPX 6,875, 收盘价) 或 (DXY 98.5, 截至美东16:00)
- 引用变化：(+1.02% vs 昨日收盘) 或 (-4.6bp vs 上周五)
- 引用指标：(VIX 19.2, 低于20日均线21.5) 或 (Put/Call 0.68, 历史30%分位)
- 引用阈值：(距离关键阻力6,950还有+1.1%) 或 (距离止损位-1.2%)

【输出格式】
严格按照提供的 Morning Briefing 模板输出，不要添加额外的解释或前言。
第一行必须是：# 📊 Morning Briefing | {DATE}

【禁止事项】
- 不要说"根据提供的数据"、"基于以上信息"等元叙述
- 不要使用"可能"、"或许"等模糊词汇
- 不要重复数据，专注于解读和含义
- 不要写成学术论文，要写成实战手册
- 不要给出没有止损的建议
- 不要给出没有失效条件的观点

【示例输出片段】

**市场主题：** 美债收益率下行推动科技股反弹，但成交量偏低显示参与度不足

**关键驱动因素：**
1. 美债10Y收益率下行4.6bp至4.35% → 降低科技股贴现率 → 纳指+1.06% | 持续性：短期（等待CPI验证）
2. 高盛Q1交易收入超预期15% → 提振金融板块情绪 → 金融+1.2% | 持续性：中期（财报季催化）
3. 原油期货曲线走陡至-$2.5/桶 → 需求担忧升温 → 能源-0.8% | 持续性：中期（关注OPEC会议）

**今日策略建议：**
- **做多：** QQQ @ 485-487 | 目标 495 (+1.8%) | 止损 480 (-1.2%) | 仓位 30% | 胜率 65%
  理由：利率下行利好科技，MACD金叉，RSI 55（中性偏强）
  风险：周四CPI若超预期则止损离场
  
- **做空：** USO @ 78-79 | 目标 75 (-4%) | 止损 81 (+3%) | 仓位 15% | 盈亏比 2:1
  理由：期货曲线走陡显示需求疲软，跌破50日均线78.5
  风险：OPEC意外减产则止损
  
- **观望：** 等待周四CPI数据 | 触发条件：核心CPI > 0.3% 转防御 / < 0.2% 加仓成长

**风险提示：** 周四CPI超预期风险（核心CPI预期0.2%，若达0.4%则利率预期逆转）| 概率 25% | 对冲：买入VIX 20-22看涨价差，成本0.5点
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

**市场主题：** [一句话总结今日市场主题，不超过30字]

**市场情绪：** 🟢 Risk-On / 🔴 Risk-Off / 🟡 中性 | **VIX**: [数值] | **Put/Call**: [数值]

**关键驱动因素：**
1. [驱动因素] → [影响] | 持续性: 短期/中期/长期 | 确定性: 高/中/低
2. [驱动因素] → [影响] | 持续性: 短期/中期/长期 | 确定性: 高/中/低
3. [驱动因素] → [影响] | 持续性: 短期/中期/长期 | 确定性: 高/中/低

**今日策略建议：**
- **做多：** [标的] @ [入场价位] | 目标 [价位] (+X%) | 止损 [价位] (-Y%) | 仓位 Z% | 胜率 W%
  理由：[1-2句话，包含技术面+基本面]
  风险：[失效条件]
  
- **做空：** [标的] @ [入场价位] | 目标 [价位] (-X%) | 止损 [价位] (+Y%) | 仓位 Z% | 盈亏比 R:1
  理由：[1-2句话，包含技术面+基本面]
  风险：[失效条件]
  
- **观望：** 等待 [具体事件/数据] | 触发条件: [具体指标阈值]

**风险提示：** [今日最大的尾部风险] | 概率: 低/中/高 (X%) | 对冲方案: [具体操作]

---

## 🌍 Market Snapshot（全球市场概览）

### 隔夜美股
{{us_equity_summary}}

**关键数据：**
- 涨跌家数比: [数值] (vs 20日均线 [数值])
- 成交量: [数值]亿 (vs 20日均线 +/-X%)
- 板块表现: [领涨板块] +X% | [领跌板块] -Y%

### 欧洲市场
{{europe_summary}}

### 亚太市场
{{asia_summary}}

**中国市场：**
- 北向资金: [流入/流出] [金额]亿 (本周累计 [金额]亿)
- 融资余额: [数值]万亿 (日变化 +/-[金额]亿)

### 外汇与大宗商品
{{fx_commodity_summary}}

**关键位：**
- DXY: [当前] | 阻力 [数值] (+X%) | 支撑 [数值] (-Y%)
- 黄金: [当前] | 阻力 [数值] | 支撑 [数值]
- 原油: [当前] | 期货曲线: Contango/Backwardation

### 固定收益
{{fixed_income_summary}}

**收益率曲线：**
- 2s10s: [数值]bp (vs 上周 [数值]bp) | 趋势: 陡峭化/平坦化
- 实际收益率: [数值]% | 通胀预期: [数值]%

---

## 📅 Macro Calendar（宏观日历）

{{macro_calendar_section}}

**重点关注：**
- [时间] | [国家] | [指标] | 预期 [数值] vs 前值 [数值] | 重要性: 🔴 高 / 🟡 中 / 🟢 低

---

## 🏛️ Central Bank Watch（央行动态）

{{central_bank_section}}

**政策预期：**
- 美联储: 下次会议 [日期] | 降息概率 [X%] (CME FedWatch)
- 欧央行: [观点]
- 中国央行: [观点]

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

## 🎯 Key Thresholds（关键阈值监控）

{{technical_levels_section}}

### 技术面关键位
| 标的 | 当前 | 关键阻力 | 距离 | 关键支撑 | 距离 | 突破概率 |
|------|------|----------|------|----------|------|----------|
| SPX | [数值] | [数值] | +X% | [数值] | -Y% | Z% |
| NDX | [数值] | [数值] | +X% | [数值] | -Y% | Z% |
| DXY | [数值] | [数值] | +X% | [数值] | -Y% | Z% |

### 宏观指标阈值
- **US 10Y**: [当前]% | 关键阈值 [数值]% | 距离 +/-Xbp | 触发后果: [描述]
- **VIX**: [当前] | 关键阈值 [数值] | 触发后果: [描述]
- **信用利差**: [当前]bp | 历史分位数 [X%] | 解读: [描述]

---

## 💡 Trading Strategy（交易策略）

### 今日重点关注
1. **[时间]** | **事件：** [描述] | **预期影响：** [正面/负面/中性] | **交易机会：** [具体操作]
2. **[时间]** | **事件：** [描述] | **预期影响：** [正面/负面/中性] | **交易机会：** [具体操作]

### 推荐交易设置

#### Trade #1: [标的名称]
- **方向：** Long/Short
- **入场点：** [价格区间] 或 [技术条件]
- **止损：** [价格] (-X%) | 触发条件: [描述]
- **目标：** [价格] (+Y%) | 分批止盈: [价格1] 50% / [价格2] 50%
- **仓位：** [X%] | 风险敞口: [金额]
- **胜率：** [X%] | 盈亏比: [R:1]
- **理由：** 
  - 基本面: [1句话]
  - 技术面: [1句话，包含具体指标]
  - 催化剂: [1句话]
- **风险：** [主要风险因素] | 失效条件: [具体指标/事件]
- **对冲：** [可选对冲方案]

#### Trade #2: [标的名称]
[同上结构]

#### Trade #3: [标的名称]
[同上结构]

### 组合风险管理
- **总仓位：** [X%] (现金 [Y%])
- **最大单日亏损：** -[X%] (触发减仓)
- **相关性风险：** [描述多头之间的相关性]
- **对冲比例：** [X%] (通过 [工具] 对冲)

---

## 📈 Chart Analysis（图表分析）

{{chart_analysis_section}}

---

## 🔮 Tomorrow's Focus（明日关注）

### 重要数据发布
- **[时间]** | [国家] | **[指标]** | 预期 [数值] vs 前值 [数值] | 重要性: 🔴/🟡/🟢
  影响: [正面/负面情景分析]

### 财报发布
- **[公司]** | [盘前/盘后] | EPS预期 $[数值] | 营收预期 $[数值]B
  关注点: [关键指标]

### 其他催化剂
- [事件描述] | 潜在影响: [描述]

### 情景分析
**情景1 (概率 X%):** 若 [条件] → 预期 [结果] → 交易策略: [操作]
**情景2 (概率 Y%):** 若 [条件] → 预期 [结果] → 交易策略: [操作]
**情景3 (概率 Z%):** 若 [条件] → 预期 [结果] → 交易策略: [操作]

---

## 📝 Disclaimer（免责声明）

本报告仅供专业投资者参考，不构成投资建议。市场有风险，投资需谨慎。
报告中的观点和预测基于当前可获得的信息，可能随市场变化而调整。
过往业绩不代表未来表现。请根据自身风险承受能力谨慎决策。

---

*Report generated at {{generation_time}}*  
*Data sources: Bloomberg, Wind, Reuters, Internal Models*  
*For institutional use only. Not for redistribution.*

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
    
    # 替换所有占位符为空字符串或实际内容
    # 如果 LLM 分析包含完整内容，则使用 LLM 内容；否则使用占位符提示
    
    # 插入各模块的格式化文本
    report = report.replace('{{macro_calendar_section}}', macro_data.get('formatted_text', '*宏观日历数据获取中...*'))
    report = report.replace('{{sector_news_section}}', sector_data.get('formatted_text', '*行业新闻数据获取中...*'))
    report = report.replace('{{market_movers_section}}', movers_data.get('formatted_text', '*市场异动数据获取中...*'))
    report = report.replace('{{risk_radar_section}}', risk_data.get('formatted_text', '*风险雷达数据获取中...*'))
    report = report.replace('{{chart_analysis_section}}', charts_section)
    
    # 替换其他占位符（如果 LLM 没有生成，则显示占位符）
    report = report.replace('{{us_equity_summary}}', '*等待 AI 分析...*')
    report = report.replace('{{europe_summary}}', '*等待 AI 分析...*')
    report = report.replace('{{asia_summary}}', '*等待 AI 分析...*')
    report = report.replace('{{fx_commodity_summary}}', '*等待 AI 分析...*')
    report = report.replace('{{fixed_income_summary}}', '*等待 AI 分析...*')
    report = report.replace('{{central_bank_section}}', '*等待 AI 分析...*')
    report = report.replace('{{technical_levels_section}}', '*等待 AI 分析...*')
    
    # 如果 LLM 分析成功，将其插入到报告开头（替换占位符部分）
    if llm_analysis and not llm_analysis.startswith('*AI 分析'):
        # LLM 应该生成完整的报告，直接使用
        # 但为了保留数据模块的内容，我们将 LLM 分析附加到末尾
        report += f"\n\n---\n\n## 🤖 AI Deep Analysis\n\n{llm_analysis}\n"
    else:
        # LLM 分析失败，显示错误信息
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
