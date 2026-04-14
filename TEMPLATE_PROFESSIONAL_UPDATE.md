# 专业模板优化总结

## 优化内容

### 1. 移除所有 Emoji 表情符号

**原因：** 专业投行报告不应使用 emoji，应保持严肃、正式的风格

**修改位置：**
- 报告标题：`📊 Morning Briefing` → `Morning Briefing`
- 章节标题：移除所有 emoji（📌, 🌍, 📅, 🏛️, 🏢, 💹, ⚠️, 🎯, 💡, 📈, 🔮, 📝, 🤖）
- 宏观日历：移除颜色指示 emoji（🔴, 🟡, 🟢）改为文字（HIGH, MEDIUM, LOW, BEAT, MISS, INLINE）
- 市场情绪：移除 emoji（🟢, 🔴, 🟡）改为纯文字（Risk-On, Risk-Off, Neutral）

### 2. 添加 AI 模型信息

**新增字段：**
```markdown
**Investment Banking Research Institute · Strategy Research Department**

Report Time: 2026-04-14 10:45:33  
Analyst Team: Strategy Research  
AI Model: MiniMax-M2.7  ← 新增
```

**实现方式：**
- 在 `format_professional_report()` 函数中添加 `model_name` 参数
- 从环境变量 `LLM_MODEL` 读取模型名称
- 如果未设置，显示 "Not Specified"

### 3. 章节编号系统化

**原因：** 专业报告应使用罗马数字编号，便于引用和导航

**章节结构：**
```
I. Executive Summary
II. Market Snapshot
III. Macro Calendar
IV. Central Bank Watch
V. Sector & Stock News
VI. Pre-market Movers
VII. Risk Radar
VIII. Key Thresholds
IX. Trading Strategy
X. Chart Analysis
XI. Tomorrow's Focus
XII. AI Deep Analysis
```

### 4. 英文化标题和术语

**修改示例：**
- `核心观点` → `Executive Summary`
- `全球市场概览` → `Market Snapshot`
- `宏观日历` → `Macro Calendar`
- `央行动态` → `Central Bank Watch`
- `行业与个股` → `Sector & Stock News`
- `盘前异动` → `Pre-market Movers`
- `风险雷达` → `Risk Radar`
- `关键阈值监控` → `Key Thresholds`
- `交易策略` → `Trading Strategy`
- `图表分析` → `Chart Analysis`
- `明日关注` → `Tomorrow's Focus`
- `免责声明` → `Disclaimer`

### 5. 数据表格专业化

**宏观日历表格：**

之前：
```markdown
| 时间 | 国家 | 指标 | 实际值 | 预期 | 前值 | 偏差 |
| 20:30 | US | CPI MoM | 0.3% | 0.2% | 0.4% | 🟢 BEAT |
```

之后：
```markdown
| Time | Country | Indicator | Actual | Forecast | Prior | Deviation |
| 20:30 | US | CPI MoM | 0.3% | 0.2% | 0.4% | BEAT |
```

### 6. System Prompt 英文化

**原因：** 
- 与国际投行标准对齐
- 提高 LLM 理解准确度（英文训练数据更多）
- 便于跨国团队使用

**关键改进：**
- 所有指令改为英文
- 保持专业术语的准确性
- 示例输出更加规范

## 文件修改清单

### 修改的文件：

1. **market_diary/modules/report_template.py**
   - 更新 `get_professional_template()` 函数：移除 emoji，英文化标题
   - 更新 `format_professional_report()` 函数：添加 `model_name` 参数
   - 更新 `PROFESSIONAL_SYSTEM_PROMPT`：英文化所有指令

2. **market_diary/modules/macro_calendar.py**
   - 更新 `format_for_report()` 函数：移除 emoji，英文化表格标题

3. **market_diary/main_professional.py**
   - 更新报告组装部分：传递 `model_name` 参数

## 测试验证

### 测试命令：
```bash
cd market_diary
python main_professional.py --date 2026-04-14 --skip-charts
```

### 验证要点：
- ✅ 报告中无任何 emoji
- ✅ 显示 AI 模型名称（或 "Not Specified"）
- ✅ 章节使用罗马数字编号
- ✅ 所有标题英文化
- ✅ 宏观日历表格使用文字而非 emoji
- ✅ 整体风格专业、正式

## 示例输出

```markdown
# Morning Briefing | 2026-04-14 (Tuesday)

---

**Investment Banking Research Institute · Strategy Research Department**

Report Time: 2026-04-14 10:45:33  
Analyst Team: Strategy Research  
AI Model: MiniMax-M2.7

---

## I. Executive Summary

**Market Theme:** [一句话总结今日市场主题]

**Market Sentiment:** Risk-On / Risk-Off / Neutral | VIX: [数值] | Put/Call Ratio: [数值]

**Key Drivers:**
1. [驱动因素] → [影响] | Duration: Short/Medium/Long-term | Confidence: High/Medium/Low
...

## III. Macro Calendar

#### Released Data (Prior Day)

| Time | Country | Indicator | Actual | Forecast | Prior | Deviation |
|------|---------|-----------|--------|----------|-------|-----------|
| 20:30 | US | CPI MoM | 0.3% | 0.2% | 0.4% | BEAT |

#### Upcoming Data (Today)

| Time | Country | Indicator | Forecast | Prior | Importance |
|------|---------|-----------|----------|-------|------------|
| 20:30 | US | Retail Sales MoM | 0.3% | 0.6% | HIGH |
```

## GitHub Actions 配置

在 GitHub Actions 中，模型名称会自动从环境变量读取：

```yaml
env:
  MINIMAX_API_KEY: ${{ secrets.MINIMAX_API_KEY }}
  LLM_BASE_URL: https://api.minimaxi.com/v1
  LLM_MODEL: MiniMax-M2.7  # 会显示在报告中
```

## 对比总结

| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| Emoji | 大量使用 | 完全移除 |
| 章节编号 | 仅 emoji | 罗马数字 I-XII |
| 标题语言 | 中英混合 | 纯英文 |
| 模型信息 | 无 | 显示模型名称 |
| 表格指示 | Emoji | 文字（HIGH/BEAT等）|
| 整体风格 | 偏休闲 | 专业正式 |

## 下一步建议

1. **数据源对接：** 接入真实的 Bloomberg/Wind API
2. **图表优化：** 使用更专业的图表样式（去除彩色，使用灰度）
3. **PDF 导出：** 添加 PDF 生成功能，便于打印和分发
4. **多语言支持：** 根据需要添加中文版本（但保持无 emoji）
5. **自定义模板：** 允许用户自定义报告结构和字段
