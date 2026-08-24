# 周一（Week-Ahead）报告设计文档

面向：香港投行研究所新人分析师，早上港深通勤、早会前用（5 分钟手机扫 + 10-15 分钟深读）。
目标：周一报告回答"这一周怎么开局、看什么、什么会改变我的判断"，而不是重播周五。

---

## 1. 数据源清单（bundle 里可用）

| 数据源 | 结构 | 用途 |
|---|---|---|
| `market_summary` | 跨资产收盘价/涨跌（美股/港股/A股/FX/商品/债/加密） | 周五收盘基线表 |
| `hk_local` / `stock_connect` / `ah_premium` | 南向/北向、成交、沽空、AH 溢价、HIBOR | 港股基线检查 |
| `macro_agenda` | 规则化宏观日历（CN LPR/PMI/CPI、US CPI/NFP、HK CPI，~12 条，带日期） | 本周宏观日历 |
| `catalysts` | 已聚合的带日期事件（宏观+财报+政策+watchlist 催化剂，7 天窗） | 本周日历表 |
| `company_events.earnings` | 财报日历（日期/公司/EPS/营收） | 本周财报 |
| `company_events.announcements` | HKEX 公告（watchlist 命中 + 高优先） | 公司事件 |
| `risk.upcoming_events` | 政策/风险事件 | 政策事件 |
| `risk.geopolitical_risks` / `sector_digest.graded_news` | 周末新闻/地缘 | 周末要闻 |
| `high_frequency` | 高频跟踪（DXY/10Y/SOXX/VIX…带 HK 传导权重） | 仍在动的资产 |
| `overview` / `attribution` | 风险得分/regime/主导驱动 | 基准/风险情景 |
| `hk_desk_view` | 港股风格/确认/证伪 | 观察清单的证伪条件 |
| `watchlists` | 核心/焦点/学习池（thesis + upcoming_catalyst） | 核心标的 |

## 2. 内容结构（周一专属）

```
# Header（mode + 三市场数据日期）

## Executive Summary（4 个固定问题，周维度）
- 本周基调（base case 一句话，来自 overview.risk_regime + attribution）
- 周五收盘基线（HSI/3033/USD-CNH 一句话）
- 本周最关键催化剂（日历里最高分事件，日期+事件）
- 周末要闻（1 条最相关的周末新闻）

## This Week at a Glance（核心，紧接 Exec Summary）
### 本周日历表（5 交易日 × 类别）
### 基准 / 风险情景（2 行）
### 重点观察清单（表格：观察什么 | 为什么 | 证伪条件）

## Visual Dashboard + Catalyst Radar（图表）

## Layer 1 | Reset (5 min)
### 1.1 上周 Call 复盘（一句话结论 + 近期命中率）
### 1.2 周五收盘基线表（紧凑，8 个关键资产）
### 1.3 港股基线快速检查
### 1.4 决策板
**本周检查清单**（must_watch 前 4）

## Layer 2 | Deep Read (10-15 min)
### 2.1 本周宏观日历（周维度宏观表）
### 2.2 公司事件与财报（周维度）
### 2.3 早会问题
### 2.4 核心标的（周维度）

## 附录（质量/绩效/来源）
```

**与交易日的差异**：去掉隔夜复盘、港股详细复盘、AI/TMT 链、资金流详表、主题深挖；把"周五收盘"从详细复盘降为**紧凑基线表**；把"本周日历/情景/观察清单"提升为**核心**。

## 3. 信息源提取逻辑

### 3.1 本周日历表（核心表）
- 输入：`catalysts`（已聚合）+ `macro_agenda`（补漏）+ `company_events.earnings`（财报）+ `risk.upcoming_events`（政策）。
- 过滤：`week_start <= date <= week_end`（周一~周五）。
- 分组：按日（5 个交易日），每行 4 列类别：**宏观 | 财报 | 政策 | 公司**。
- 去重：按 (date, event 规范化) 去重（复用 `analytics_briefing._dedupe_key`）。
- 排序：按 score 降序；无事件的日子标"无已排期催化剂"（诚实，不推断日历安静）。

### 3.2 基准/风险情景
- base_case / risk_case 由 `overview.risk_regime` 三分支模板生成（现有 `build_week_ahead` 已有雏形）。
- 加一行"关键假设"，指向本周最关键证伪点（`hk_desk_view.invalidation` 或首个催化剂）。

### 3.3 重点观察清单（表格）
- 行：南向资金 / HSI vs 3033 风格 / USD-CNH / 成交 vs 20D / 首个催化剂。
- 每行三列：**观察什么 | 为什么（transmission）| 证伪条件**。
- 证伪条件来自 `hk_desk_view.invalidation` + 确定性模板（不能是空话）。

### 3.4 周末要闻
- 输入：`sector_digest.graded_news`（A/B 级）+ `risk.geopolitical_risks` + `high_frequency`（仍在动的 FX/商品/加密）。
- 表格：Channel | Signal | Why。

### 3.5 周五收盘基线表
- 8 个关键资产：HSI、3033.HK、USD/CNH、USD/HKD、US 10Y、DXY、Gold、VIX。
- 列：资产 | 周五收盘 | 涨跌。来自 `market_summary`（`build_market_snapshot`）。

## 4. 图表方案

| 图表 | 周一用？ | 说明 |
|---|---|---|
| **Catalyst Radar** | ✅ 保留 | 本来就是周维度催化队列，周一核心图 |
| **Dashboard** | ✅ 保留 | 周五市场状态全景（基线可视化） |
| Daily One Chart | ❌ 跳过 | 单日盘中图，与周维度不匹配 |
| AI/TMT 链图 | ❌ 跳过 | 单日读，周一冗余 |
| HK Trend Pack | ❌ 跳过 | 仅周日周复盘用 |

实现：`main_professional.py` 中 `should_render_daily_chart` / ai_tmt_chart 在 `week_ahead` 模式置 False。

## 5. 长度与编排

- 周一目标 ~2500-3500 词（比交易日 4200-6000 更短；`runtime_audit` 对 week_ahead 用更低字数下限）。
- 顺序：结论（Exec）→ 本周一览（日历/情景/观察）→ 基线（Layer 1）→ 深读（宏观/公司）→ 附录。
- 冗余规则：周五数据只出现在"基线表"一次，不再在别处重复。

## 6. LLM 分工（层层拆解，现有架构）

- `news_selection`（周维度）：从周末新闻里选本周最相关的 3-5 条。
- `macro_interpretation`：解读本周宏观日历（不再是单日）。
- `final_framing`：产出"本周基调"一句话 + 基准/风险情景（喂给 Exec Summary）。
- `overnight_review` / `hk_review` / `company_commentary`：周一降权或跳过（避免重播周五）。

## 7. 待审计点

1. 本周日历表"无已排期催化剂"是否真的诚实（宏观规则稀疏时会不会大面积空表）？
2. 三市场日期（US/HK/A股）在周一日历里是否体现（例如某天港股休市但美股开市）？
3. 观察清单的"证伪条件"是否有可靠来源（会不会是空话）？
4. 图表跳过（Daily/AI-TMT）会不会导致周一报告缺图？
5. LLM 周维度任务（news/macro/final_framing）的 prompt 是否要单独为 week_ahead 写？

---

## 8. 自查修订（第 1 轮：数据源实测）

用空数据 bundle 实测了各字段，发现原设计有几处对数据可用性过于乐观：

1. **财报日历实际为空**：`company_events.earnings` 恒为 `[]`（无财报 feed 配置）。且 `build_company_event_digest` 的 earnings 行**没有 `date` 字段**（只有 `as_of`）。→ 本周日历的"财报"列基本是空的，不能靠它撑起一列。带日期的财报事件其实在 `catalysts`（`build_catalyst_calendar` 从原始 `earnings_calendar` 提取 `date`）。→ 修订：本周日历改为**"本周关键事件"排名列表**（date + 类别 + 影响），不搞会大面积空格的 4 列表。

2. **宏观规则稀疏**：`macro_schedule` 只有 ~12 条月频规则（每条一月一次），分摊到某周通常只有 2-5 条。→ 5 日 × 4 列的表会大半是"无事件"。→ 修订：**先列"本周关键事件"**（2-5 条，带日期和为什么重要），再用一个紧凑的 Mon-Fri 条带显示哪几天有事件。

3. **`risk.upcoming_events` 与 `macro_agenda` 重复**：都来自同一个 `scheduled_events` 规则。→ 修订：去重，本周日历只取一份宏观。

4. **证伪条件来源可靠**：`hk_desk_view.invalidation` 存在且非空（如 "Any style claim remains invalid while the required relative-performance fields are absent"）。→ 观察清单的"证伪"列可落地。

5. **LLM 管道已具备 day_mode 上下文**：`_build_task_context` 已注入 `_day_mode_context`（mode/label/note/report_horizon），`week_ahead` 会传到各任务。→ 周维度 prompt 只需微调措辞，无需重构管道。

6. **三市场逐日开闭可做**：`date_policy` 已有 `is_hk/us/cn_trading_day`。→ 本周日历每行可标注"HK 休市 / 美股开市 / A股开市"。

## 9. 修订后的核心表设计

**T1 本周关键事件（排名列表，~5 条）**：列 `日期 | 类别 | 事件 | 为什么重要`。数据源 = `catalysts`（已含宏观+财报+政策+watchlist，去重）。

**T2 本周日历（紧凑条带）**：Mon-Fri 5 列，每列显示"有事件的日子标事件数 + 三市场开闭状态"，无事件标"无"。

**T3 周五收盘基线（8 资产）**：`资产 | 收盘 | 涨跌`，来自 `build_market_snapshot`。

**T4 重点观察清单**：`观察什么 | 为什么（transmission）| 证伪条件`，证伪来自 `hk_desk_view.invalidation` + 模板。

**T5 周末要闻**：`渠道 | 信号 | 为什么`，来自 graded_news + geopolitical + still-moving。
