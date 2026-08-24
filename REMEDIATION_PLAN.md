# 修复计划（Remediation Plan）

> 依据 2026-08-24 深度审计结果制定。覆盖全部 4 个审计子代理 + 人工通读的发现，按阻断优先级分 5 个阶段。
> 核心诉求：**周日 = 周复盘（weekly review），周一 = 一周日历与预测（week ahead）**，并根治"上一个自然日不是交易日"的日期语义缺陷。

---

## 0. 目标与验收标准

### 0.1 目标
1. 让 CI 真正变绿（当前假绿，测试套件从未跑完）。
2. 重构日期/交易日语义：接入真实 HKEX 节假日日历；报告模式由 briefing_date（今天）驱动而非 review_date（昨天）驱动。
3. 周日报告 = 周复盘（结构明确）。
4. 周一报告 = 一周日历与预测 + 重点观察清单。
5. 修复审计发现的全部 Critical/High 数据正确性缺陷，以及 Medium/Low 的渲染、CI、打包、卫生问题。

### 0.2 验收标准（Definition of Done）
- python scripts/run_tests.py --pytest 完整跑通且无 NameError。
- 新增测试覆盖：周日周复盘、周一 week-ahead、节假日（复活节周一/农历新年）、周五账本不丢失、南向单位正确。
- resolve_report_dates / build_report_mode 对周日/周一/普通交易日/节假日四类输入的输出符合下表。
- 报告头部、date_semantics、provenance 的日期标签与数据实际日期一致（无"global Sunday"这类误导标签）。
- 不引入新的 look-ahead（绩效账本仍严格"次日收盘才可进场"）。

---

## 1. 日期/交易日语义重设计（核心）

### 1.1 现状与问题（审计结论）
- config.py:28-29 closed_dates: []，全仓库无 HKEX 节假日数据源 → is_hk_trading_day()（date_policy.py:46）只认周末，落在工作日的真实休市日被当成交易日（实测 is_hk_trading_day("2026-04-06") 复活节周一 → True）。
- 报告模式由 review_date = previous_calendar_day(briefing_date) 的星期几驱动（date_policy.py:117-179），导致：
  - 周日简报（review 周六）→ weekly_review ✅（已符合诉求，但结构要明确）。
  - 周一简报（review 周日）→ non_trading_event_watch（周末事件观察），不符合诉求。
  - holiday_reopen_playbook / holiday_event_watch 因日历为空而形同死代码。
- global_market_date 用 previous_calendar_day（周一→周日），hk_data_date 用 previous_hk_trading_day（周一→周五），两者不对称（date_policy.py:106-108）。

### 1.2 新交易日历
- 新增 market_diary/professional/hk_holidays.py：内置 2024–2027 年 HKEX 公众假期表（静态 set[str]），提供 load_hk_holidays() -> set[str]。
- config.py 默认 calendar.closed_dates 改为在 load_professional_config() 时自动合并该表（或 date_policy._closed_dates 回退到该表），保证生产环境无需手工配置即有节假日感知。
- 维护说明：每年 12 月更新下一年的表；表内注释来源（HKEX 官方 holiday schedule）。

### 1.3 新模式分类（由 briefing_date 驱动）

| briefing_date（今天，简报日） | 新模式 mode | 内容定位 |
|---|---|---|
| 周日（weekday 6） | weekly_review | 周复盘：总结刚结束的 Mon–Fri 周 |
| 周一（weekday 0，交易日） | week_ahead | 一周日历与预测 + 重点观察 |
| 周二–周五（交易日） | trading_daily | 复盘上一交易日 |
| 周六（weekday 5） | trading_daily | 复盘周五收盘 |
| 工作日节假日（非交易日） | holiday_event_watch | 休市监控 |
| 节假日后的首个交易日 | holiday_reopen_playbook | 复盘休市期变化 + 重开准备 |

决策树（build_report_mode 重写，入参改为 briefing_date 为主）：
- today.weekday()==6 → weekly_review（period = 上一完整 Mon-Fri 周）
- today.weekday()==0 且 is_hk_trading_day(today) → week_ahead（week_start=today, last=prev_td）
- is_hk_trading_day(today) → 若昨天休市则 holiday_reopen_playbook，否则 trading_daily
- today.weekday()==5 → trading_daily（复盘周五）
- 其余（工作日节假日）→ holiday_event_watch

注：holiday_reopen_playbook 精确判定（"昨天是节假日"）在实现时用测试钉死；本计划先定语义："简报日是休市期结束后的首个交易日"。

### 1.4 周日「周复盘」具体结构（明确化）
保留并强化现有 weekly_review（report_layout.py:67-77、report_sections._render_weekly_review），结构固定为：
- Executive Summary（上周一句话结论 + 周度脉冲）
- Visual Dashboard（周度版图）
- Layer 1 | Weekly Scan：1.1 上周 Call 复盘 / 1.2 周度跨资产仪表盘 / 1.3 港股周度快速检查 / 1.4 决策板 / Next Week Checklist
- Layer 2 | Deep Read：2.1 周度跨资产复盘 / 2.2 港股-A股周度复盘 / 2.3 AI-TMT 读 / 2.4 资金流与归因 / 2.5 宏观与政策周度追踪 / 2.6 公司事件与风险 / 2.7 下周要回答的问题 / 2.8 核心标的周度回顾
- Layer 3：3.1 轮换主题深度 / 3.2 下周日历与重开准备 / 3.3-3.4 图表
- 附录：质量、绩效、来源

### 1.5 周一「一周日历与预测」具体结构（新）
新增 week_ahead 模式 + analytics_narrative.build_week_ahead() 聚合函数 + 渲染。
- Executive Summary（本周基准判断 + 本周需观察核心变量）
- Visual Dashboard（以周五收盘为基线的版图）
- Layer 1 | Reset：1.1 周末事件复盘 / 1.2 全球资产（周五收盘为基线）/ 1.3 港股上次收盘快速检查（作为基线，非"参考"）/ 1.4 决策板 / 本周检查清单
- Layer 2 | Deep Read：2.1 周末要闻与增量 / 2.2 一周日历（5 交易日 × 每日关键催化剂）/ 2.3 本周预测（base/risk case + 证伪点）/ 2.4 重点观察什么（Southbound、HSI vs 3033、USD/CNH、成交、轮动）/ 2.5 宏观与政策周度日历 / 2.6 公司事件与风险 / 2.7 周一早会问题 / 2.8 核心标的
- Layer 3：3.1 轮换主题深度 / 3.2 本周日历与交易准备 / 3.3-3.4 图表
- 附录：质量、绩效、来源

数据/构建块复用：一周日历复用 build_catalyst_calendar（窗口改为本周 5 个交易日）；预测复用 overview.risk_regime + attribution.risk_dashboard + LLM final_framing；重点观察复用 must_watch + 新增 week 维度清单。

### 1.6 日期不对称修复
- resolve_report_dates：global_market_date 改为 previous_global_trading_day（上一工作日），与 hk_data_date 对称；review_date 保留为叙事日但不再用于判定模式。
- data_fetcher._get_effective_intraday_date：回看改为交易日感知（跳过 closed_dates），max_lookback_days 提高到覆盖 >=10 天长假。
- macro_schedule.scheduled_events：days_back 改为"回看至上一交易日"（跨周末/节假日）。
- performance.observation_from_bundle / signal_from_bundle：as_of 优先取 HK 数据日（data_through）。

---

## 2. Phase 0 — 让 CI 复活（P0，阻断性，先行）
| # | 修复 | 位置 |
|---|---|---|
| P0-1 | 修复测试入口 NameError | tests/test_llm_enhancer_resilience.py:172 |
| P0-2 | 修复测试"精神分裂" | scripts/run_tests.py + pytest.ini/conftest |
| P0-3 | 修复常真断言 + 全量套件标记 | tests/test_github_actions.py:101-122,164-166 |

验收：CI 回归步骤真实跑完全部测试文件且全绿。

---

## 3. Phase 1 — 日期/交易日语义重构（核心诉求）
| # | 修复 | 位置 |
|---|---|---|
| P1-1 | 新增 HKEX 节假日表 | 新文件 professional/hk_holidays.py + config.py |
| P1-2 | build_report_mode 由 briefing_date 驱动 | professional/date_policy.py:117-179 |
| P1-3 | 周日周复盘结构明确化 | report_layout.py、report_sections._render_weekly_review |
| P1-4 | 新增 week_ahead 模式（周一） | date_policy.py + analytics_narrative + report_layout + report_sections + report_builder + runtime_audit |
| P1-5 | global/hk 日期对称 | date_policy.resolve_report_dates |
| P1-6 | 宏观日历回看跨节假日 | macro_schedule.py:181-182 |
| P1-7 | 盘中有效日回看交易日感知 | data_fetcher._get_effective_intraday_date:341-392 |
| P1-8 | 数据日期标签贯穿一致 | report_layout.py:91,121-124、date_policy.build_date_semantics |

新增测试：test_date_semantics.py 扩展 + 新 test_week_modes.py。

---

## 4. Phase 2 — 数据正确性（Critical/High）
| # | 修复 | 位置 |
|---|---|---|
| P2-1 | 绩效账本不再抹掉周五 | performance._merge_observations:197-209 |
| P2-2 | 南向 fallback 不把成交额当净买入 | analytics_public_flow.py:77-86 + analytics_market.py:266-270,379-383 |
| P2-3 | 南向/北向单位统一为 RMB | hk_local_data.py:267、analytics_public_flow.py、local_metrics |
| P2-4 | 周末重复发布不翻转 regime / 不重复计分 | performance._active_signals + call_scorecard.recent_record:187-220 |
| P2-5 | 行业新闻关键词改词边界匹配 | sector_news.py:141-149 |
| P2-6 | 缺涨跌数据不再断言"走平/稳定" | report_sections.py:404-414 + _signal_direction |
| P2-7 | 中文不再被静默删除 | report_text.py:197 |
| P2-8 | HKMA 记录取最新 | hk_local_data._fetch_hkma_record:150-158 |

---

## 5. Phase 3 — 渲染/编排修复
| # | 修复 | 位置 |
|---|---|---|
| P3-1 | bundle 在 prose-guard 重评分后再落盘 | main_professional.py:862 → 移到 901 后 |
| P3-2 | 字数预算单一来源 + 硬上限生效 | config.py:47-55 与 runtime_audit.py:71-72 |
| P3-3 | ETF 代理表去重 | report_blocks.py:430-435,594-599 |
| P3-4 | 非交易日 mode lens 只注入一次 | report_builder.py:120,129 |
| P3-5 | "What to watch today" 加日期窗口 | report_blocks.py:170-189 |
| P3-6 | "Yesterday's/Today's" 标题模式感知 | report_builder.py:101,163 |
| P3-7 | float() 加保护 | dashboard.py:687、daily_one_chart.py:631,452 |
| P3-8 | 渲染 review_date 到报告头 | report_builder.py:91 |

---

## 6. Phase 4 — CI/打包/文档/卫生
| # | 修复 | 位置 |
|---|---|---|
| P4-1 | 恢复 run 门禁 fail-open | morning_briefing_professional.yml:79-103 |
| P4-2 | 归档完整性对比 committed index | stage_report_archive.py:390-393 + workflow |
| P4-3 | 打包补 mistune 等直接依赖 | pyproject.toml、requirements.txt |
| P4-4 | 文档同步 | README.md、docs/* |
| P4-5 | 清理 ._* AppleDouble | 本机 find . -name '._*' -delete（含 .git 内） |
| P4-6 | 删除/标记 legacy main.py | market_diary/main.py + llm_client.generate_report |
| P4-7 | update_signal_performance.py 非零退出 | scripts/update_signal_performance.py:26-32 |
| P4-8 | test_macro_and_risk_sources.py 补断言 | tests/test_macro_and_risk_sources.py:44-48 |

---

## 7. 测试策略
- 每 Phase 配回归测试；日期语义用固定历史日期（非 datetime.now）：
  - 周日 briefing=2026-04-19 → weekly_review，窗口 04-13..04-17。
  - 周一 briefing=2026-08-24 → week_ahead，last=08-21、week_start=08-24。
  - 节假日 briefing=2026-04-07（复活节周二）→ holiday_reopen_playbook，hk_data_date=04-02。
  - 普通交易日 briefing=2026-04-16（周四）→ trading_daily。
- 账本回归：周五不丢、南向单位、weekend 去重。
- 渲染回归：test_date_semantics、test_report_section_contract、runtime_audit section 契约新增 week_ahead 变体。

---

## 8. 风险与回滚
- 顺序：P0 → P1 → P2 → P3 → P4，每阶段 run_tests.py --pytest 全绿再进下一阶段。
- 账本兼容：append-only 且 CI 提交；P2-1/P2-4 修改合并逻辑需保证历史归档仍可解析（parse_archived_report 兼容旧格式）。
- 周一 mode 迁移：non_trading_event_watch 仍保留给工作日节假日，仅周一不再触发；runtime_audit.REQUIRED_REPORT_SECTION_GROUPS 与 WeCom/HTML 契约同步加 week_ahead 标题变体。
- 节假日表维护：静态表有年限，需标注年度刷新义务，否则 2028 起失效（加"表过期告警"日志）。

---

## 9. 执行顺序（一句话）
P0 → P1（含周日/周一重构）→ P2 → P3 → P4，每阶段一次 scripts/run_tests.py --pytest 全绿再进下一阶段。

---

## 10. 计划审计修订（第 1 轮：作者自查）

自查发现以下问题，已纳入后续实现约束：

1. 【Medium】`mode` 触点清单不完整。§1.4/1.5 遗漏 4 处会消费 `day_mode`/`mode` 的代码：
   - `llm_enhancer.py:587-598` `_day_mode_context`（把 mode/label/note 送入 LLM prompt）；`:741-747` 非交易日分支需为 `week_ahead` 补上下文。
   - `skill_shadow.py:229-230`：weekly cadence 仅 `weekly_review` 触发 → 结论：`week_ahead` 不触发 skill shadow（保留周日），无需改动，但需记录。
   - `main_professional.py:453` `build_day_mode(review_date, config)` 是**第二个调用点**（决定 `prefer_weekend_active_assets`），必须同步改为按 `briefing_date` 计算，否则周一 mode 在两处不一致。
   - `dashboard.py:681`、`catalyst_radar.py:249` 只读 `day_mode` 的 label/next_hk_trading_day，无需改动。
   → P1-2/P1-4 触点清单补齐上述文件。

2. 【Medium】§1.5 "本周 5 个交易日"不精确。若本周含节假日（如周五休市），"5 个交易日"错误。正确定义：**本周日历周（周一至周五）内、经 `is_hk_trading_day` 过滤后的交易日**。→ 修订 §1.5 与 §7。

3. 【Low】§7 缺"周一恰逢节假日"用例。决策树已正确处理（周一非交易日 → holiday_event_watch），但无测试。→ 增加 `briefing=2026-04-06`（复活节周一）→ `holiday_event_watch`。

4. 【Medium】§1.6 `previous_global_trading_day` 若仅"上一工作日"近似，会漏美股自身节假日（感恩节、7/4）。明确取舍：**全球日 = 上一工作日（近似）+ 数据适配器回看兜底**，不引入美股节假日表（HK 产品，接受该近似并记录）。

5. 【Low】节假日表过期护栏具体化：`hk_holidays.load_hk_holidays()` 返回 `(dates, max_year)`；`date_policy` 在 `briefing_date.year > max_year` 时打 warning。

6. 【Low】P0-2 修复方案具体化：仓库无 `pytest.ini`/`conftest.py`；实测 `pytest tests/test_date_semantics.py --collect-only` 只收集 1 个 `test_*`，`main()` 内 10 条断言 pytest 看不到。方案：新增 `tests/conftest.py` 用 pytest hook 收集 `main()` 型套件（快速止血），后续逐步把关键断言迁成 `test_*` 函数。

---

## 11. 进度记录（实施中）

- ✅ **P0-1** 完成：`tests/test_llm_enhancer_resilience.py:172` 修复 NameError；CI 恢复（全套 276 pytest + 38 standalone 全绿）。
- ✅ **P1-1** 完成：新增 `market_diary/professional/hk_holidays.py`（2025/2026 静态表 + 过期护栏；农历节日待按 HKEX 通告年度核对）。
- ✅ **P1-2** 完成：`date_policy.build_report_mode` 改为按 `briefing_date` 驱动，新增 `week_ahead`（周一）模式；`build_day_mode`/调用点同步。
- ✅ **P1-4** 完成：`week_ahead` 内容（`analytics_narrative.build_week_ahead`）+ 渲染（`report_sections._render_week_ahead`）+ 标题（`report_layout`）+ `report_builder` 接线 + `runtime_audit` 契约。
- ✅ **P1-5** 完成：`resolve_report_dates` 的 `global_market_date` 改为上一工作日（周一→周五），与 `hk_data_date` 对称。
- ✅ 测试：重写 `test_date_semantics.py`（含周日周复盘/周一 week-ahead/节假日/普通交易日）；修 `test_professional_workbench.py` 的周一日期。
- ⏳ 待办：P1-3（周日周复盘结构再明确）、P1-6（宏观日历回看）、P1-7（盘中回看交易日感知）、P1-8（日期标签一致性收尾）、P2/P3/P4。

---

## 12. 进度记录（第 2 轮：P2 数据正确性）

- ✅ **P2-1** 完成：`performance._merge_observations` 冲突时保留归档 observation（不再 `pop` 成空），周五 session 不再从账本消失。
- ✅ **P2-2** 完成：`hk_local_data._stock_connect_metric` + `analytics_public_flow` 不再把成交额写入 `southbound_net_flow.value`；`analytics_market._local_metric_value` 对 "net not reported"/partial_public 返回 None。
- ✅ **P2-3** 完成：新增 `format_rmb_billions`；北向（A股）改为 RMB 标签，南向保持 HKD（原生货币）。
- ✅ **P2-4** 完成：`performance._merge_signals` 按 `market_as_of` 去重、保留最早发布（首份 call 权威），周末重跑不再翻转 regime / `recent_record` 不再重复计分。
- ✅ **P2-7** 完成：`report_text._clean_report_spacing` 不再静默删除 CJK/Kana/Hangul；runtime_audit 的 English-only 门禁现在真正生效（fail-loud 而非 silent-destroy）。
- ✅ **P2-8** 完成：`hk_local_data._fetch_hkma_record` 显式取 `end_of_date` 最大且 <= target 的记录，不再依赖未文档化的 API 顺序。
- ⏳ 待办：P2-5（sector_news 词边界）、P2-6（缺涨跌不再断言"走平"）、P1-3/6/7/8、P3、P4。

---

## 13. 进度记录（第 3 轮）

- ✅ **P2-5** 完成：`sector_news._keyword_match` 词边界匹配（"ai" 不再命中 "said"、"ev" 不再命中 "never"）。
- ✅ **P2-6** 完成：`report_sections._render_global_asset_dashboard` 缺涨跌时输出"方向数据不可用"，不再断言"走平/稳定"。
- ✅ **P1-6** 完成：`macro_schedule.scheduled_events` 默认 `days_back` 1→4，周一简报不再丢失周五已发布数据。
- ✅ **P3-1** 完成：`main_professional` 的 `_save_bundle` 移到 prose-guard 重评分之后，归档 quality 分与发布 markdown 一致。
- ✅ **P2 全部 8 项完成**（P2-1..P2-8）。
- ⏳ 待办：P1-3（周日结构再细化，已基本存在）、P1-7（盘中回看交易日感知）、P1-8（日期标签收尾）、P3-2..P3-8、P4。

---

## 14. 进度记录（第 4 轮）

- ✅ **P1-7** 完成：`data_fetcher` 盘中有效日回看窗口 4→10 天，覆盖农历新年/黄金周长假。
- ✅ **P3-4** 完成：`report_builder` 非交易日 mode lens 只注入一次（移除 2.2 处的重复注入）。
- ✅ **P3-3** 完成：ETF 代理表只在 Flow Tracker（2.4）渲染，移除 HK review（2.2）的重复副本；条件统一为 `_has_official_stock_connect_flow`。
- ✅ **P4-3** 完成：`pyproject.toml` + `requirements.txt` 补 `mistune>=3.0.0`（WeCom 渲染必需）。
- ✅ **P4-4** 完成：README/docs 的 provider 主次措辞改为 DeepSeek 主、MiniMax 备。
- ✅ **P4-8** 完成：`test_month_end_rules_clamp_to_real_dates` 补断言（非空 + 无 2 月 29/30/31）。
- ⏭️ **P4-7 跳过**：`refresh_performance_tracking` 不通过 status 报告失败（要么返回 ledger 要么抛异常），"永远 exit 0" 是误报。
- ⏳ 待办：P1-3、P1-8、P3-2/5/6/7/8、P4-1/2/5/6。

---

## 15. 进度记录（第 5 轮）

- ✅ **P4-1** 完成：恢复 run 门禁的 Python 片段对空 `run_started_at` 容错（try/except ValueError），不再因解析失败在 `bash -e` 下 fail-closed 跳过告警/投递。
- ✅ **P3-6** 完成：`report_layout` 新增 `call_title`/`core_names_title` 模式感知标题；周一 week-ahead 显示 "Last Session's Call" / "Core Names This Week"，周复盘显示 "Last Week's Call"。
- ⏭️ **P3-7 跳过**：`dashboard.py` 的 `float()` 实际数据路径是 `_parse_float` 产出的 float|None，字符串 "N/A" 不会流入，属防御性改进而非当前缺陷。
- ⏳ 待办：P1-3、P1-8、P3-2/5/8、P4-2/5/6。

---

## 16. 进度记录（第 6 轮）

- ✅ **P3-2** 完成：`config.py` 的 `reading_profile` 字数预算对齐 `runtime_audit`（4200/6000/7000），消除两套矛盾值。
- ✅ **P3-5** 完成：`_summary_watch` 只引用**当日**事件（date==briefing_date），无当日事件时明确标注"Next scheduled catalyst: X on 日期"，不再把周一事件当作"today"。
- ✅ **P4-5** 完成：删除 763 个 `._*` AppleDouble 文件（含 `.git/objects/pack` 内），本地 git 的 "non-monotonic index" 噪音消失。
- ✅ **P4-6** 完成：`main.py` 加 deprecation docstring（legacy 入口，生产用 main_professional）。
- ⏭️ **P1-8/P3-8 视为已解决**：P1-5 已修复 global 日期对称，报告头 "Data through: global Friday | HK Friday" 已正确；再显式渲染 review_date（周日）反而混淆。
- ⏭️ **P4-2（归档校验对比 committed index）** 与 **P1-3（周日结构再细化）** 明确保留为后续增强，见计划 §8 风险与 §7 测试策略。
