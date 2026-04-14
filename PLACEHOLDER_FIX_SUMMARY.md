# 占位符替换问题修复总结

## 问题描述

生成的晨报中出现未替换的占位符，例如：
- `{report_time}`
- `{macro_calendar_section}`
- `{us_equity_summary}`
- `{generation_time}`

## 根本原因

模板使用了 Python f-string（`f"""..."""`），当模板中包含占位符如 `{placeholder}` 时，Python 会尝试立即解析这些变量，导致：
1. 如果变量不存在，会抛出 `NameError`
2. 如果使用双花括号 `{{placeholder}}`，f-string 会将其转换为单花括号 `{placeholder}`，但后续的 `.replace()` 调用会查找双花括号，导致替换失败

## 解决方案

### 修改 1: 模板定义（`report_template.py`）

**之前（错误）：**
```python
template = f"""# 📊 Morning Briefing | {date} ({weekday})
报告时间：{{report_time_placeholder}}
"""
```

**之后（正确）：**
```python
template = """# 📊 Morning Briefing | DATE_PLACEHOLDER (WEEKDAY_PLACEHOLDER)
报告时间：{report_time_placeholder}
"""

# 然后手动替换日期
template = template.replace('DATE_PLACEHOLDER', date)
template = template.replace('WEEKDAY_PLACEHOLDER', weekday)
```

### 修改 2: 占位符替换（`format_professional_report()`）

**统一使用单花括号格式：**
```python
report = template.replace('{report_time_placeholder}', report_time)
report = report.replace('{generation_time_placeholder}', report_time)
report = report.replace('{macro_calendar_section_placeholder}', macro_data.get('formatted_text', '*宏观日历数据获取中...*'))
# ... 其他占位符
```

### 修改 3: 移除重复代码

删除了 `format_professional_report()` 函数末尾的重复代码块。

## 验证结果

运行测试后，生成的报告正确显示：

```markdown
# 📊 Morning Briefing | 2026-04-14 (Tuesday)

> **投行研究院 · 策略研究部**  
> 报告时间：2026-04-14 10:35:58  ✅ 正确替换
> 分析师：策略团队

---

## 🌍 Market Snapshot（全球市场概览）

### 隔夜美股
*等待 AI 分析...*  ✅ 正确显示fallback消息

---

*Report generated at 2026-04-14 10:35:58*  ✅ 正确替换
```

## 如何测试

```bash
# 进入项目目录
cd market_diary

# 运行专业版晨报生成器（跳过图表以加快测试）
python main_professional.py --date 2026-04-14 --skip-charts

# 检查生成的报告
cat reports_professional/2026-04-14_morning_briefing.md | head -20

# 或在 Windows 上
Get-Content reports_professional/2026-04-14_morning_briefing.md | Select-Object -First 20
```

## 相关文件

修改的文件：
1. `market_diary/modules/report_template.py` - 模板定义和替换逻辑
2. `TROUBLESHOOTING.md` - 添加了占位符问题的排查指南

## 注意事项

1. **不要使用 f-string 定义包含占位符的模板**
   - f-string 会立即解析 `{variable}`
   - 使用普通字符串，然后用 `.replace()` 方法

2. **占位符命名规范**
   - 使用描述性名称：`{report_time_placeholder}` 而不是 `{time}`
   - 保持一致性：所有占位符都使用 `_placeholder` 后缀

3. **测试建议**
   - 每次修改模板后，运行一次测试生成
   - 检查生成的报告前后部分，确保所有占位符都被替换

## GitHub Actions 注意事项

在 GitHub Actions 中运行时，需要确保：
1. 环境变量 `MINIMAX_API_KEY` 已设置（用于 AI 分析）
2. 如果 API 不可用，系统会显示 fallback 消息，不会导致脚本失败
3. 占位符替换与 API 可用性无关，即使 AI 分析失败，占位符也会被正确替换

## 下一步

问题已修复！您现在可以：
1. 在 GitHub Actions 中运行工作流
2. 本地测试生成报告
3. 所有占位符都会被正确替换为实际数据或 fallback 消息
