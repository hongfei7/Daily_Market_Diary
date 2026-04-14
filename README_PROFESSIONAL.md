# 投行研究院专业晨报系统

## 📋 项目简介

这是一个专为头部券商投行研究院设计的自动化晨报（Morning Briefing）生成系统。系统整合全球市场数据、宏观经济日历、行业新闻、资金流向等多维度信息，结合 AI 分析，生成符合专业投资者需求的每日晨报。

## 🎯 核心功能

### 1. 市场概览（Market Snapshot）
- **全球市场仪表盘**：隔夜美股、欧洲、亚太市场表现
- **外汇与大宗商品**：美元指数、主要货币对、黄金/原油/铜价格
- **固定收益**：美债收益率曲线、利差变化
- **波动率指标**：VIX、MOVE 指数
- **信用利差**：IG/HY 利差变化
- **北向资金**：A股外资流向（如适用）

### 2. 宏观经济日历（Macro Calendar）
- **已公布数据**：昨日经济数据及其与预期的偏差分析
- **待公布数据**：今日重要经济数据发布时间表
- **央行动态**：央行会议、官员讲话安排
- **政策变化**：货币政策、财政政策最新动向

### 3. 央行观察（Central Bank Watch）
- 美联储、欧央行、日央行、中国央行最新表态
- 会议纪要摘要
- 降息/加息概率变化（CME FedWatch）
- 政策路径预期

### 4. 行业与个股要闻（Sector & Stock News）
- **按行业分类**：科技、金融、医疗、能源、消费等
- **重大事件**：并购交易、财报发布、监管政策
- **分析师评级**：Upgrade/Downgrade 及目标价调整
- **财报日历**：今日盘前/盘后财报发布

### 5. 盘前异动（Pre-market Movers）
- **涨跌幅榜**：盘前涨跌幅最大的个股及原因
- **ETF 资金流向**：主要 ETF 净流入/流出排名
- **期权异动**：Unusual Options Activity
- **大宗交易**：A股龙虎榜数据（如适用）

### 6. 风险雷达（Risk Radar）
- **地缘政治风险**：冲突、制裁、贸易摩擦
- **重大事件日历**：FOMC 会议、期权到期日、IPO、解禁
- **技术面关键位**：主要指数支撑/阻力位
- **市场情绪指标**：AAII 情绪、恐惧贪婪指数、Put/Call Ratio

### 7. 交易策略（Trading Strategy）
- **今日策略建议**：做多/做空/观望的具体标的
- **交易设置**：入场点、止损、目标价、仓位建议
- **对冲方案**：风险对冲建议
- **情景分析**：不同市场情景下的应对策略

## 🏗️ 系统架构

```
market_diary/
├── main_professional.py          # 专业版主程序
├── main.py                        # 原学生版主程序（保留）
├── modules/
│   ├── data_fetcher.py           # 市场数据获取
│   ├── chart_features.py         # 图表特征提取
│   ├── llm_client.py             # LLM 客户端
│   ├── macro_calendar.py         # 宏观日历模块 ✨ 新增
│   ├── sector_news.py            # 行业新闻模块 ✨ 新增
│   ├── market_movers.py          # 市场异动模块 ✨ 新增
│   ├── risk_radar.py             # 风险雷达模块 ✨ 新增
│   └── report_template.py        # 专业报告模板 ✨ 新增
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥（环境变量）
export MINIMAX_API_KEY="your_api_key_here"
export LLM_BASE_URL="https://api.minimaxi.com/v1"  # 可选
export LLM_MODEL="MiniMax-M2.7"  # 可选
```

### 2. 生成晨报

```bash
# 生成昨天的晨报（默认）
python market_diary/main_professional.py

# 生成指定日期的晨报
python market_diary/main_professional.py --date 2026-04-13

# 指定输出目录
python market_diary/main_professional.py --output-dir reports_pro

# 跳过图表生成（加快测试）
python market_diary/main_professional.py --skip-charts

# 调试模式（保存中间数据）
python market_diary/main_professional.py --debug
```

### 3. 查看报告

生成的晨报保存在 `reports_professional/` 目录下：
- 主报告：`YYYY-MM-DD_morning_briefing.md`
- 图表：`charts/` 子目录
- 图表特征：`charts/features_YYYY-MM-DD.json`

## 📊 数据源配置

### 当前支持的数据源

1. **市场数据**：Yahoo Finance (yfinance)
2. **新闻源**：Reuters, Bloomberg RSS, CNBC, WSJ
3. **宏观日历**：需要接入 Trading Economics / Investing.com API
4. **央行数据**：需要接入 Bloomberg / Reuters API
5. **期权数据**：需要接入 Unusual Whales / Market Chameleon API

### 生产环境数据源升级建议

为了达到投行级别的数据质量，建议接入以下专业数据源：

#### 必备数据源
- **Bloomberg Terminal API**：全面的市场数据、新闻、分析
- **Wind 万得**：中国市场数据（A股、债券、宏观）
- **Refinitiv Eikon**：全球市场数据、新闻

#### 可选数据源
- **Trading Economics API**：宏观经济日历
- **FactSet**：财务数据、分析师评级
- **S&P Capital IQ**：公司基本面数据
- **Unusual Whales**：期权流数据
- **CME FedWatch**：美联储政策概率

### 数据源接入示例

```python
# 在 modules/macro_calendar.py 中接入 Trading Economics
import tradingeconomics as te

te.login('your_api_key')
calendar = te.getCalendarData(
    country='United States',
    initDate='2026-04-13',
    endDate='2026-04-13'
)
```

## 🎨 报告定制

### 修改报告模板

编辑 `modules/report_template.py` 中的 `get_professional_template()` 函数：

```python
def get_professional_template(date: str) -> str:
    template = f"""
    # 📊 Morning Briefing | {date}
    
    > **您的机构名称 · 策略研究部**
    
    ## 您的自定义章节
    ...
    """
    return template
```

### 调整 LLM 分析风格

修改 `modules/report_template.py` 中的 `PROFESSIONAL_SYSTEM_PROMPT`：

```python
PROFESSIONAL_SYSTEM_PROMPT = """
您是 [您的机构] 的首席策略分析师...
[自定义指令]
"""
```

### 添加自定义数据模块

1. 在 `modules/` 下创建新模块，例如 `custom_data.py`
2. 实现数据获取函数
3. 在 `main_professional.py` 中导入并调用
4. 在报告模板中添加对应章节

## 🔧 高级配置

### 1. 多语言支持

系统默认使用中文，可以通过修改模板和 prompt 支持英文或其他语言。

### 2. 定时自动生成

使用 cron（Linux/Mac）或任务计划程序（Windows）：

```bash
# Linux/Mac crontab 示例（每天早上 6:00 生成）
0 6 * * * cd /path/to/project && python market_diary/main_professional.py
```

### 3. 邮件/企业微信推送

在 `main_professional.py` 末尾添加推送逻辑：

```python
# 生成报告后
send_email(
    to='team@yourcompany.com',
    subject=f'Morning Briefing | {report_date}',
    body=final_report
)
```

### 4. 性能优化

- 使用缓存减少重复 API 调用
- 并行获取多个数据源
- 使用更快的 LLM 模型（如 GPT-4-turbo）

## 📈 与原版本对比

| 功能 | 学生版 (main.py) | 专业版 (main_professional.py) |
|------|------------------|-------------------------------|
| 市场数据 | ✅ 基础价格 | ✅ 全面市场概览 |
| 图表分析 | ✅ 6张图表 | ✅ 6张图表 + 特征提取 |
| 宏观日历 | ❌ | ✅ 经济数据 + 央行事件 |
| 行业新闻 | ⚠️ 简单RSS | ✅ 分类聚合 + 财报 + 评级 |
| 市场异动 | ❌ | ✅ 盘前异动 + ETF流向 + 期权 |
| 风险监控 | ❌ | ✅ 地缘政治 + 事件日历 + 技术位 |
| 报告结构 | ⚠️ 学术风格 | ✅ 投行专业风格 |
| 交易策略 | ⚠️ 理论分析 | ✅ 可执行交易设置 |

## 🛠️ 故障排查

### 问题：API 调用失败

```bash
# 检查 API 密钥
echo $MINIMAX_API_KEY

# 测试 API 连接
python -c "from modules.llm_client import get_client; client = get_client(); print('OK')"
```

### 问题：数据获取超时

- 检查网络连接
- 增加超时时间（在 `data_fetcher.py` 中修改）
- 使用代理（如需要）

### 问题：图表生成失败

- 确保安装了 matplotlib
- 检查数据是否为空
- 使用 `--skip-charts` 跳过图表生成

## 📝 开发路线图

### v2.0 计划功能
- [ ] 接入 Bloomberg Terminal API
- [ ] 接入 Wind 万得 API
- [ ] 实时数据流（WebSocket）
- [ ] 多语言支持（中英双语）
- [ ] Web 界面
- [ ] 移动端推送
- [ ] 历史报告检索
- [ ] 自定义指标监控

### v3.0 计划功能
- [ ] 机器学习预测模型
- [ ] 情绪分析（社交媒体）
- [ ] 量化回测框架
- [ ] 多策略组合优化
- [ ] 实时风险监控仪表盘

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发规范
1. 代码风格：遵循 PEP 8
2. 文档：所有函数必须有 docstring
3. 测试：添加单元测试
4. 提交信息：使用清晰的 commit message

## 📄 许可证

本项目仅供学习和研究使用。商业使用请联系作者获取授权。

## ⚠️ 免责声明

本系统生成的报告仅供参考，不构成投资建议。
市场有风险，投资需谨慎。
使用本系统进行投资决策的风险由使用者自行承担。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Email: [your-email@example.com]

---

**祝您使用愉快！Happy Trading! 📈**
