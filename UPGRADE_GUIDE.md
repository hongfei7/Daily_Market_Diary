# 从学生版升级到专业版指南

## 📊 核心改进对比

### 1. 报告结构升级

#### 学生版结构
```
# Market Diary: 2026-04-13
├── AI Macro Analysis
│   ├── Chart read
│   ├── One-line takeaway
│   ├── Market tape (Asia/Europe/US)
│   ├── Cross-asset dashboard
│   ├── What changed the narrative
│   ├── Rates & USD
│   ├── Flows & positioning
│   ├── Trading Plan
│   └── What to watch tomorrow
└── Charts (6张图表)
```

#### 专业版结构
```
# Morning Briefing | 2026-04-13
├── 📌 Executive Summary（核心观点）
│   ├── 市场主题
│   ├── 关键驱动因素
│   ├── 今日策略建议
│   └── 风险提示
├── 🌍 Market Snapshot（全球市场概览）
│   ├── 隔夜美股
│   ├── 欧洲市场
│   ├── 亚太市场
│   ├── 外汇与大宗商品
│   └── 固定收益
├── 📅 Macro Calendar（宏观日历）✨ 新增
│   ├── 已公布数据及偏差分析
│   ├── 今日待公布数据
│   └── 央行官员讲话安排
├── 🏛️ Central Bank Watch（央行动态）✨ 新增
│   ├── 美联储/欧央行/日央行/中国央行
│   ├── 会议纪要摘要
│   └── 降息/加息概率
├── 🏢 Sector & Stock News（行业与个股）✨ 升级
│   ├── 按行业分类新闻（8大行业）
│   ├── 财报日历
│   └── 分析师评级调整
├── 💹 Pre-market Movers（盘前异动）✨ 新增
│   ├── 盘前涨跌幅榜
│   ├── ETF 资金流向
│   ├── 期权市场异动
│   └── A股大宗交易
├── ⚠️ Risk Radar（风险雷达）✨ 新增
│   ├── 地缘政治风险
│   ├── 重大事件日历
│   ├── 技术面关键位
│   └── 市场情绪指标
├── 💡 Trading Strategy（交易策略）✨ 升级
│   ├── 今日重点关注
│   ├── 推荐交易设置（具体入场/止损/目标）
│   └── 对冲建议
├── 📊 Chart Analysis（图表分析）
│   └── 6张专业图表 + 特征提取
└── 🔮 Tomorrow's Focus（明日关注）
    ├── 重要数据发布
    ├── 财报发布
    └── 其他催化剂
```

### 2. 数据源升级

| 数据类型 | 学生版 | 专业版 |
|---------|--------|--------|
| 市场价格 | Yahoo Finance | Yahoo Finance + 支持 Bloomberg/Wind 接入 |
| 新闻源 | 3个 RSS | 5+ RSS + 行业分类 + 重要性评分 |
| 宏观数据 | ❌ 无 | ✅ 经济日历 + 央行事件 |
| 财报数据 | ❌ 无 | ✅ 财报日历 + EPS 预期 |
| 分析师评级 | ❌ 无 | ✅ Upgrade/Downgrade 追踪 |
| ETF 流向 | ❌ 无 | ✅ 13个主要 ETF 流向监控 |
| 期权数据 | ❌ 无 | ✅ Unusual Options Activity |
| 风险事件 | ❌ 无 | ✅ 地缘政治 + 事件日历 |
| 技术位 | ❌ 无 | ✅ 4个主要指数关键位 |
| 情绪指标 | ❌ 无 | ✅ AAII + 恐惧贪婪 + Put/Call |

### 3. 分析深度升级

#### 学生版分析
- 偏学术化，理论分析为主
- 缺少具体交易建议
- 风险提示不够具体
- 没有情景分析

#### 专业版分析
- 投行实战风格，直击要点
- 具体交易设置（入场/止损/目标/仓位）
- 明确风险提示和对冲方案
- 多情景分析（if-then 逻辑）
- 时间敏感性强（开盘前必读）

### 4. 代码架构升级

#### 学生版（单文件）
```python
main.py (800+ 行)
├── 数据获取
├── 图表生成
├── LLM 分析
└── 报告组装
```

#### 专业版（模块化）
```python
main_professional.py (主程序)
├── modules/
│   ├── data_fetcher.py (市场数据)
│   ├── macro_calendar.py (宏观日历) ✨
│   ├── sector_news.py (行业新闻) ✨
│   ├── market_movers.py (市场异动) ✨
│   ├── risk_radar.py (风险雷达) ✨
│   ├── report_template.py (专业模板) ✨
│   ├── chart_features.py (图表特征)
│   └── llm_client.py (LLM 客户端)
```

## 🚀 迁移步骤

### Step 1: 安装新依赖

```bash
pip install -r market_diary/requirements.txt
```

新增依赖：
- `beautifulsoup4` - 网页解析
- `python-dateutil` - 日期处理

### Step 2: 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入 API 密钥
nano .env
```

### Step 3: 测试运行

```bash
# 使用快速启动脚本（推荐）
./run_morning_briefing.sh

# 或直接运行 Python
python market_diary/main_professional.py --date 2026-04-13
```

### Step 4: 对比输出

```bash
# 生成学生版报告
python market_diary/main.py --date 2026-04-13

# 生成专业版报告
python market_diary/main_professional.py --date 2026-04-13

# 对比两个报告
diff reports/2026-04-13.md reports_professional/2026-04-13_morning_briefing.md
```

## 📝 定制化建议

### 1. 修改机构名称

编辑 `modules/report_template.py`：

```python
def get_professional_template(date: str) -> str:
    template = f"""# 📊 Morning Briefing | {date}

> **[您的机构名称] · 策略研究部**  
> 报告时间: {{report_time}}  
> 分析师: [您的团队名称]
```

### 2. 调整行业覆盖

编辑 `modules/sector_news.py`：

```python
SECTORS = {
    'Technology': ['tech', 'software', 'AI'],
    'Financials': ['bank', 'insurance'],
    # 添加您关注的行业
    'New Energy': ['solar', 'wind', 'battery'],
}
```

### 3. 自定义技术位

编辑 `modules/risk_radar.py`：

```python
KEY_LEVELS = {
    'SPX': {
        'resistance': [6950, 7000, 7100],  # 根据实际调整
        'support': [6800, 6750, 6700],
    },
    # 添加您关注的标的
    'AAPL': {
        'resistance': [180, 185, 190],
        'support': [170, 165, 160],
    },
}
```

### 4. 接入专业数据源

#### 接入 Bloomberg Terminal

```python
# 在 modules/data_fetcher.py 中添加
import blpapi

def fetch_bloomberg_data(tickers, fields):
    session = blpapi.Session()
    session.start()
    # ... Bloomberg API 调用
    return data
```

#### 接入 Wind 万得

```python
# 在 modules/macro_calendar.py 中添加
from WindPy import w

w.start()
calendar_data = w.edb("M0017142", "2026-04-13", "2026-04-13")
```

## 🎯 生产环境部署

### 1. 定时任务设置

#### Linux/Mac (crontab)

```bash
# 编辑 crontab
crontab -e

# 添加定时任务（每天早上 6:00）
0 6 * * * cd /path/to/project && ./run_morning_briefing.sh >> logs/cron.log 2>&1
```

#### Windows (任务计划程序)

1. 打开"任务计划程序"
2. 创建基本任务
3. 触发器：每天 6:00
4. 操作：启动程序 `run_morning_briefing.bat`

### 2. 邮件推送

在 `main_professional.py` 末尾添加：

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_report(report_content, date):
    msg = MIMEMultipart()
    msg['From'] = os.getenv('SMTP_USER')
    msg['To'] = os.getenv('NOTIFICATION_EMAIL')
    msg['Subject'] = f'Morning Briefing | {date}'
    
    msg.attach(MIMEText(report_content, 'plain'))
    
    with smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT'))) as server:
        server.starttls()
        server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
        server.send_message(msg)

# 在 main() 函数末尾调用
send_email_report(final_report, report_date)
```

### 3. 企业微信推送

```python
import requests

def send_wechat_notification(report_summary, date):
    webhook = os.getenv('WECHAT_WEBHOOK')
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": f"# Morning Briefing | {date}\n\n{report_summary}"
        }
    }
    requests.post(webhook, json=data)
```

## 🔧 性能优化

### 1. 启用缓存

```python
# 在 modules/data_fetcher.py 中添加
import functools
import time

@functools.lru_cache(maxsize=128)
def fetch_market_data_cached(date):
    return fetch_market_data(date)
```

### 2. 并行数据获取

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_all_data_parallel(date):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            'market': executor.submit(fetch_market_data, date),
            'macro': executor.submit(fetch_macro_data, date),
            'sector': executor.submit(fetch_sector_data, date),
            'movers': executor.submit(fetch_movers_data, date),
            'risk': executor.submit(fetch_risk_data, {}),
        }
        
        results = {key: future.result() for key, future in futures.items()}
    return results
```

### 3. 使用更快的 LLM

```bash
# 切换到 GPT-4-turbo
export LLM_MODEL=gpt-4-turbo

# 或使用本地模型（需要 Ollama）
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3
```

## 📊 效果对比示例

### 学生版输出（节选）
```markdown
## 1) Market tape (session-by-session, Asia → Europe → US)

### Asia
- **What moved:** USD/JPY turned lower (-0.25pp net) after early Tokyo spike...
- **Why:** Asia session was largely positioned around weekend risk cleanup...
```

### 专业版输出（节选）
```markdown
## 🌍 Market Snapshot（全球市场概览）

### 隔夜美股
标普500收涨1.02%至6,875点，纳指涨1.06%。科技股领涨，高盛财报超预期推动金融板块。
VIX回落至19.12，信用利差收窄，风险偏好回升。

**驱动因素：**
1. 美债收益率下行（10Y -4.6bp）降低贴现率，提振成长股估值
2. 高盛Q1交易收入创纪录，验证金融板块盈利韧性
3. 美元走弱（DXY -0.61%）利好跨国公司盈利预期

**板块表现：** 科技 +1.5% | 金融 +1.2% | 能源 -0.8%
```

## 🎓 学习资源

### 推荐阅读
1. **《Macro Trading and Investment Strategies》** - Gabriel Burstein
2. **《The Art of Currency Trading》** - Brent Donnelly
3. **Bloomberg Terminal 使用指南**
4. **投行研究报告写作规范**

### 在线课程
- Coursera: Financial Markets (Yale)
- CFA Institute: Equity Research
- Bloomberg Market Concepts (BMC)

## ❓ 常见问题

### Q1: 专业版比学生版慢很多？
A: 专业版获取的数据源更多。可以：
- 使用 `--skip-charts` 跳过图表
- 启用缓存
- 并行获取数据

### Q2: 如何只生成某些章节？
A: 修改 `config_example.json` 中的 `sections` 配置。

### Q3: 可以生成英文报告吗？
A: 可以，修改 `report_template.py` 中的模板和 `PROFESSIONAL_SYSTEM_PROMPT`。

### Q4: 数据源 API 费用如何？
A: 
- Yahoo Finance: 免费
- Bloomberg Terminal: ~$2,000/月
- Wind 万得: ~¥10,000/年
- Trading Economics: $50-500/月

### Q5: 可以部署到云端吗？
A: 可以，支持：
- AWS Lambda + EventBridge
- Google Cloud Functions + Cloud Scheduler
- Azure Functions + Timer Trigger
- 阿里云函数计算 + 定时触发器

## 📞 技术支持

遇到问题？
1. 查看 `README_PROFESSIONAL.md`
2. 检查 GitHub Issues
3. 联系开发团队

---

**祝您升级顺利！🚀**
