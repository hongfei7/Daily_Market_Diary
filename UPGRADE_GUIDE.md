# Upgrade Guide: From Student Version to Professional Version

## Core Improvements Overview

### 1. Report Structure Upgrade

#### Student Version Structure
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
└── Charts (6 charts)
```

#### Professional Version Structure
```
# Morning Briefing | 2026-04-13
├── I. Executive Summary (Core views)
│   ├── Market Theme
│   ├── Key Drivers
│   ├── Strategy Recommendations
│   └── Risk Alert
├── II. Market Snapshot (Global market overview)
│   ├── Overnight US Equities
│   ├── European Markets
│   ├── Asia-Pacific Markets
│   ├── FX & Commodities
│   └── Fixed Income
├── III. Macro Calendar ✨ NEW
│   ├── Released data with deviation analysis
│   ├── Upcoming data today
│   └── Central bank speeches
├── IV. Central Bank Watch ✨ NEW
│   ├── Fed/ECB/BOJ/PBOC
│   ├── Meeting minutes summary
│   └── Rate cut/hike probabilities
├── V. Sector & Stock News ✨ UPGRADED
│   ├── News by sector (8 major sectors)
│   ├── Earnings calendar
│   └── Analyst rating changes
├── VI. Pre-market Movers ✨ NEW
│   ├── Pre-market gainers/losers
│   ├── ETF fund flows
│   ├── Options market activity
│   └── Block trades
├── VII. Risk Radar ✨ NEW
│   ├── Geopolitical risks
│   ├── Major event calendar
│   ├── Technical key levels
│   └── Market sentiment indicators
├── VIII. Key Thresholds ✨ NEW
│   ├── Technical levels for major indices
│   └── Macro indicator thresholds
├── IX. Trading Strategy ✨ UPGRADED
│   ├── Today's key focus
│   ├── Recommended trade setups (entry/stop/target)
│   └── Hedge recommendations
├── X. Chart Analysis
│   └── 6 professional charts + feature extraction
├── XI. Tomorrow's Focus
│   ├── Important data releases
│   ├── Earnings releases
│   └── Other catalysts
└── XII. AI Deep Analysis
    └── AI-powered insights
```

### 2. Data Source Upgrade

| Data Type | Student Version | Professional Version |
|-----------|----------------|---------------------|
| Market Prices | Yahoo Finance | Yahoo Finance + Bloomberg/Wind support |
| News Sources | 3 RSS feeds | 5+ RSS + sector classification + importance scoring |
| Macro Data | ❌ None | ✅ Economic calendar + central bank events |
| Earnings Data | ❌ None | ✅ Earnings calendar + EPS estimates |
| Analyst Ratings | ❌ None | ✅ Upgrade/Downgrade tracking |
| ETF Flows | ❌ None | ✅ 13 major ETF flow monitoring |
| Options Data | ❌ None | ✅ Unusual Options Activity |
| Risk Events | ❌ None | ✅ Geopolitical + event calendar |
| Technical Levels | ❌ None | ✅ 4 major indices key levels |
| Sentiment Indicators | ❌ None | ✅ AAII + Fear/Greed + Put/Call |

### 3. Analysis Depth Upgrade

#### Student Version Analysis
- Academic-oriented, theory-focused
- Lacks specific trading recommendations
- Risk alerts not specific enough
- No scenario analysis

#### Professional Version Analysis
- Investment bank practical style, straight to the point
- Specific trade setups (entry/stop/target/position size)
- Clear risk alerts and hedge plans
- Multi-scenario analysis (if-then logic)
- Time-sensitive (must-read before market open)

### 4. Code Architecture Upgrade

#### Student Version (Single File)
```python
main.py (800+ lines)
├── Data fetching
├── Chart generation
├── LLM analysis
└── Report assembly
```

#### Professional Version (Modular)
```python
main_professional.py (Main program)
├── modules/
│   ├── data_fetcher.py (Market data)
│   ├── macro_calendar.py (Macro calendar) ✨
│   ├── sector_news.py (Sector news) ✨
│   ├── market_movers.py (Market movers) ✨
│   ├── risk_radar.py (Risk radar) ✨
│   ├── report_template.py (Professional template) ✨
│   ├── chart_features.py (Chart features)
│   └── llm_client.py (LLM client)
```

## Migration Steps

### Step 1: Install New Dependencies

```bash
pip install -r market_diary/requirements.txt
```

New dependencies:
- `beautifulsoup4` - Web parsing
- `python-dateutil` - Date handling

### Step 2: Configure Environment Variables

```bash
# Copy configuration template
cp .env.example .env

# Edit .env file and fill in API key
nano .env
```

### Step 3: Test Run

```bash
# Use quick start script (recommended)
./run_morning_briefing.sh

# Or run Python directly
python market_diary/main_professional.py --date 2026-04-13
```

### Step 4: Compare Output

```bash
# Generate student version report
python market_diary/main.py --date 2026-04-13

# Generate professional version report
python market_diary/main_professional.py --date 2026-04-13

# Compare the two reports
diff reports/2026-04-13.md reports_professional/2026-04-13_morning_briefing.md
```

## Customization Recommendations

### 1. Modify Institution Name

Edit `modules/report_template.py`:

```python
def get_professional_template(date: str) -> str:
    template = f"""# Morning Briefing | {date}

**[Your Institution Name] · Strategy Research Department**  
Report Time: {{report_time}}  
Analyst Team: [Your Team Name]
```

### 2. Adjust Sector Coverage

Edit `modules/sector_news.py`:

```python
SECTORS = {
    'Technology': ['tech', 'software', 'AI'],
    'Financials': ['bank', 'insurance'],
    # Add sectors you focus on
    'New Energy': ['solar', 'wind', 'battery'],
}
```

### 3. Customize Technical Levels

Edit `modules/risk_radar.py`:

```python
KEY_LEVELS = {
    'SPX': {
        'resistance': [6950, 7000, 7100],  # Adjust based on actual levels
        'support': [6800, 6750, 6700],
    },
    # Add assets you track
    'AAPL': {
        'resistance': [180, 185, 190],
        'support': [170, 165, 160],
    },
}
```

### 4. Connect Professional Data Sources

#### Connect Bloomberg Terminal

```python
# Add in modules/data_fetcher.py
import blpapi

def fetch_bloomberg_data(tickers, fields):
    session = blpapi.Session()
    session.start()
    # ... Bloomberg API calls
    return data
```

#### Connect Wind

```python
# Add in modules/macro_calendar.py
from WindPy import w

w.start()
calendar_data = w.edb("M0017142", "2026-04-13", "2026-04-13")
```

## Production Deployment

### 1. Scheduled Task Setup

#### Linux/Mac (crontab)

```bash
# Edit crontab
crontab -e

# Add scheduled task (daily at 6:00 AM)
0 6 * * * cd /path/to/project && ./run_morning_briefing.sh >> logs/cron.log 2>&1
```

#### Windows (Task Scheduler)

1. Open "Task Scheduler"
2. Create Basic Task
3. Trigger: Daily at 6:00 AM
4. Action: Start Program `run_morning_briefing.bat`

### 2. Email Push

Add at the end of `main_professional.py`:

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

# Call at the end of main() function
send_email_report(final_report, report_date)
```

## Performance Optimization

### 1. Enable Caching

```python
# Add in modules/data_fetcher.py
import functools
import time

@functools.lru_cache(maxsize=128)
def fetch_market_data_cached(date):
    return fetch_market_data(date)
```

### 2. Parallel Data Fetching

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

### 3. Use Faster LLM

```bash
# Switch to GPT-4-turbo
export LLM_MODEL=gpt-4-turbo

# Or use local model (requires Ollama)
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3
```

## FAQ

### Q1: Professional version is much slower than student version?
A: Professional version fetches more data sources. You can:
- Use `--skip-charts` to skip charts
- Enable caching
- Fetch data in parallel

### Q2: How to generate only certain sections?
A: Modify the `sections` configuration in `config_example.json`.

### Q3: Can I generate English reports?
A: Yes, the system now generates reports in English by default.

### Q4: What are the data source API costs?
A: 
- Yahoo Finance: Free
- Bloomberg Terminal: ~$2,000/month
- Wind: ~¥10,000/year
- Trading Economics: $50-500/month

### Q5: Can I deploy to the cloud?
A: Yes, supports:
- AWS Lambda + EventBridge
- Google Cloud Functions + Cloud Scheduler
- Azure Functions + Timer Trigger
- Alibaba Cloud Function Compute + Scheduled Trigger

## Technical Support

Having issues?
1. Check `README_PROFESSIONAL.md`
2. Review GitHub Issues
3. Contact development team

---

**Happy upgrading! 🚀**
