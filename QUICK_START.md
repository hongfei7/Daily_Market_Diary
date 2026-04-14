# Quick Start Guide

## Get Started with Investment Bank Morning Briefing System in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
# After cloning or downloading the project, enter the project directory
cd market_diary

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key (1 minute)

```bash
# Method 1: Set environment variable (recommended)
export MINIMAX_API_KEY="your_api_key_here"

# Method 2: Create .env file
cp .env.example .env
# Then edit the .env file and fill in your API key
```

### Step 3: Run Tests (1 minute)

```bash
# Test if the system is working properly
python test_professional_system.py
```

If you see "🎉 All tests passed!" the system is configured correctly.

### Step 4: Generate Your First Morning Briefing (2 minutes)

```bash
# Use quick start script (recommended)
./run_morning_briefing.sh

# Or run Python directly
python market_diary/main_professional.py
```

### Step 5: View the Report

Reports are saved in the `reports_professional/` directory:

```bash
# View the latest report
ls -lt reports_professional/*.md | head -1

# Open in browser (requires Markdown preview plugin)
# Or use VS Code / Typora or other editors
```

---

## Common Commands

### Generate Report for Specific Date

```bash
python market_diary/main_professional.py --date 2026-04-13
```

### Quick Test (Skip Charts)

```bash
python market_diary/main_professional.py --skip-charts
```

### Debug Mode (Save Intermediate Data)

```bash
python market_diary/main_professional.py --debug
```

### Specify Output Directory

```bash
python market_diary/main_professional.py --output-dir my_reports
```

---

## Report Example

The generated morning briefing includes the following sections:

```
Morning Briefing | 2026-04-13
├── I. Executive Summary (Market at a glance)
├── II. Market Snapshot (Global market overview)
├── III. Macro Calendar (Today's important data)
├── IV. Central Bank Watch (Central bank updates)
├── V. Sector & Stock News (Industry news)
├── VI. Pre-market Movers (Pre-market movements)
├── VII. Risk Radar (Risk alerts)
├── VIII. Key Thresholds (Technical levels)
├── IX. Trading Strategy (Trading recommendations)
├── X. Chart Analysis (6 professional charts)
├── XI. Tomorrow's Focus (Tomorrow's watchlist)
└── XII. AI Deep Analysis (AI-powered insights)
```

---

## Scheduled Automatic Generation

### Linux/Mac

```bash
# Edit crontab
crontab -e

# Add scheduled task (daily at 6:00 AM)
0 6 * * * cd /path/to/project && ./run_morning_briefing.sh
```

### Windows

1. Open "Task Scheduler"
2. Create Basic Task
3. Trigger: Daily at 6:00 AM
4. Action: Start Program `run_morning_briefing.bat`

---

## Troubleshooting

### Issue 1: Python Not Found

```bash
# Check Python version (requires 3.8+)
python --version

# Or use python3
python3 --version
```

### Issue 2: API Key Error

```bash
# Check environment variable
echo $MINIMAX_API_KEY

# If empty, reset it
export MINIMAX_API_KEY="your_key"
```

### Issue 3: Dependency Installation Failed

```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall dependencies
pip install -r market_diary/requirements.txt --force-reinstall
```

### Issue 4: Data Fetch Timeout

```bash
# Check network connection
ping www.google.com

# If proxy is needed
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

---

## Next Steps

- 📖 Read [README_PROFESSIONAL.md](README_PROFESSIONAL.md) for complete features
- 🔧 Check [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) to learn customization
- 🎯 Connect professional data sources (Bloomberg / Wind) to improve data quality
- 📧 Configure email push to receive morning briefings automatically

---

## Need Help?

- View documentation: `README_PROFESSIONAL.md`
- Run tests: `python test_professional_system.py`
- Submit Issue: GitHub Issues
- Contact developer: [your-email@example.com]

---

**Enjoy using the system! 📈**
