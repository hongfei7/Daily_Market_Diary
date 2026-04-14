# 快速开始指南

## 5分钟上手投行研究院晨报系统

### 第一步：安装依赖（1分钟）

```bash
# 克隆或下载项目后，进入项目目录
cd market_diary

# 安装 Python 依赖
pip install -r requirements.txt
```

### 第二步：配置 API 密钥（1分钟）

```bash
# 方式1：设置环境变量（推荐）
export MINIMAX_API_KEY="your_api_key_here"

# 方式2：创建 .env 文件
cp .env.example .env
# 然后编辑 .env 文件，填入你的 API 密钥
```

### 第三步：运行测试（1分钟）

```bash
# 测试系统是否正常
python test_professional_system.py
```

如果看到 "🎉 所有测试通过！" 说明系统配置正确。

### 第四步：生成第一份晨报（2分钟）

```bash
# 使用快速启动脚本（推荐）
./run_morning_briefing.sh

# 或者直接运行 Python
python market_diary/main_professional.py
```

### 第五步：查看报告

报告保存在 `reports_professional/` 目录：

```bash
# 查看最新报告
ls -lt reports_professional/*.md | head -1

# 在浏览器中打开（需要 Markdown 预览插件）
# 或使用 VS Code / Typora 等编辑器打开
```

---

## 常用命令

### 生成指定日期的报告

```bash
python market_diary/main_professional.py --date 2026-04-13
```

### 快速测试（跳过图表）

```bash
python market_diary/main_professional.py --skip-charts
```

### 调试模式（保存中间数据）

```bash
python market_diary/main_professional.py --debug
```

### 指定输出目录

```bash
python market_diary/main_professional.py --output-dir my_reports
```

---

## 报告示例

生成的晨报包含以下章节：

```
📊 Morning Briefing | 2026-04-13
├── 📌 Executive Summary（一眼看懂今日市场）
├── 🌍 Market Snapshot（全球市场概览）
├── 📅 Macro Calendar（今日重要数据）
├── 🏛️ Central Bank Watch（央行动态）
├── 🏢 Sector & Stock News（行业要闻）
├── 💹 Pre-market Movers（盘前异动）
├── ⚠️ Risk Radar（风险提示）
├── 💡 Trading Strategy（交易策略）
├── 📊 Chart Analysis（6张专业图表）
└── 🔮 Tomorrow's Focus（明日关注）
```

---

## 定时自动生成

### Linux/Mac

```bash
# 编辑 crontab
crontab -e

# 添加定时任务（每天早上 6:00）
0 6 * * * cd /path/to/project && ./run_morning_briefing.sh
```

### Windows

1. 打开"任务计划程序"
2. 创建基本任务
3. 触发器：每天 6:00
4. 操作：启动程序 `run_morning_briefing.bat`

---

## 故障排查

### 问题1：找不到 Python

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 或者使用 python3
python3 --version
```

### 问题2：API 密钥错误

```bash
# 检查环境变量
echo $MINIMAX_API_KEY

# 如果为空，重新设置
export MINIMAX_API_KEY="your_key"
```

### 问题3：依赖安装失败

```bash
# 升级 pip
pip install --upgrade pip

# 重新安装依赖
pip install -r market_diary/requirements.txt --force-reinstall
```

### 问题4：数据获取超时

```bash
# 检查网络连接
ping www.google.com

# 如果需要代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

---

## 下一步

- 📖 阅读 [README_PROFESSIONAL.md](README_PROFESSIONAL.md) 了解完整功能
- 🔧 查看 [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) 学习定制化
- 🎯 接入专业数据源（Bloomberg / Wind）提升数据质量
- 📧 配置邮件推送，每天自动收到晨报

---

## 需要帮助？

- 查看文档：`README_PROFESSIONAL.md`
- 运行测试：`python test_professional_system.py`
- 提交 Issue：GitHub Issues
- 联系开发者：[your-email@example.com]

---

**祝您使用愉快！📈**
