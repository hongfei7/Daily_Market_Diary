# GitHub Actions 快速参考

## 🚀 一键配置

### 1. 设置 API 密钥（必须）

```
仓库 → Settings → Secrets and variables → Actions → New repository secret
```

添加：
- **Name**: `MINIMAX_API_KEY`
- **Value**: 你的 MiniMax API Key

### 2. 启用写入权限（必须）

```
仓库 → Settings → Actions → General → Workflow permissions
```

选择：
- ✅ Read and write permissions

### 3. 启用 Actions（必须）

```
仓库 → Actions 标签页
```

点击：
- ✅ I understand my workflows, go ahead and enable them

## ⏰ 运行时间

| Workflow | 时间 | 输出 |
|----------|------|------|
| 学生版 | 每天 03:58 北京时间 | `reports/YYYY-MM-DD.md` |
| 专业版 | 每天 06:00 北京时间 | `reports_professional/YYYY-MM-DD_morning_briefing.md` |

## 🎯 手动运行

### 网页界面
```
Actions → 选择 workflow → Run workflow
```

### 命令行
```bash
# 专业版（默认昨天）
gh workflow run "Daily Morning Briefing (Professional)"

# 专业版（指定日期）
gh workflow run "Daily Morning Briefing (Professional)" -f date=2026-04-13
```

## 🔍 查看结果

### 在线查看
```
Actions → 点击运行记录 → 查看日志
```

### 下载报告
```bash
git pull
ls reports_professional/
```

## ❌ 常见错误

### 错误1: API 密钥未设置
```
✗ 未设置 API 密钥
```
**解决**: 在 Settings → Secrets 中添加 `MINIMAX_API_KEY`

### 错误2: 无写入权限
```
Error: Permission denied
```
**解决**: Settings → Actions → General → 选择 "Read and write permissions"

### 错误3: 模块导入失败
```
ModuleNotFoundError: No module named 'xxx'
```
**解决**: 检查 `requirements.txt` 是否包含该模块

### 错误4: LLM 调用失败
```
Error: API rate limit exceeded
```
**解决**: 等待一段时间后重试，或升级 API 套餐

## 📝 修改运行时间

编辑 `.github/workflows/morning_briefing_professional.yml`:

```yaml
schedule:
  # 北京时间 = UTC + 8
  # 06:00 北京 = 22:00 UTC
  - cron: "0 22 * * *"
  
  # 07:00 北京 = 23:00 UTC
  # - cron: "0 23 * * *"
  
  # 08:00 北京 = 00:00 UTC
  # - cron: "0 0 * * *"
```

## 🧪 本地测试

```bash
# 测试系统
python test_github_actions.py

# 生成报告
python market_diary/main_professional.py --date 2026-04-13
```

## 📞 获取帮助

- 详细文档: `GITHUB_ACTIONS_SETUP.md`
- 快速开始: `QUICK_START.md`
- 完整文档: `README_PROFESSIONAL.md`

---

**配置完成后，每天自动生成晨报！📈**
