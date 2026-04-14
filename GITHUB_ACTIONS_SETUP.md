# GitHub Actions 自动化配置指南

## 📋 概述

本项目提供两个 GitHub Actions workflow：

1. **学生版晨报** (`market_diary.yml`) - 每天北京时间 03:58 运行
2. **专业版晨报** (`morning_briefing_professional.yml`) - 每天北京时间 06:00 运行

## 🔧 配置步骤

### Step 1: 设置 API 密钥

1. 进入你的 GitHub 仓库
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 添加以下密钥：

| 名称 | 值 | 说明 |
|------|-----|------|
| `MINIMAX_API_KEY` | 你的 MiniMax API Key | 必填 |

### Step 2: 启用 GitHub Actions

1. 进入仓库的 **Actions** 标签页
2. 如果看到提示，点击 **I understand my workflows, go ahead and enable them**
3. 确认两个 workflow 都已启用：
   - ✅ Daily Market Diary
   - ✅ Daily Morning Briefing (Professional)

### Step 3: 配置权限

确保 GitHub Actions 有写入权限：

1. 进入 **Settings** → **Actions** → **General**
2. 滚动到 **Workflow permissions**
3. 选择 **Read and write permissions**
4. 勾选 **Allow GitHub Actions to create and approve pull requests**
5. 点击 **Save**

## 📅 运行时间

### 学生版晨报
- **定时运行**: 每天 UTC 19:58 (北京时间 03:58)
- **输出目录**: `reports/`
- **文件名**: `YYYY-MM-DD.md`

### 专业版晨报
- **定时运行**: 每天 UTC 22:00 (北京时间 06:00)
- **输出目录**: `reports_professional/`
- **文件名**: `YYYY-MM-DD_morning_briefing.md`

## 🚀 手动触发

### 方式1: GitHub 网页界面

1. 进入 **Actions** 标签页
2. 选择要运行的 workflow
3. 点击 **Run workflow** 按钮
4. 选择分支（通常是 `main`）
5. （专业版）可选：输入自定义日期 (YYYY-MM-DD)
6. 点击 **Run workflow**

### 方式2: GitHub CLI

```bash
# 运行学生版
gh workflow run "Daily Market Diary"

# 运行专业版（默认昨天）
gh workflow run "Daily Morning Briefing (Professional)"

# 运行专业版（指定日期）
gh workflow run "Daily Morning Briefing (Professional)" -f date=2026-04-13
```

## 📊 查看运行结果

### 在线查看

1. 进入 **Actions** 标签页
2. 点击最近的运行记录
3. 查看各个步骤的日志

### 下载报告

生成的报告会自动提交到仓库：

```bash
# 拉取最新报告
git pull

# 查看学生版报告
ls reports/

# 查看专业版报告
ls reports_professional/
```

### 下载 Artifacts

每次运行都会上传报告作为 artifacts（保留30天）：

1. 进入 workflow 运行详情页
2. 滚动到底部的 **Artifacts** 部分
3. 点击下载 `morning-briefing-XXX.zip`

## 🔍 故障排查

### 问题1: Workflow 没有自动运行

**可能原因：**
- 仓库是 fork 的（GitHub 默认禁用 fork 的定时任务）
- Workflow 文件有语法错误
- 仓库长期无活动（GitHub 会自动禁用）

**解决方法：**
```bash
# 检查 workflow 语法
gh workflow list

# 手动触发一次
gh workflow run "Daily Morning Briefing (Professional)"
```

### 问题2: API 密钥错误

**检查步骤：**
1. 确认 `MINIMAX_API_KEY` 已正确设置
2. 查看 workflow 日志中的 "Debug environment" 步骤
3. 确认 `MINIMAX_API_KEY: present=True, len=XX`

**常见错误：**
- 密钥前后有空格
- 密钥过期或无效
- 密钥权限不足

### 问题3: 报告生成失败

**查看日志：**
1. 进入 Actions → 点击失败的运行
2. 查看 "Generate Morning Briefing" 步骤
3. 查看错误信息

**常见错误：**
```
Error: name 'List' is not defined
→ 已修复：添加了 typing 导入

Error: No module named 'beautifulsoup4'
→ 已修复：添加了可选导入

Error: API rate limit exceeded
→ 等待一段时间后重试
```

### 问题4: 无法推送到仓库

**可能原因：**
- 没有写入权限
- 分支保护规则阻止

**解决方法：**
1. 检查 Settings → Actions → General → Workflow permissions
2. 确保选择了 "Read and write permissions"
3. 如果有分支保护，添加例外规则允许 `github-actions[bot]`

## 📝 自定义配置

### 修改运行时间

编辑 `.github/workflows/morning_briefing_professional.yml`：

```yaml
schedule:
  # 改为每天北京时间 07:00 (UTC 23:00)
  - cron: "0 23 * * *"
```

**Cron 表达式说明：**
```
┌───────────── 分钟 (0 - 59)
│ ┌───────────── 小时 (0 - 23)
│ │ ┌───────────── 日期 (1 - 31)
│ │ │ ┌───────────── 月份 (1 - 12)
│ │ │ │ ┌───────────── 星期 (0 - 6, 0=周日)
│ │ │ │ │
* * * * *
```

**常用时间：**
- 每天 06:00 北京时间: `0 22 * * *` (UTC)
- 每天 07:00 北京时间: `0 23 * * *` (UTC)
- 每天 08:00 北京时间: `0 0 * * *` (UTC)
- 工作日 06:00: `0 22 * * 1-5` (UTC)

### 修改输出目录

编辑 `market_diary/main_professional.py`：

```python
parser.add_argument(
    "--output-dir",
    type=str,
    default="reports_professional",  # 改为你想要的目录
    help="输出目录",
)
```

### 添加通知

在 workflow 末尾添加通知步骤：

```yaml
- name: Send notification
  if: success()
  run: |
    # 企业微信通知
    curl -X POST "${{ secrets.WECHAT_WEBHOOK }}" \
      -H 'Content-Type: application/json' \
      -d '{
        "msgtype": "text",
        "text": {
          "content": "✅ 晨报已生成: '"$(date -d 'yesterday' +%F)"'"
        }
      }'
```

## 🔐 安全建议

### 1. 保护 API 密钥

- ✅ 使用 GitHub Secrets 存储
- ✅ 不要在代码中硬编码
- ✅ 不要在日志中打印完整密钥
- ✅ 定期轮换密钥

### 2. 限制权限

```yaml
permissions:
  contents: write  # 只授予必要的权限
```

### 3. 使用环境保护

对于生产环境，可以设置环境保护规则：

1. Settings → Environments → New environment
2. 添加保护规则（如需要审批）
3. 在 workflow 中指定环境：

```yaml
jobs:
  generate-briefing:
    environment: production  # 使用受保护的环境
```

## 📊 监控和日志

### 查看运行历史

```bash
# 列出最近的运行
gh run list --workflow="morning_briefing_professional.yml"

# 查看特定运行的日志
gh run view <run-id> --log
```

### 设置告警

1. 进入 Settings → Notifications
2. 配置 Actions 失败通知
3. 选择通知方式（邮件/移动推送）

## 🎯 最佳实践

### 1. 测试 Workflow

在推送到 main 分支前，先在本地测试：

```bash
# 安装 act (本地运行 GitHub Actions)
brew install act  # macOS
# 或
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# 本地运行 workflow
act -j generate-briefing
```

### 2. 使用缓存

workflow 已配置 pip 缓存：

```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: 'pip'  # 缓存 pip 依赖
```

### 3. 并行运行

如果需要同时生成多个版本：

```yaml
strategy:
  matrix:
    version: [student, professional]
```

### 4. 条件执行

只在工作日运行：

```yaml
- name: Check if weekday
  id: check
  run: |
    if [ $(date +%u) -le 5 ]; then
      echo "is_weekday=true" >> $GITHUB_OUTPUT
    fi

- name: Generate report
  if: steps.check.outputs.is_weekday == 'true'
  run: ...
```

## 📞 获取帮助

### 文档资源
- [GitHub Actions 官方文档](https://docs.github.com/en/actions)
- [Workflow 语法参考](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Cron 表达式生成器](https://crontab.guru/)

### 常见问题
- 查看 `QUICK_START.md` - 快速开始
- 查看 `README_PROFESSIONAL.md` - 完整文档
- 提交 GitHub Issue - 报告问题

---

**配置完成后，系统将每天自动生成晨报并提交到仓库！📈**
