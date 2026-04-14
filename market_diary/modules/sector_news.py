"""
sector_news.py — 行业与个股新闻聚合模块

功能：
1. 按行业板块分类抓取重要新闻
2. 财报发布与盈利惊喜
3. 并购交易、监管政策
4. 分析师评级调整
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    import feedparser
except ImportError:
    feedparser = None


class SectorNewsAggregator:
    """行业新闻聚合器"""
    
    # 行业分类
    SECTORS = {
        'Technology': ['tech', 'software', 'semiconductor', 'AI', 'cloud'],
        'Financials': ['bank', 'insurance', 'fintech', 'payment'],
        'Healthcare': ['pharma', 'biotech', 'medical', 'health'],
        'Energy': ['oil', 'gas', 'renewable', 'energy'],
        'Consumer': ['retail', 'consumer', 'e-commerce'],
        'Industrials': ['manufacturing', 'aerospace', 'defense'],
        'Materials': ['mining', 'metals', 'chemicals'],
        'Real Estate': ['property', 'REIT', 'real estate'],
    }
    
    # 新闻源配置
    NEWS_SOURCES = {
        'reuters_business': 'http://feeds.reuters.com/reuters/businessNews',
        'reuters_markets': 'http://feeds.reuters.com/reuters/marketsNews',
        'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
        'wsj_markets': 'https://feeds.content.dowjones.io/public/rss/mw_topstories',
    }
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_sector_news(self, max_per_sector: int = 3) -> Dict[str, List[Dict]]:
        """
        按行业获取新闻
        
        Returns:
            {
                'Technology': [{title, summary, source, time, importance}],
                'Financials': [...],
                ...
            }
        """
        all_news = self._fetch_all_news()
        categorized = self._categorize_news(all_news)
        
        # 每个行业只保留最重要的 N 条
        filtered = {}
        for sector, news_list in categorized.items():
            filtered[sector] = sorted(
                news_list,
                key=lambda x: x.get('importance_score', 0),
                reverse=True
            )[:max_per_sector]
        
        return filtered
    
    def _fetch_all_news(self) -> List[Dict]:
        """从所有新闻源获取新闻"""
        all_news = []
        
        for source_name, url in self.NEWS_SOURCES.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:  # 每个源取前20条
                    all_news.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', '')[:200],
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': source_name,
                    })
            except Exception as e:
                print(f"[sector_news] Error fetching {source_name}: {e}")
        
        return all_news
    
    def _categorize_news(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """将新闻分类到各个行业"""
        categorized = {sector: [] for sector in self.SECTORS}
        
        for news in news_list:
            text = (news['title'] + ' ' + news['summary']).lower()
            
            # 计算重要性分数
            importance_score = self._calculate_importance(news)
            news['importance_score'] = importance_score
            
            # 分类到行业
            matched = False
            for sector, keywords in self.SECTORS.items():
                if any(kw.lower() in text for kw in keywords):
                    categorized[sector].append(news)
                    matched = True
                    break
            
            # 未匹配的放入 Other
            if not matched:
                if 'Other' not in categorized:
                    categorized['Other'] = []
                categorized['Other'].append(news)
        
        return categorized
    
    def _calculate_importance(self, news: Dict) -> float:
        """计算新闻重要性分数"""
        score = 0.0
        text = (news['title'] + ' ' + news['summary']).lower()
        
        # 高优先级关键词
        high_priority = ['merger', 'acquisition', 'earnings', 'guidance', 'upgrade', 'downgrade',
                        'regulation', 'approval', 'breakthrough', 'crisis', 'default']
        for kw in high_priority:
            if kw in text:
                score += 2.0
        
        # 中优先级关键词
        medium_priority = ['deal', 'contract', 'partnership', 'launch', 'expansion']
        for kw in medium_priority:
            if kw in text:
                score += 1.0
        
        # 来源权重
        if 'bloomberg' in news['source']:
            score += 1.5
        elif 'reuters' in news['source']:
            score += 1.2
        
        return score
    
    def fetch_earnings_calendar(self, date: str) -> List[Dict]:
        """获取财报日历（盘前/盘后）"""
        # 实际需要接入 Earnings Whispers / Yahoo Finance API
        return [
            {
                'ticker': 'AAPL',
                'company': 'Apple Inc.',
                'time': 'After Market Close',
                'eps_estimate': '1.45',
                'revenue_estimate': '89.5B',
            }
        ]
    
    def fetch_analyst_changes(self, date: str) -> List[Dict]:
        """获取分析师评级变动"""
        # 实际需要接入 Bloomberg / FactSet
        return [
            {
                'ticker': 'TSLA',
                'firm': 'Morgan Stanley',
                'action': 'Upgrade',
                'from_rating': 'Equal Weight',
                'to_rating': 'Overweight',
                'price_target': '350',
                'previous_target': '280',
            }
        ]
    
    def format_for_report(self, sector_news: Dict, earnings: List, analyst_changes: List) -> str:
        """格式化为晨报文本"""
        lines = []
        
        # 行业新闻
        lines.append("### 行业与个股要闻")
        lines.append("")
        
        for sector, news_list in sector_news.items():
            if not news_list:
                continue
            
            lines.append(f"#### {sector}")
            lines.append("")
            
            for news in news_list:
                lines.append(f"- **{news['title']}**")
                if news.get('summary'):
                    lines.append(f"  {news['summary']}")
                lines.append(f"  *来源: {news['source']}*")
                lines.append("")
        
        # 财报日历
        if earnings:
            lines.append("#### 今日财报发布")
            lines.append("")
            lines.append("| 股票 | 公司 | 时间 | EPS预期 | 营收预期 |")
            lines.append("|------|------|------|---------|----------|")
            for e in earnings:
                lines.append(
                    f"| {e['ticker']} | {e['company']} | {e['time']} | "
                    f"${e['eps_estimate']} | ${e['revenue_estimate']} |"
                )
            lines.append("")
        
        # 分析师评级
        if analyst_changes:
            lines.append("#### 分析师评级调整")
            lines.append("")
            for change in analyst_changes:
                action_emoji = "⬆️" if change['action'] == 'Upgrade' else "⬇️" if change['action'] == 'Downgrade' else "➡️"
                lines.append(
                    f"- {action_emoji} **{change['ticker']}** | {change['firm']}: "
                    f"{change['from_rating']} → {change['to_rating']} | "
                    f"目标价 ${change['previous_target']} → ${change['price_target']}"
                )
            lines.append("")
        
        return "\n".join(lines)


def fetch_sector_data(date: str) -> Dict:
    """
    主入口函数：获取行业新闻数据
    
    Args:
        date: YYYY-MM-DD 格式
        
    Returns:
        包含行业新闻、财报、评级的字典
    """
    aggregator = SectorNewsAggregator()
    
    sector_news = aggregator.fetch_sector_news(max_per_sector=3)
    earnings = aggregator.fetch_earnings_calendar(date)
    analyst_changes = aggregator.fetch_analyst_changes(date)
    
    return {
        'sector_news': sector_news,
        'earnings_calendar': earnings,
        'analyst_changes': analyst_changes,
        'formatted_text': aggregator.format_for_report(sector_news, earnings, analyst_changes)
    }
