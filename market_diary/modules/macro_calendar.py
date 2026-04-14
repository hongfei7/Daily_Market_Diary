"""
macro_calendar.py — 宏观经济日历与数据获取模块

功能：
1. 获取已公布的经济数据及其与预期的偏差
2. 获取今日/明日待公布的经济数据日历
3. 央行官员讲话安排
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class MacroCalendar:
    """宏观经济日历数据获取器"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_economic_calendar(self, date: str) -> Dict:
        """
        获取指定日期的经济日历
        
        Args:
            date: YYYY-MM-DD 格式
            
        Returns:
            {
                'released': [...],  # 已公布数据
                'upcoming': [...]   # 待公布数据
            }
        """
        try:
            # 使用 Investing.com 经济日历 API（需要适配实际可用的数据源）
            # 这里提供框架，实际部署时需要接入 Wind/Bloomberg/Trading Economics
            
            released_data = self._fetch_released_data(date)
            upcoming_data = self._fetch_upcoming_data(date)
            
            return {
                'released': released_data,
                'upcoming': upcoming_data,
                'meta': {
                    'date': date,
                    'fetch_time': datetime.now().isoformat()
                }
            }
        except Exception as e:
            print(f"[macro_calendar] Error fetching calendar: {e}")
            return {'released': [], 'upcoming': [], 'error': str(e)}
    
    def _fetch_released_data(self, date: str) -> List[Dict]:
        """获取已公布的经济数据"""
        # 模拟数据结构（实际需要接入真实数据源）
        return [
            {
                'time': '20:30',
                'country': 'US',
                'indicator': 'CPI MoM',
                'actual': '0.3%',
                'forecast': '0.2%',
                'previous': '0.4%',
                'impact': 'high',  # high/medium/low
                'surprise': 'beat'  # beat/miss/inline
            }
        ]
    
    def _fetch_upcoming_data(self, date: str) -> List[Dict]:
        """获取待公布的经济数据"""
        return [
            {
                'time': '20:30',
                'country': 'US',
                'indicator': 'Retail Sales MoM',
                'forecast': '0.3%',
                'previous': '0.6%',
                'impact': 'high'
            }
        ]
    
    def fetch_central_bank_events(self, date: str) -> List[Dict]:
        """获取央行事件和官员讲话安排"""
        try:
            # 实际需要接入 Bloomberg/Reuters 央行日历
            return [
                {
                    'time': '22:00',
                    'bank': 'Federal Reserve',
                    'event_type': 'speech',  # speech/meeting/minutes/decision
                    'speaker': 'Jerome Powell',
                    'title': 'Economic Outlook',
                    'importance': 'high'
                }
            ]
        except Exception as e:
            print(f"[macro_calendar] Error fetching CB events: {e}")
            return []
    
    def format_for_report(self, calendar_data: Dict, cb_events: List[Dict]) -> str:
        """格式化为晨报文本"""
        lines = []
        
        # 已公布数据
        if calendar_data.get('released'):
            lines.append("#### Released Data (Prior Day)")
            lines.append("")
            lines.append("| Time | Country | Indicator | Actual | Forecast | Prior | Deviation |")
            lines.append("|------|---------|-----------|--------|----------|-------|-----------|")
            
            for item in calendar_data['released']:
                surprise_text = "MISS" if item['surprise'] == 'miss' else "BEAT" if item['surprise'] == 'beat' else "INLINE"
                lines.append(
                    f"| {item['time']} | {item['country']} | {item['indicator']} | "
                    f"{item['actual']} | {item['forecast']} | {item['previous']} | "
                    f"{surprise_text} |"
                )
            lines.append("")
        
        # 待公布数据
        if calendar_data.get('upcoming'):
            lines.append("#### Upcoming Data (Today)")
            lines.append("")
            lines.append("| Time | Country | Indicator | Forecast | Prior | Importance |")
            lines.append("|------|---------|-----------|----------|-------|------------|")
            
            for item in calendar_data['upcoming']:
                impact_text = "HIGH" if item['impact'] == 'high' else "MEDIUM" if item['impact'] == 'medium' else "LOW"
                lines.append(
                    f"| {item['time']} | {item['country']} | {item['indicator']} | "
                    f"{item['forecast']} | {item['previous']} | {impact_text} |"
                )
            lines.append("")
        
        # 央行事件
        if cb_events:
            lines.append("#### Central Bank Events & Speeches")
            lines.append("")
            for event in cb_events:
                lines.append(f"- **{event['time']}** | {event['bank']} | {event['speaker']}: {event['title']}")
            lines.append("")
        
        return "\n".join(lines)


def fetch_macro_data(date: str) -> Dict:
    """
    主入口函数：获取宏观日历数据
    
    Args:
        date: YYYY-MM-DD 格式
        
    Returns:
        包含经济日历和央行事件的字典
    """
    calendar = MacroCalendar()
    
    calendar_data = calendar.fetch_economic_calendar(date)
    cb_events = calendar.fetch_central_bank_events(date)
    
    return {
        'calendar': calendar_data,
        'central_bank_events': cb_events,
        'formatted_text': calendar.format_for_report(calendar_data, cb_events)
    }
