"""
risk_radar.py — 风险提示与关注事项模块

功能：
1. 地缘政治风险监控
2. 重大事件日历（FOMC、期权到期、IPO、解禁）
3. 技术面关键支撑/阻力位
4. 市场情绪指标
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None


class RiskRadar:
    """风险雷达监控器"""
    
    # 关键技术位配置
    KEY_LEVELS = {
        'SPX': {
            'resistance': [6950, 7000, 7100],
            'support': [6800, 6750, 6700],
        },
        'NDX': {
            'resistance': [20500, 21000, 21500],
            'support': [19800, 19500, 19000],
        },
        'DXY': {
            'resistance': [99.50, 100.00, 100.50],
            'support': [98.00, 97.50, 97.00],
        },
        'US10Y': {
            'resistance': [4.40, 4.50, 4.60],
            'support': [4.20, 4.10, 4.00],
        },
    }
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_geopolitical_risks(self) -> List[Dict]:
        """
        获取地缘政治风险事件
        
        实际需要接入 Reuters / Bloomberg 地缘政治新闻流
        """
        return [
            {
                'region': 'Middle East',
                'event': 'Iran-Israel tensions escalate',
                'risk_level': 'high',  # high/medium/low
                'impact': 'Oil prices, safe haven demand',
                'last_update': datetime.now().isoformat(),
            },
            {
                'region': 'Asia-Pacific',
                'event': 'Taiwan Strait military exercises',
                'risk_level': 'medium',
                'impact': 'Semiconductor supply chain, regional equities',
                'last_update': datetime.now().isoformat(),
            }
        ]
    
    def fetch_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """
        获取未来N天的重大事件日历
        
        Args:
            days_ahead: 向前看的天数
            
        Returns:
            事件列表
        """
        events = []
        
        # FOMC 会议
        events.append({
            'date': '2026-05-06',
            'type': 'FOMC Meeting',
            'description': 'Federal Reserve Interest Rate Decision',
            'importance': 'critical',
        })
        
        # 期权到期日
        events.append({
            'date': '2026-04-18',
            'type': 'Options Expiry',
            'description': 'Monthly options expiration (OpEx)',
            'importance': 'high',
        })
        
        # IPO
        events.append({
            'date': '2026-04-20',
            'type': 'IPO',
            'description': 'XYZ Corp IPO pricing',
            'importance': 'medium',
        })
        
        # 解禁
        events.append({
            'date': '2026-04-25',
            'type': 'Lock-up Expiry',
            'description': 'ABC Inc. insider lock-up expires (500M shares)',
            'importance': 'high',
        })
        
        # 过滤未来N天的事件
        today = datetime.now().date()
        cutoff = today + timedelta(days=days_ahead)
        
        filtered = []
        for event in events:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d').date()
            if today <= event_date <= cutoff:
                filtered.append(event)
        
        return sorted(filtered, key=lambda x: x['date'])
    
    def fetch_technical_levels(self, current_prices: Dict) -> Dict:
        """
        获取关键技术位及当前价格距离
        
        Args:
            current_prices: {'SPX': 6850, 'DXY': 98.5, ...}
            
        Returns:
            技术位分析
        """
        analysis = {}
        
        for symbol, levels in self.KEY_LEVELS.items():
            current = current_prices.get(symbol)
            if current is None:
                continue
            
            # 找到最近的支撑和阻力
            nearest_resistance = min(
                [r for r in levels['resistance'] if r > current],
                default=None
            )
            nearest_support = max(
                [s for s in levels['support'] if s < current],
                default=None
            )
            
            analysis[symbol] = {
                'current': current,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_distance': round((nearest_resistance / current - 1) * 100, 2) if nearest_resistance else None,
                'support_distance': round((current / nearest_support - 1) * 100, 2) if nearest_support else None,
            }
        
        return analysis
    
    def fetch_sentiment_indicators(self) -> Dict:
        """
        获取市场情绪指标
        
        实际需要接入 AAII Sentiment / CNN Fear & Greed / Put-Call Ratio
        """
        return {
            'aaii_bull_bear': {
                'bullish': 42.5,
                'neutral': 28.3,
                'bearish': 29.2,
                'interpretation': 'Neutral to slightly bullish',
            },
            'fear_greed_index': {
                'value': 58,
                'level': 'Greed',  # Extreme Fear / Fear / Neutral / Greed / Extreme Greed
            },
            'put_call_ratio': {
                'equity': 0.68,
                'index': 1.15,
                'interpretation': 'Equity optimism, index hedging',
            },
            'vix_term_structure': {
                'front_month': 19.2,
                'second_month': 20.5,
                'slope': 'contango',  # contango / backwardation
                'interpretation': 'Normal risk pricing, no immediate stress',
            }
        }
    
    def format_for_report(self, geo_risks: List, events: List, tech_levels: Dict, sentiment: Dict) -> str:
        """格式化为晨报文本"""
        lines = []
        
        lines.append("### 风险提示与关注事项")
        lines.append("")
        
        # 地缘政治风险
        if geo_risks:
            lines.append("#### 地缘政治风险")
            lines.append("")
            for risk in geo_risks:
                risk_emoji = "🔴" if risk['risk_level'] == 'high' else "🟡" if risk['risk_level'] == 'medium' else "🟢"
                lines.append(
                    f"- {risk_emoji} **{risk['region']}**: {risk['event']} | "
                    f"影响: {risk['impact']}"
                )
            lines.append("")
        
        # 重大事件日历
        if events:
            lines.append("#### 本周重大事件")
            lines.append("")
            lines.append("| 日期 | 类型 | 事件 | 重要性 |")
            lines.append("|------|------|------|--------|")
            
            for event in events:
                importance_emoji = "🔴" if event['importance'] == 'critical' else "🟡" if event['importance'] == 'high' else "🟢"
                lines.append(
                    f"| {event['date']} | {event['type']} | {event['description']} | {importance_emoji} |"
                )
            lines.append("")
        
        # 技术面关键位
        if tech_levels:
            lines.append("#### 技术面关键位")
            lines.append("")
            lines.append("| 标的 | 当前价 | 最近阻力 | 距离 | 最近支撑 | 距离 |")
            lines.append("|------|--------|----------|------|----------|------|")
            
            for symbol, data in tech_levels.items():
                res_str = f"{data['nearest_resistance']:.2f} (+{data['resistance_distance']:.1f}%)" if data['nearest_resistance'] else "N/A"
                sup_str = f"{data['nearest_support']:.2f} (-{data['support_distance']:.1f}%)" if data['nearest_support'] else "N/A"
                
                lines.append(
                    f"| {symbol} | {data['current']:.2f} | {res_str} | | {sup_str} | |"
                )
            lines.append("")
        
        # 市场情绪指标
        if sentiment:
            lines.append("#### 市场情绪指标")
            lines.append("")
            
            if 'aaii_bull_bear' in sentiment:
                aaii = sentiment['aaii_bull_bear']
                lines.append(
                    f"- **AAII 情绪调查**: 看多 {aaii['bullish']:.1f}% | "
                    f"中性 {aaii['neutral']:.1f}% | 看空 {aaii['bearish']:.1f}% "
                    f"({aaii['interpretation']})"
                )
            
            if 'fear_greed_index' in sentiment:
                fg = sentiment['fear_greed_index']
                lines.append(f"- **恐惧贪婪指数**: {fg['value']} ({fg['level']})")
            
            if 'put_call_ratio' in sentiment:
                pc = sentiment['put_call_ratio']
                lines.append(
                    f"- **Put/Call Ratio**: 个股 {pc['equity']:.2f} | "
                    f"指数 {pc['index']:.2f} ({pc['interpretation']})"
                )
            
            if 'vix_term_structure' in sentiment:
                vix = sentiment['vix_term_structure']
                lines.append(
                    f"- **VIX 期限结构**: 近月 {vix['front_month']:.1f} | "
                    f"次月 {vix['second_month']:.1f} | {vix['slope']} "
                    f"({vix['interpretation']})"
                )
            
            lines.append("")
        
        return "\n".join(lines)


def fetch_risk_data(current_prices: Dict = None) -> Dict:
    """
    主入口函数：获取风险监控数据
    
    Args:
        current_prices: 当前价格字典 {'SPX': 6850, ...}
        
    Returns:
        包含风险事件、技术位、情绪指标的字典
    """
    radar = RiskRadar()
    
    if current_prices is None:
        current_prices = {}
    
    geo_risks = radar.fetch_geopolitical_risks()
    events = radar.fetch_upcoming_events(days_ahead=7)
    tech_levels = radar.fetch_technical_levels(current_prices)
    sentiment = radar.fetch_sentiment_indicators()
    
    return {
        'geopolitical_risks': geo_risks,
        'upcoming_events': events,
        'technical_levels': tech_levels,
        'sentiment_indicators': sentiment,
        'formatted_text': radar.format_for_report(geo_risks, events, tech_levels, sentiment)
    }
