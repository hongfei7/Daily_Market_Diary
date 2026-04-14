"""
market_movers.py — 盘前异动与资金流向模块

功能：
1. 盘前涨跌幅最大的个股
2. 大宗交易和龙虎榜（A股）
3. ETF资金流向
4. 期权市场异常活跃标的
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd


class MarketMoversAnalyzer:
    """市场异动分析器"""
    
    # 主要市场 ETF 列表
    MAJOR_ETFS = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'IWM': 'Russell 2000',
        'EEM': 'Emerging Markets',
        'FXI': 'China Large-Cap',
        'EWJ': 'Japan',
        'EWG': 'Germany',
        'GLD': 'Gold',
        'SLV': 'Silver',
        'USO': 'Oil',
        'TLT': '20+ Year Treasury',
        'HYG': 'High Yield Corp',
        'LQD': 'Investment Grade Corp',
    }
    
    def __init__(self):
        pass
    
    def fetch_premarket_movers(self, top_n: int = 10) -> Dict:
        """
        获取盘前异动股票
        
        Returns:
            {
                'gainers': [...],
                'losers': [...],
                'most_active': [...]
            }
        """
        try:
            # 实际需要接入实时盘前数据源（如 IEX Cloud, Polygon.io）
            # 这里提供框架结构
            
            gainers = self._fetch_top_gainers(top_n)
            losers = self._fetch_top_losers(top_n)
            active = self._fetch_most_active(top_n)
            
            return {
                'gainers': gainers,
                'losers': losers,
                'most_active': active,
            }
        except Exception as e:
            print(f"[market_movers] Error fetching premarket movers: {e}")
            return {'gainers': [], 'losers': [], 'most_active': []}
    
    def _fetch_top_gainers(self, n: int) -> List[Dict]:
        """获取盘前涨幅最大的股票"""
        # 模拟数据结构
        return [
            {
                'ticker': 'NVDA',
                'company': 'NVIDIA Corp',
                'price': 485.20,
                'change': 15.30,
                'change_pct': 3.25,
                'volume': 2500000,
                'catalyst': 'Earnings beat expectations'
            }
        ]
    
    def _fetch_top_losers(self, n: int) -> List[Dict]:
        """获取盘前跌幅最大的股票"""
        return [
            {
                'ticker': 'INTC',
                'company': 'Intel Corp',
                'price': 42.15,
                'change': -2.85,
                'change_pct': -6.33,
                'volume': 1800000,
                'catalyst': 'Guidance cut'
            }
        ]
    
    def _fetch_most_active(self, n: int) -> List[Dict]:
        """获取盘前成交最活跃的股票"""
        return []
    
    def fetch_etf_flows(self, date: str) -> List[Dict]:
        """
        获取 ETF 资金流向
        
        Args:
            date: YYYY-MM-DD 格式
            
        Returns:
            ETF 流入/流出数据列表
        """
        flows = []
        
        for ticker, name in self.MAJOR_ETFS.items():
            try:
                # 实际需要接入 ETF.com API 或 Bloomberg
                # 这里使用 yfinance 获取基础数据
                etf = yf.Ticker(ticker)
                hist = etf.history(period='5d')
                
                if hist.empty:
                    continue
                
                # 简化计算：用成交量变化估算资金流向
                recent_volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                # 用价格变化 * 成交量估算净流入
                price_change_pct = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                estimated_flow = recent_volume * price_change_pct / 100  # 简化估算
                
                flows.append({
                    'ticker': ticker,
                    'name': name,
                    'price': round(hist['Close'].iloc[-1], 2),
                    'change_pct': round(price_change_pct, 2),
                    'volume': int(recent_volume),
                    'volume_ratio': round(volume_ratio, 2),
                    'estimated_flow_direction': 'inflow' if estimated_flow > 0 else 'outflow',
                })
            except Exception as e:
                print(f"[market_movers] Error fetching ETF {ticker}: {e}")
        
        # 按成交量比率排序
        flows.sort(key=lambda x: abs(x['volume_ratio'] - 1.0), reverse=True)
        
        return flows[:10]  # 返回前10个异常的
    
    def fetch_block_trades_cn(self, date: str) -> List[Dict]:
        """
        获取A股大宗交易数据
        
        实际需要接入 Wind / 东方财富 API
        """
        return [
            {
                'ticker': '600519.SH',
                'name': '贵州茅台',
                'volume': 50000,
                'price': 1680.00,
                'premium': -2.5,  # 折价率
                'buyer': '机构专用',
                'seller': '机构专用',
            }
        ]
    
    def fetch_unusual_options(self, date: str) -> List[Dict]:
        """
        获取期权市场异常活跃标的
        
        实际需要接入 Unusual Whales / Market Chameleon API
        """
        return [
            {
                'ticker': 'TSLA',
                'option_type': 'Call',
                'strike': 250,
                'expiry': '2026-05-15',
                'volume': 15000,
                'open_interest': 8000,
                'volume_oi_ratio': 1.88,
                'implied_vol': 65.5,
                'sentiment': 'bullish',
            }
        ]
    
    def format_for_report(self, movers: Dict, etf_flows: List, block_trades: List, options: List) -> str:
        """格式化为晨报文本"""
        lines = []
        
        lines.append("### 盘前异动与资金流向")
        lines.append("")
        
        # 盘前涨跌幅榜
        if movers.get('gainers'):
            lines.append("#### 盘前涨幅榜 Top 5")
            lines.append("")
            lines.append("| 股票 | 公司 | 价格 | 涨跌幅 | 成交量 | 催化剂 |")
            lines.append("|------|------|------|--------|--------|--------|")
            
            for stock in movers['gainers'][:5]:
                lines.append(
                    f"| {stock['ticker']} | {stock['company']} | ${stock['price']:.2f} | "
                    f"+{stock['change_pct']:.2f}% | {stock['volume']:,} | {stock['catalyst']} |"
                )
            lines.append("")
        
        if movers.get('losers'):
            lines.append("#### 盘前跌幅榜 Top 5")
            lines.append("")
            lines.append("| 股票 | 公司 | 价格 | 涨跌幅 | 成交量 | 催化剂 |")
            lines.append("|------|------|------|--------|--------|--------|")
            
            for stock in movers['losers'][:5]:
                lines.append(
                    f"| {stock['ticker']} | {stock['company']} | ${stock['price']:.2f} | "
                    f"{stock['change_pct']:.2f}% | {stock['volume']:,} | {stock['catalyst']} |"
                )
            lines.append("")
        
        # ETF 资金流向
        if etf_flows:
            lines.append("#### ETF 资金流向异常")
            lines.append("")
            lines.append("| ETF | 名称 | 价格 | 涨跌幅 | 成交量比率 | 流向 |")
            lines.append("|-----|------|------|--------|------------|------|")
            
            for etf in etf_flows[:8]:
                flow_emoji = "📈" if etf['estimated_flow_direction'] == 'inflow' else "📉"
                lines.append(
                    f"| {etf['ticker']} | {etf['name']} | ${etf['price']:.2f} | "
                    f"{etf['change_pct']:+.2f}% | {etf['volume_ratio']:.2f}x | "
                    f"{flow_emoji} {etf['estimated_flow_direction']} |"
                )
            lines.append("")
        
        # 期权异动
        if options:
            lines.append("#### 期权市场异常活跃")
            lines.append("")
            for opt in options[:5]:
                sentiment_emoji = "🐂" if opt['sentiment'] == 'bullish' else "🐻"
                lines.append(
                    f"- {sentiment_emoji} **{opt['ticker']}** {opt['option_type']} ${opt['strike']} "
                    f"到期 {opt['expiry']} | 成交量 {opt['volume']:,} | "
                    f"Vol/OI {opt['volume_oi_ratio']:.2f}x | IV {opt['implied_vol']:.1f}%"
                )
            lines.append("")
        
        # A股大宗交易（如果有）
        if block_trades:
            lines.append("#### A股大宗交易")
            lines.append("")
            lines.append("| 代码 | 名称 | 成交量 | 成交价 | 溢价率 | 买方 | 卖方 |")
            lines.append("|------|------|--------|--------|--------|------|------|")
            
            for trade in block_trades[:5]:
                premium_str = f"{trade['premium']:+.2f}%"
                lines.append(
                    f"| {trade['ticker']} | {trade['name']} | {trade['volume']:,} | "
                    f"¥{trade['price']:.2f} | {premium_str} | {trade['buyer']} | {trade['seller']} |"
                )
            lines.append("")
        
        return "\n".join(lines)


def fetch_movers_data(date: str) -> Dict:
    """
    主入口函数：获取市场异动数据
    
    Args:
        date: YYYY-MM-DD 格式
        
    Returns:
        包含盘前异动、ETF流向、期权数据的字典
    """
    analyzer = MarketMoversAnalyzer()
    
    movers = analyzer.fetch_premarket_movers(top_n=10)
    etf_flows = analyzer.fetch_etf_flows(date)
    block_trades = analyzer.fetch_block_trades_cn(date)
    options = analyzer.fetch_unusual_options(date)
    
    return {
        'premarket_movers': movers,
        'etf_flows': etf_flows,
        'block_trades': block_trades,
        'unusual_options': options,
        'formatted_text': analyzer.format_for_report(movers, etf_flows, block_trades, options)
    }
