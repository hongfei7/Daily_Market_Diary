# Professional Report Chart Matrix

## Design principle

For a Hong Kong sell-side morning note, charts should answer four questions quickly:

1. What changed in the global regime overnight?
2. Did Hong Kong local flow and funding confirm or contradict that regime?
3. What is the single highest-signal chart worth deeper attention today?
4. Which tables are needed for traceability rather than decoration?

## Visual dashboard

The visual dashboard should be a four-panel decision page rather than a text wall:

1. Global regime board
   - Cross-asset 1D moves for S&P 500, Nasdaq 100, HSI, HSTECH, FXI, US 10Y, DXY, WTI, Gold, VIX
   - Purpose: establish the overnight risk regime in one glance

2. Hong Kong local tape
   - Turnover vs 20D
   - Southbound flow
   - Short-selling ratio
   - HIBOR 1M
   - Aggregate Balance
   - A/H premium
   - Purpose: tell whether Hong Kong has local confirmation or only proxy confirmation

3. Flow / pressure concentration
   - Prefer Southbound active names
   - Fallback to short-selling concentration
   - Fallback to attribution ranking
   - Purpose: identify whether the move is broad or concentrated

4. Catalyst ladder
   - Today's macro releases
   - Earnings / policy / conference catalysts
   - Purpose: convert the note into a work plan

Risk score, Hong Kong style, data coverage, and report mode remain compact header fields. Coverage-name detail stays in the report so the visual does not become a text dashboard.

## Daily One Chart

Daily One Chart should never duplicate the dashboard. It should be one focused analytical panel with:

- Main chart area
- Right-side takeaway box
- Bottom caption and source

Recommended story hierarchy:

1. HKEX short-selling pressure map
2. Southbound active-name concentration
3. A/H premium dispersion
4. Turnover conviction versus short pressure
5. FX pressure versus Hong Kong growth
6. Oil shock read-through
7. Cross-asset attribution board
8. Composite risk score fallback

## Hong Kong Trend Pack

The report now includes a dedicated historical context page. This should stay distinct from both the dashboard and the Daily One Chart.

Current four-panel trend pack:

1. Southbound cumulative flow
   - Daily Southbound net buy bars
   - 20-session cumulative line
   - Purpose: show whether Connect support is persistent or episodic

2. HIBOR and Aggregate Balance
   - 1M HIBOR line
   - Aggregate Balance area / line
   - Purpose: show whether Hong Kong funding is tightening while local liquidity stays flat or drains

3. Relative leadership
   - Indexed HSI vs HSCEI vs HSTECH over 20-30 sessions
   - Purpose: distinguish broad beta, state-owned / old-economy, and growth leadership

4. A/H premium heatmap
   - Last 5 sessions across the widest covered pairs
   - Purpose: identify where relative-value pressure is broadening or narrowing

## Core tables in the markdown report

These are the tables a professional morning report should always keep:

1. Global Asset Price Dashboard
2. Hong Kong Key Data Quick Check
3. Cross-Asset Attribution Table
4. Local Flow and Funding Checks
5. Stock Connect Southbound Active Names
6. A/H Premium Dispersion
7. HKEX Short-Selling Watch
8. Macro and Policy Agenda
9. HKEX Announcements
10. Earnings / Results Watch
11. Rating Changes
12. Coverage Pool Table

## Next chart upgrades

The next chart upgrades worth building are:

1. HSI / HSCEI / HSTECH breadth heatmap
2. Sector rotation rank-change heatmap
3. A/H premium change heatmap
4. Earnings calendar timeline
5. Southbound active-name concentration trend
6. ETF shorting versus spot turnover divergence chart
