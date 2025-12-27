# ==============================
# 增强版回测引擎
# ==============================
"""
🔧 权益计算修复说明 (2024-12-13):

修复前的问题:
1. 权益计算错误 - 持仓时只计算capital + unrealized_pnl，忽略了持仓价值
2. 开仓后权益接近0，实际应该正确反映持仓价值
3. 导致equity文件与trades文件数据不匹配

修复后的逻辑:
1. 多头持仓: equity = capital + (shares * current_price)
2. 空头持仓: equity = capital + (shares * entry_price) + unrealized_pnl
3. 无持仓: equity = capital
4. 平仓后正确计算可用资金，避免重复扣费

修复效果:
- equity曲线将正确反映实际权益变化
- equity文件最终权益将与trades文件PNL总和匹配
- 解决数据不一致问题
"""

from datetime import datetime

import numpy as np
import pandas as pd

from simple_strategy import STRATEGIES, get_strategy_list

# 可选的图表组件
try:
    from trading_visualizer import TradingVisualizer
except Exception:
    TradingVisualizer = None


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital=10000, commission=0.0006, slippage=0.0005,
                 enable_visualization=False, chart_dir="charts"):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
            enable_visualization: 是否输出K线与权益曲线图
            chart_dir: 图表保存目录
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = {}
        self.enable_visualization = enable_visualization and TradingVisualizer is not None
        self.chart_dir = chart_dir
        self.visualizer = TradingVisualizer(figsize=(18, 10), dpi=110) if self.enable_visualization else None

    def prepare_dataframe(self, df):
        """准备DataFrame"""
        df = df.copy()
        # 标准化列名
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # 设置时间索引
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        # 确保数值类型
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def backtest_strategy(self, df, strategy_name, strategy_params=None):
        """
        回测单个策略

        Args:
            df: 价格数据
            strategy_name: 策略名称
            strategy_params: 策略参数

        Returns:
            回测结果字典
        """
        if strategy_name not in STRATEGIES:
            raise ValueError(f"未知策略: {strategy_name}")

        df = self.prepare_dataframe(df)
        strategy_func = STRATEGIES[strategy_name]

        # 生成信号
        if strategy_params:
            signals = strategy_func(df, **strategy_params)
        else:
            signals = strategy_func(df)

        # 执行回测
        return self._execute_backtest(df, signals, strategy_name)

    def _execute_backtest(self, df, signals, strategy_name):
        """执行回测逻辑"""
        position = 0  # 0: 无仓位, 1: 多头, -1: 空头
        entry_price = 0.0
        capital = self.initial_capital
        shares = 0.0
        entry_commission = 0.0  # 记录开仓手续费

        # 记录数据
        equity_curve = []
        trades = []
        daily_returns = []

        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            signal = signals.iloc[i]

            # 修复: 正确计算当前权益
            if position == 1:
                # 多头持仓: 现金 + 持仓价值
                position_value = shares * current_price
                current_equity = capital + position_value
            elif position == -1:
                # 空头持仓: 现金 + 开仓价值 + 未实现盈亏
                # 空头的权益 = 剩余现金 + 开仓时获得的资金 + 价格变动带来的盈亏
                unrealized_pnl = (entry_price - current_price) * shares
                current_equity = capital + shares * entry_price + unrealized_pnl
            else:
                # 无持仓
                current_equity = capital

            equity_curve.append({
                'datetime': current_time,
                'equity': current_equity,
                'position': position,
                'price': current_price
            })

            # 处理交易信号
            if position == 0 and signal['long_entry']:
                # 修复: 开多头 - 正确计算购买股数和记录开仓成本
                total_cost_per_share = current_price * (1 + self.commission + self.slippage)
                shares = capital / total_cost_per_share
                entry_price = current_price * (1 + self.slippage)  # 实际成交价格（含滑点）
                entry_commission = entry_price * shares * self.commission  # 开仓手续费
                capital = 0  # 所有资金用于购买股票
                position = 1
                entry_reason = signal['long_entry_reason'] if 'long_entry_reason' in signals.columns else ''
                if not entry_reason:
                    entry_reason = 'signal_long_entry'
                trades.append({
                    'datetime': current_time,
                    'action': 'BUY',
                    'price': entry_price,
                    'shares': shares,
                    'type': 'Long',
                    'reason': entry_reason,
                    'commission': entry_commission
                })

            elif position == 0 and signal['short_entry']:
                # 修复: 开空头 - 正确计算空头股数和记录开仓收益
                total_cost_per_share = current_price * (1 + self.commission + self.slippage)
                shares = capital / total_cost_per_share
                entry_price = current_price * (1 - self.slippage)  # 空头成交价格（含滑点）
                entry_commission = entry_price * shares * self.commission  # 开仓手续费
                # 空头开仓时获得资金，但需要扣除手续费
                capital = 0  # 简化处理，权益计算时已考虑
                position = -1
                entry_reason = signal['short_entry_reason'] if 'short_entry_reason' in signals.columns else ''
                if not entry_reason:
                    entry_reason = 'signal_short_entry'
                trades.append({
                    'datetime': current_time,
                    'action': 'SELL_SHORT',
                    'price': entry_price,
                    'shares': shares,
                    'type': 'Short',
                    'reason': entry_reason,
                    'commission': entry_commission
                })

            elif position == 1 and signal['long_exit']:
                # 修复: 平多头 - 正确计算平仓后的现金
                exit_price = current_price * (1 - self.slippage)  # 平仓价格（含滑点）
                pnl = (exit_price - entry_price) * shares  # 价格变动带来的盈亏
                abs_entry_exit_price_diff = abs(exit_price - entry_price)
                exit_commission = exit_price * shares * self.commission  # 平仓手续费
                capital = shares * exit_price - exit_commission  # 平仓后获得的现金
                exit_reason = signal['long_exit_reason'] if 'long_exit_reason' in signals.columns else ''
                if not exit_reason:
                    exit_reason = 'signal_long_exit'
                trades.append({
                    'datetime': current_time,
                    'action': 'SELL',
                    'price': exit_price,
                    'shares': shares,
                    'type': 'Long',
                    'reason': exit_reason,
                    'abs_entry_exit_price_diff': abs_entry_exit_price_diff,
                    'pnl': pnl,
                    'commission': exit_commission
                })
                position = 0
                entry_price = 0.0
                shares = 0.0

            elif position == -1 and signal['short_exit']:
                # 修复: 平空头 - 正确计算平空后的现金
                exit_price = current_price * (1 + self.slippage)  # 平空价格（含滑点）
                pnl = (entry_price - exit_price) * shares  # 空头盈亏（高卖低买）
                abs_entry_exit_price_diff = abs(exit_price - entry_price)
                exit_commission = exit_price * shares * self.commission  # 平仓手续费
                # 空头平仓后的现金 = 开仓时获得的资金 + 盈亏 - 手续费
                capital = shares * entry_price + pnl - exit_commission
                exit_reason = signal['short_exit_reason'] if 'short_exit_reason' in signals.columns else ''
                if not exit_reason:
                    exit_reason = 'signal_short_exit'
                trades.append({
                    'datetime': current_time,
                    'action': 'BUY_TO_COVER',
                    'price': exit_price,
                    'shares': shares,
                    'type': 'Short',
                    'reason': exit_reason,
                    'abs_entry_exit_price_diff': abs_entry_exit_price_diff,
                    'pnl': pnl,
                    'commission': exit_commission
                })
                position = 0
                entry_price = 0.0
                shares = 0.0

        # 修复: 计算最终权益 - 与实时权益计算逻辑保持一致
        final_price = df['Close'].iloc[-1]
        if position == 1:
            # 多头持仓: 现金 + 持仓价值
            position_value = shares * final_price
            final_equity = capital + position_value
        elif position == -1:
            # 空头持仓: 现金 + 开仓价值 + 未实现盈亏
            unrealized_pnl = (entry_price - final_price) * shares
            final_equity = capital + shares * entry_price + unrealized_pnl
        else:
            # 无持仓
            final_equity = capital

        # 计算回测统计
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades)

        stats = self._calculate_stats(equity_df, trades_df, final_equity)

        result = {
            'strategy_name': strategy_name,
            'equity_curve': equity_df,
            'trades': trades_df,
            'stats': stats
        }

        # 可选输出可视化图表
        if self.enable_visualization:
            self._save_charts(df, trades_df, equity_df, strategy_name)

        return result

    def _save_charts(self, price_df, trades_df, equity_df, strategy_name):
        """使用TradingVisualizer生成K线和权益曲线图"""
        try:
            import os
            os.makedirs(self.chart_dir, exist_ok=True)

            # 价格数据恢复datetime列，仅保留必要字段避免重复列
            price_export = price_df.reset_index().rename(columns={'index': 'datetime'})
            cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            price_export = price_export[[c for c in cols if c in price_export.columns]]

            # 交易数据
            trades_export = trades_df.copy()
            if not trades_export.empty:
                trades_export['datetime'] = pd.to_datetime(trades_export['datetime'])

            # K线 + 交易点
            kline_path = os.path.join(self.chart_dir, f"{strategy_name}_kline.png")
            self.visualizer.plot_kline_with_trades(
                price_data=price_export,
                trades_data=trades_export,
                title=f"{strategy_name} - K线与交易点",
                save_path=kline_path
            )

            # 权益曲线
            if not equity_df.empty:
                equity_export = equity_df.copy()
                equity_export['datetime'] = pd.to_datetime(equity_export['datetime'])
                equity_path = os.path.join(self.chart_dir, f"{strategy_name}_equity.png")
                self.visualizer.plot_equity_curve(
                    equity_data=equity_export[['datetime', 'equity']],
                    title=f"{strategy_name} - 权益曲线",
                    save_path=equity_path
                )
        except Exception as e:
            print(f"⚠️ 生成图表失败: {e}")

    def _calculate_stats(self, equity_df, trades_df, final_equity):
        """计算回测统计数据"""
        if len(equity_df) == 0:
            return {}

        # 基础统计
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        total_trades = len(trades_df)

        # 计算每日收益率
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()

        # 计算已平仓交易
        completed_trades = trades_df[trades_df['pnl'].notna()]
        if len(completed_trades) > 0:
            win_trades = completed_trades[completed_trades['pnl'] > 0]
            lose_trades = completed_trades[completed_trades['pnl'] <= 0]

            win_rate = len(win_trades) / len(completed_trades)
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
            profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if lose_trades['pnl'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # 计算最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()

        # 计算夏普比率
        if len(daily_returns) > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # 计算卡玛比率
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_daily_return': daily_returns.mean() if len(daily_returns) > 0 else 0,
            'volatility': daily_returns.std() if len(daily_returns) > 0 else 0
        }

    def backtest_all_strategies(self, df):
        """回测所有策略"""
        results = {}
        strategy_list = get_strategy_list()

        print(f"开始回测 {len(strategy_list)} 个策略...")
        print("-" * 60)

        for strategy_name in strategy_list:
            try:
                print(f"回测策略: {strategy_name}...")
                result = self.backtest_strategy(df, strategy_name)
                results[strategy_name] = result
                print(f"  完成 - 总收益率: {result['stats']['total_return_pct']:.2f}%")
            except Exception as e:
                print(f"  失败 - 错误: {str(e)}")
                results[strategy_name] = None

        return results

    def generate_report(self, results, save_path='backtest_report.html'):
        """生成回测报告"""
        html_content = self._generate_html_report(results)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"回测报告已保存到: {save_path}")
        return save_path

    def _generate_html_report(self, results):
        """生成HTML报告"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>策略回测报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .metric {{ margin: 10px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ float: right; }}
            </style>
        </head>
        <body>
            <h1>策略回测报告</h1>
            <p>生成时间: {}</p>
            <p>初始资金: ${:,.2f}</p>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.initial_capital)

        # 汇总表格
        html += "<h2>策略汇总</h2>"
        html += "<table>"
        html += "<tr><th>策略名称</th><th>总收益率</th><th>夏普比率</th><th>最大回撤</th><th>胜率</th><th>交易次数</th></tr>"

        for strategy_name, result in results.items():
            if result and result['stats']:
                stats = result['stats']
                return_class = 'positive' if stats['total_return'] > 0 else 'negative'
                html += f"""
                <tr>
                    <td>{strategy_name}</td>
                    <td class="{return_class}">{stats['total_return_pct']:.2f}%</td>
                    <td>{stats['sharpe_ratio']:.2f}</td>
                    <td class="negative">{stats['max_drawdown_pct']:.2f}%</td>
                    <td>{stats['win_rate']:.2%}</td>
                    <td>{stats['completed_trades']}</td>
                </tr>
                """
            else:
                html += f"<tr><td>{strategy_name}</td><td colspan='5'>回测失败</td></tr>"

        html += "</table>"

        # 详细统计
        html += "<h2>详细统计</h2>"
        for strategy_name, result in results.items():
            if result and result['stats']:
                html += f"<h3>{strategy_name}</h3>"
                stats = result['stats']
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>初始资金:</span><span class='metric-value'>${stats['initial_capital']:,.2f}</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>最终权益:</span><span class='metric-value'>${stats['final_equity']:,.2f}</span>"
                html += "</div>"
                html += "<div class='metric'>"
                return_class = 'positive' if stats['total_return'] > 0 else 'negative'
                html += f"<span class='metric-name'>总收益率:</span><span class='metric-value {return_class}'>{stats['total_return_pct']:.2f}%</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>夏普比率:</span><span class='metric-value'>{stats['sharpe_ratio']:.2f}</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>最大回撤:</span><span class='metric-value negative'>{stats['max_drawdown_pct']:.2f}%</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>胜率:</span><span class='metric-value'>{stats['win_rate']:.2%}</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>总交易次数:</span><span class='metric-value'>{stats['completed_trades']}</span>"
                html += "</div>"
                html += "<div class='metric'>"
                html += f"<span class='metric-name'>盈亏比:</span><span class='metric-value'>{stats['profit_factor']:.2f}</span>"
                html += "</div>"

        html += "</body></html>"
        return html
