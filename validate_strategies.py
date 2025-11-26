#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略验证脚本
验证strategy.py中的所有交易策略是否能正常工作
并生成策略性能报告
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入策略函数
from strategy import (
    STRATEGIES, get_strategy_list, run_strategy,
    rsi_reversal_strategy,
    trend_atr_signal,
    boll_rsi_signal,
    trend_volatility_stop_signal,
    breakout_strategy,
    mean_reversion_strategy,
    momentum_strategy,
    macd_strategy
)


class StrategyValidator:
    """策略验证器"""

    def __init__(self):
        self.strategies = STRATEGIES
        self.results = {}

    def generate_sample_data(self, days=30, trend='random'):
        """
        生成示例数据

        Args:
            days: 数据天数
            trend: 趋势类型 ('random', 'up', 'down', 'sideways')
        """
        # 生成时间序列（5分钟K线，每天288条）
        periods = days * 288
        dates = pd.date_range('2024-01-01', periods=periods, freq='5min')

        np.random.seed(42)

        # 基础价格
        base_price = 43000

        if trend == 'random':
            # 随机游走
            returns = np.random.normal(0, 0.002, periods)
        elif trend == 'up':
            # 上涨趋势
            returns = np.random.normal(0.001, 0.002, periods)
        elif trend == 'down':
            # 下跌趋势
            returns = np.random.normal(-0.001, 0.002, periods)
        else:  # sideways
            # 横盘震荡
            returns = np.random.normal(0, 0.001, periods)

        # 累积计算价格
        price_changes = returns.cumsum()
        close_prices = base_price * (1 + price_changes)

        # 生成开高低收数据
        data = {
            'Close': close_prices,
            'Volume': np.random.uniform(100, 1000, periods)
        }

        df = pd.DataFrame(data, index=dates)

        # 生成OHLC数据
        df['Open'] = df['Close'].shift(1).fillna(base_price)
        price_noise = np.random.normal(0, 0.001, periods) * df['Close']
        df['High'] = np.maximum(df['Close'], df['Open']) + np.abs(price_noise) * 0.5
        df['Low'] = np.minimum(df['Close'], df['Open']) - np.abs(price_noise) * 0.5

        # 确保高低开收的逻辑关系
        df['High'] = np.maximum(df['High'], df['Close'])
        df['Low'] = np.minimum(df['Low'], df['Close'])

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def validate_strategy(self, strategy_name, strategy_func, df):
        """
        验证单个策略

        Returns:
            dict: 验证结果
        """
        print(f"\n=== 验证策略: {strategy_name} ===")

        try:
            # 计算策略信号
            signals = strategy_func(df)

            if signals.empty:
                return {
                    'status': 'error',
                    'message': '策略函数返回空DataFrame'
                }

            # 基本检查
            required_columns = ['long_entry', 'long_exit', 'short_entry', 'short_exit']
            missing_cols = [col for col in required_columns if col not in signals.columns]

            if missing_cols:
                return {
                    'status': 'error',
                    'message': f'缺少必要列: {missing_cols}'
                }

            # 统计信号
            signal_stats = {
                'long_entry': signals['long_entry'].sum(),
                'long_exit': signals['long_exit'].sum(),
                'short_entry': signals['short_entry'].sum(),
                'short_exit': signals['short_exit'].sum(),
                'total_signals': signals[required_columns].sum().sum()
            }

            # 找到信号发生的位置
            signal_locations = {}
            for col in required_columns:
                signal_mask = signals[col]
                if signal_mask.any():
                    signal_locations[col] = signals[signal_mask].index.tolist()[:5]  # 只显示前5个

            # 检查信号逻辑
            logic_issues = self._check_signal_logic(signals)

            # 性能模拟
            performance = self._simulate_performance(df, signals)

            result = {
                'status': 'success',
                'signal_stats': signal_stats,
                'signal_locations': signal_locations,
                'logic_issues': logic_issues,
                'performance': performance,
                'data_length': len(df),
                'signal_length': len(signals)
            }

            print(f"✓ 策略验证成功")
            print(f"  数据长度: {len(df)} 条K线")
            print(f"  信号长度: {len(signals)} 条")
            print(f"  总信号数: {signal_stats['total_signals']}")
            print(f"  做多信号: {signal_stats['long_entry']} 次")
            print(f"  做空信号: {signal_stats['short_entry']} 次")

            if logic_issues:
                print(f"  ⚠️ 逻辑问题: {len(logic_issues)} 个")

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"✗ 策略验证失败: {error_msg}")

            return {
                'status': 'error',
                'message': error_msg
            }

    def _check_signal_logic(self, signals):
        """检查信号逻辑"""
        issues = []

        # 检查是否在平仓前有对应的持仓信号
        for i in range(1, len(signals)):
            prev_row = signals.iloc[i-1]
            curr_row = signals.iloc[i]

            # 平多仓但之前没有多仓
            if curr_row['long_exit'] and not prev_row['long_entry']:
                issues.append(f"位置 {i}: 平多仓信号但没有对应的多仓开仓")

            # 平空仓但之前没有空仓
            if curr_row['short_exit'] and not prev_row['short_entry']:
                issues.append(f"位置 {i}: 平空仓信号但没有对应的空仓开仓")

            # 同时有开仓和平仓信号
            if curr_row['long_entry'] and curr_row['long_exit']:
                issues.append(f"位置 {i}: 同时有开多仓和平多仓信号")

            if curr_row['short_entry'] and curr_row['short_exit']:
                issues.append(f"位置 {i}: 同时有开空仓和平空仓信号")

        return issues[:10]  # 只返回前10个问题

    def _simulate_performance(self, df, signals):
        """模拟策略性能"""
        try:
            balance = 10000  # 初始资金
            position = 0  # 0: 空仓, 1: 多仓, -1: 空仓
            position_size = 0.1  # 每次交易的BTC数量
            entry_price = 0

            trades = []
            balances = []

            for i, (timestamp, signal_row) in enumerate(signals.iterrows()):
                current_price = df.loc[timestamp, 'Close']

                # 执行交易信号
                if signal_row['long_entry'] and position != 1:
                    position = 1
                    entry_price = current_price
                    cost = entry_price * position_size
                    balance -= cost
                    trades.append({
                        'time': timestamp,
                        'action': 'long_entry',
                        'price': entry_price,
                        'balance': balance
                    })

                elif signal_row['short_entry'] and position != -1:
                    position = -1
                    entry_price = current_price
                    trades.append({
                        'time': timestamp,
                        'action': 'short_entry',
                        'price': entry_price,
                        'balance': balance
                    })

                elif signal_row['long_exit'] and position == 1:
                    position = 0
                    revenue = current_price * position_size
                    balance += revenue
                    pnl = revenue - entry_price * position_size
                    trades.append({
                        'time': timestamp,
                        'action': 'long_exit',
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance
                    })

                elif signal_row['short_exit'] and position == -1:
                    position = 0
                    revenue = entry_price * position_size
                    balance += revenue
                    pnl = revenue - current_price * position_size
                    trades.append({
                        'time': timestamp,
                        'action': 'short_exit',
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance
                    })

                # 记录余额
                balances.append({
                    'time': timestamp,
                    'balance': balance,
                    'price': current_price,
                    'position': position
                })

            # 计算统计指标
            if trades:
                pnl_trades = [t for t in trades if 'pnl' in t]
                total_trades = len(pnl_trades)
                winning_trades = len([t for t in pnl_trades if t['pnl'] > 0])
                losing_trades = len([t for t in pnl_trades if t['pnl'] < 0])

                total_pnl = sum(t['pnl'] for t in pnl_trades) if pnl_trades else 0
                avg_win = np.mean([t['pnl'] for t in pnl_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t['pnl'] for t in pnl_trades if t['pnl'] < 0]) if losing_trades > 0 else 0

                performance = {
                    'initial_balance': 10000,
                    'final_balance': balance,
                    'total_return': balance - 10000,
                    'return_percentage': (balance - 10000) / 10000 * 100,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                }
            else:
                performance = {
                    'initial_balance': 10000,
                    'final_balance': balance,
                    'total_return': 0,
                    'return_percentage': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                }

            return performance

        except Exception as e:
            return {
                'error': f'性能模拟失败: {str(e)}'
            }

    def validate_all_strategies(self, df=None):
        """验证所有策略"""
        print("开始验证所有交易策略...")
        print("=" * 60)

        # 如果没有提供数据，生成示例数据
        if df is None:
            df = self.generate_sample_data(days=7, trend='random')
            print("已生成7天随机趋势示例数据")

        all_results = {}

        for strategy_name, strategy_func in self.strategies.items():
            result = self.validate_strategy(strategy_name, strategy_func, df)
            all_results[strategy_name] = result

            # 添加延迟避免输出过快
            import time
            time.sleep(0.5)

        self.results = all_results
        return all_results

    def generate_report(self, output_file='strategy_validation_report.json'):
        """生成验证报告"""
        if not self.results:
            print("没有验证结果，请先运行验证")
            return

        # 统计汇总
        total_strategies = len(self.results)
        successful_strategies = len([r for r in self.results.values() if r['status'] == 'success'])
        failed_strategies = total_strategies - successful_strategies

        # 策略性能对比
        performance_comparison = []
        for name, result in self.results.items():
            if result['status'] == 'success' and 'performance' in result:
                perf = result['performance']
                if 'total_trades' in perf and perf['total_trades'] > 0:
                    performance_comparison.append({
                        'strategy': name,
                        'return_percentage': perf['return_percentage'],
                        'win_rate': perf['win_rate'],
                        'total_trades': perf['total_trades'],
                        'profit_factor': perf['profit_factor']
                    })

        # 按收益率排序
        performance_comparison.sort(key=lambda x: x['return_percentage'], reverse=True)

        report = {
            'validation_time': datetime.now().isoformat(),
            'summary': {
                'total_strategies': total_strategies,
                'successful_strategies': successful_strategies,
                'failed_strategies': failed_strategies,
                'success_rate': successful_strategies / total_strategies if total_strategies > 0 else 0
            },
            'performance_comparison': performance_comparison,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # 打印摘要
        print("\n" + "=" * 60)
        print("策略验证报告摘要")
        print("=" * 60)
        print(f"验证时间: {report['validation_time']}")
        print(f"总策略数: {total_strategies}")
        print(f"成功验证: {successful_strategies}")
        print(f"验证失败: {failed_strategies}")
        print(f"成功率: {report['summary']['success_rate']*100:.1f}%")

        if performance_comparison:
            print(f"\n策略性能排名 (按收益率):")
            for i, comp in enumerate(performance_comparison[:5], 1):
                print(f"{i}. {comp['strategy']}")
                print(f"   收益率: {comp['return_percentage']:+.2f}%")
                print(f"   胜率: {comp['win_rate']*100:.1f}%")
                print(f"   交易次数: {comp['total_trades']}")
                print(f"   盈亏比: {comp['profit_factor']:.2f}")

        print(f"\n详细报告已保存至: {output_file}")

        return report

    def _generate_recommendations(self):
        """生成策略建议"""
        recommendations = []

        for name, result in self.results.items():
            if result['status'] == 'error':
                recommendations.append(f"{name}: 修复策略函数错误 - {result['message']}")
            elif 'logic_issues' in result and result['logic_issues']:
                recommendations.append(f"{name}: 修复信号逻辑问题，发现 {len(result['logic_issues'])} 个问题")
            elif 'performance' in result and 'total_trades' in result['performance']:
                perf = result['performance']
                if perf['total_trades'] == 0:
                    recommendations.append(f"{name}: 策略过于保守，未产生任何交易信号")
                elif perf['win_rate'] < 0.3:
                    recommendations.append(f"{name}: 胜率较低 ({perf['win_rate']*100:.1f}%)，建议优化参数")
                elif perf['return_percentage'] < -10:
                    recommendations.append(f"{name}: 收益表现较差 ({perf['return_percentage']:.1f}%)，不建议使用")
                else:
                    recommendations.append(f"{name}: 策略表现良好，可以考虑用于实盘交易")

        return recommendations


def main():
    """主函数"""
    print("BTC交易策略验证工具")
    print("=" * 60)

    validator = StrategyValidator()

    # 验证所有策略
    print("1. 验证策略功能正确性...")
    results = validator.validate_all_strategies()

    # 生成报告
    print("\n2. 生成验证报告...")
    report = validator.generate_report()

    # 可选：生成不同市场环境的测试
    print("\n3. 测试不同市场环境下的策略表现...")
    market_types = ['random', 'up', 'down', 'sideways']

    for market_type in market_types:
        print(f"\n--- 测试 {market_type} 市场 ---")
        df = validator.generate_sample_data(days=3, trend=market_type)

        # 只测试表现最好的几个策略
        top_strategies = ['趋势ATR策略', '布林RSI策略', 'RSI反转策略']

        for strategy_name in top_strategies:
            if strategy_name in validator.strategies:
                result = validator.validate_strategy(
                    f"{strategy_name} ({market_type})",
                    validator.strategies[strategy_name],
                    df
                )

    print("\n验证完成！请查看详细报告了解各策略表现。")


if __name__ == "__main__":
    main()