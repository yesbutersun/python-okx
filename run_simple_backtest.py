#!/usr/bin/env python3
# ==============================
# 运行简化优化策略回测
# ==============================
import os
import sys
import json

import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_backtest import BacktestEngine
from optimized_strategies_simple import SIMPLE_OPTIMIZED_STRATEGIES, get_simple_optimized_strategy_list


def main():
    """主函数"""
    print("=" * 60)
    print("简化优化策略回测")
    print("=" * 60)

    # 加载数据
    csv_path = "stock_data/0_kline_20241116_20251116.csv"
    print(f"\n加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"数据加载完成: {len(df)} 条记录")
    print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005
    )

    # 存储结果
    results = {}

    # 运行每个策略
    strategy_list = get_simple_optimized_strategy_list()
    print(f"\n开始回测 {len(strategy_list)} 个优化策略...")
    print("-" * 60)

    for strategy_name in strategy_list:
        print(f"\n测试策略: {strategy_name}")
        try:
            # 准备数据
            df_prepared = engine.prepare_dataframe(df)

            # 生成信号
            strategy_func = SIMPLE_OPTIMIZED_STRATEGIES[strategy_name]
            signals = strategy_func(df_prepared)

            print(f"  信号生成完成")
            print(f"  多头信号: {signals['long_entry'].sum()}")
            print(f"  空头信号: {signals['short_entry'].sum()}")

            # 执行回测
            result = engine._execute_backtest(df_prepared, signals, strategy_name)
            results[strategy_name] = result

            if result and result['stats']:
                stats = result['stats']
                print(f"  ✅ 回测完成")
                print(f"     总收益率: {stats['total_return_pct']:.2f}%")
                print(f"     夏普比率: {stats['sharpe_ratio']:.2f}")
                print(f"     最大回撤: {stats['max_drawdown_pct']:.2f}%")
                print(f"     胜率: {stats['win_rate']:.1%}")
                print(f"     交易次数: {stats['completed_trades']}")
            else:
                print(f"  ❌ 回测失败")
                results[strategy_name] = None

        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            results[strategy_name] = None

    # 保存结果
    print("\n" + "-" * 60)
    print("保存回测结果...")

    # 创建结果目录
    os.makedirs('backtest_results', exist_ok=True)

    # 保存详细结果
    for strategy_name, result in results.items():
        if result and result['trades'] is not None and len(result['trades']) > 0:
            trades_df = result['trades']
            trades_df.to_csv(f'backtest_results/{strategy_name}_trades.csv', index=False)

        if result and result['equity_curve'] is not None:
            equity_df = result['equity_curve']
            equity_df.to_csv(f'backtest_results/{strategy_name}_equity.csv', index=False)

    # 保存统计摘要
    summary = {}
    for strategy_name, result in results.items():
        if result and result['stats']:
            summary[strategy_name] = result['stats']

    with open('backtest_results/simple_optimized_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 打印汇总
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    successful_results = {k: v for k, v in results.items() if v is not None}
    if successful_results:
        # 按收益率排序
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1]['stats']['total_return'],
            reverse=True
        )

        print(f"{'策略名称':<15} {'收益率':<10} {'夏普':<6} {'回撤':<8} {'胜率':<8} {'交易':<6}")
        print("-" * 60)

        for strategy_name, result in sorted_results:
            stats = result['stats']
            return_symbol = "📈" if stats['total_return'] > 0 else "📉"
            print(f"{strategy_name:<15} {return_symbol} {stats['total_return_pct']:>7.2f}% "
                  f"{stats['sharpe_ratio']:>5.2f} "
                  f"{stats['max_drawdown_pct']:>7.2f}% "
                  f"{stats['win_rate']:>7.1%} "
                  f"{stats['completed_trades']:>5}")

        # 找出最佳策略
        best_strategy = sorted_results[0]
        print(f"\n🏆 最佳策略: {best_strategy[0]}")
        print(f"   收益率: {best_strategy[1]['stats']['total_return_pct']:.2f}%")
        print(f"   夏普比率: {best_strategy[1]['stats']['sharpe_ratio']:.2f}")

    print("\n✅ 回测完成！")
    print(f"📊 详细结果保存在: backtest_results/")
    print(f"📄 统计摘要: simple_optimized_summary.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  回测被用户中断")
    except Exception as e:
        print(f"\n\n❌ 回测出错: {e}")
        import traceback
        traceback.print_exc()
