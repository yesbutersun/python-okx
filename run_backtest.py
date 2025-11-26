#!/usr/bin/env python3
# ==============================
# 运行所有策略回测的主程序
# ==============================
import os
import sys

import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_backtest import BacktestEngine


def load_data(csv_path):
    """加载CSV数据"""
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"数据加载完成:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    print(f"  - 价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def main():
    """主函数"""
    print("开始策略回测")
    print("=" * 60)

    # 加载数据
    csv_path = "stock_data/0_kline_20241116_20251116.csv"
    try:
        df = load_data(csv_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建回测引擎
    print("\n初始化回测引擎...")
    engine = BacktestEngine(
        initial_capital=10000,  # 初始资金 $10,000
        commission=0.001,       # 手续费 0.1%
        slippage=0.0005         # 滑点 0.05%
    )

    # 运行所有策略回测
    print("\n开始回测所有策略...")
    results = engine.backtest_all_strategies(df)

    # 生成报告
    print("\n生成回测报告...")
    report_path = engine.generate_report(results, 'backtest_report.html')

    # 保存详细结果
    print("\n保存详细结果...")
    save_detailed_results(results)

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

        print(f"{'策略名称':<15} {'总收益率':<10} {'夏普比率':<8} {'最大回撤':<10} {'胜率':<8} {'交易次数':<8}")
        print("-" * 60)

        for strategy_name, result in sorted_results:
            stats = result['stats']
            return_symbol = "+" if stats['total_return'] > 0 else "-"
            print(f"{strategy_name:<15} {return_symbol} {stats['total_return_pct']:>7.2f}% "
                  f"{stats['sharpe_ratio']:>7.2f} "
                  f"{stats['max_drawdown_pct']:>8.2f}% "
                  f"{stats['win_rate']:>7.1%} "
                  f"{stats['completed_trades']:>7}")

    print(f"\n回测完成！")
    print(f"报告文件: {report_path}")
    print(f"详细结果: backtest_results/")


def save_detailed_results(results):
    """保存详细的回测结果"""
    import json

    # 创建结果目录
    os.makedirs('backtest_results', exist_ok=True)

    # 保存交易记录
    for strategy_name, result in results.items():
        if result and result['trades'] is not None and len(result['trades']) > 0:
            trades_df = result['trades']
            trades_df.to_csv(f'backtest_results/{strategy_name}_trades.csv', index=False)

        # 保存权益曲线
        if result and result['equity_curve'] is not None:
            equity_df = result['equity_curve']
            equity_df.to_csv(f'backtest_results/{strategy_name}_equity.csv', index=False)

    # 保存统计摘要
    summary = {}
    for strategy_name, result in results.items():
        if result and result['stats']:
            summary[strategy_name] = result['stats']

    with open('backtest_results/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("详细结果已保存到 backtest_results/ 目录")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n回测被用户中断")
    except Exception as e:
        print(f"\n回测过程中出错: {e}")
        import traceback
        traceback.print_exc()
