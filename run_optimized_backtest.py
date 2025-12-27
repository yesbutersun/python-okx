#!/usr/bin/env python3
# ==============================
# 运行优化策略回测的主程序
# ==============================
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_cli_utils import create_engine, load_data, save_detailed_results
from optimized_strategies import OPTIMIZED_STRATEGIES, get_optimized_strategy_list


def main():
    """主函数"""
    print("开始优化策略回测")
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
    engine = create_engine(
        initial_capital=10000,  # 初始资金 $10,000
        commission=0.001,       # 手续费 0.1%
        slippage=0.0005,         # 滑点 0.05%
        stop_loss_threshold=500
    )

    # 运行所有优化策略回测
    print("\n开始回测所有优化策略...")
    results = {}
    strategy_list = get_optimized_strategy_list()

    for strategy_name in strategy_list:
        try:
            print(f"\n回测策略: {strategy_name}...")
            # 直接使用优化策略
            signals = OPTIMIZED_STRATEGIES[strategy_name](df)
            result = engine._execute_backtest(engine.prepare_dataframe(df), signals, strategy_name)
            results[strategy_name] = result
            if result:
                print(f"  完成 - 总收益率: {result['stats']['total_return_pct']:.2f}%")
                print(f"         夏普比率: {result['stats']['sharpe_ratio']:.2f}")
                print(f"         最大回撤: {result['stats']['max_drawdown_pct']:.2f}%")
                print(f"         胜率: {result['stats']['win_rate']:.1%}")
        except Exception as e:
            print(f"  失败 - 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = None

    # 生成报告
    print("\n生成回测报告...")
    report_path = engine.generate_report(results, 'optimized_backtest_report.html')

    # 保存详细结果
    print("\n保存详细结果...")
    save_detailed_results(results, summary_filename="optimized_summary.json")

    # 打印汇总
    print("\n" + "=" * 60)
    print("优化策略回测结果汇总")
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n回测被用户中断")
    except Exception as e:
        print(f"\n回测过程中出错: {e}")
        import traceback
        traceback.print_exc()
