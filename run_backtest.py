#!/usr/bin/env python3
# ==============================
# 运行所有策略回测的主程序
# ==============================
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_cli_utils import (
    create_engine,
    load_data,
    save_detailed_results,
    load_strategy_params,
    load_strategy_config,
)


def main():
    """主函数"""
    print("开始策略回测")
    print("=" * 60)

    # 加载数据
    csv_path = "stock_data/ETHUSDT_kline_20250101_20251228.csv"
    try:
        df = load_data(csv_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 初始化回测引擎（按 EMA 配置的 instrument 止损阈值）
    print("\n初始化回测引擎...")
    ema_cfg = load_strategy_config("EMA均值回归策略") or {}
    ema_instrument = ema_cfg.get("instrument") or {}
    stop_loss_threshold = ema_instrument.get("stop_loss_threshold", 50)
    engine = create_engine(
        stop_loss_threshold=stop_loss_threshold,
        # stop_loss_policy=None,  # 关闭止损；开启止损请注释/删除该行
    )

    # 运行所有策略回测，传入参数加载函数
    print("\n开始回测所有策略...")
    from strategy import get_strategy_list
    for strategy_name in get_strategy_list():
        cfg = load_strategy_config(strategy_name)
        if cfg:
            symbol = cfg.get('symbol') or 'N/A'
            instrument = cfg.get('instrument') or {}
            params = cfg.get('strategy_params')
            if params:
                print(f"  参数配置: {strategy_name} ({symbol}) -> {params}")
            else:
                print(f"  参数配置: {strategy_name} ({symbol}) -> 使用默认参数")
            if instrument:
                print(f"  交易配置: {strategy_name} ({symbol}) -> {instrument}")
        else:
            print(f"  参数配置: {strategy_name} -> 使用默认参数")
    results = engine.backtest_all_strategies(df, load_params_fn=load_strategy_params)

    # 生成报告
    print("\n生成回测报告...")
    report_path = engine.generate_report(results, 'backtest_report.html')

    # 保存详细结果
    print("\n保存详细结果...")
    save_detailed_results(results, summary_filename="summary.json")

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
            monthly_returns = stats.get('monthly_returns') or {}
            if monthly_returns:
                monthly_str = ", ".join(
                    [f"{month}:{value:+.2f}%" for month, value in monthly_returns.items()]
                )
                print(f"{'':<15} 月度收益: {monthly_str}")

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
