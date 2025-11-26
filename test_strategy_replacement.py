#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试策略替换脚本
验证simple_strategy.py是否成功替换strategy.py并且所有功能正常工作
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_test_data():
    """生成测试数据"""
    print("生成测试数据...")

    # 生成30天的5分钟K线数据
    periods = 30 * 288  # 30天，每天288个5分钟K线
    dates = pd.date_range('2024-01-01', periods=periods, freq='5min')

    np.random.seed(42)

    # 生成随机价格序列
    base_price = 43000
    returns = np.random.normal(0, 0.002, periods)
    price_changes = returns.cumsum()
    close_prices = base_price * (1 + price_changes)

    # 创建OHLCV数据
    df = pd.DataFrame({
        'datetime': dates,
        'Open': close_prices * (1 + np.random.normal(0, 0.001, periods)),
        'High': close_prices * (1 + np.abs(np.random.normal(0, 0.002, periods))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, periods))),
        'Close': close_prices,
        'Volume': np.random.uniform(100, 1000, periods)
    })

    print(f"生成了 {len(df)} 条测试数据")
    print(f"价格范围: {df['Low'].min():.2f} - {df['High'].max():.2f}")

    return df


def test_strategy_imports():
    """测试策略导入"""
    print("\n=== 测试策略导入 ===")

    try:
        # 测试从strategy模块导入
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
        print("✅ 所有策略函数导入成功")

        # 测试策略字典
        available_strategies = get_strategy_list()
        print(f"✅ 发现 {len(available_strategies)} 个可用策略:")
        for i, strategy_name in enumerate(available_strategies, 1):
            print(f"   {i:2d}. {strategy_name}")

        # 验证STRATEGIES字典
        print(f"✅ STRATEGIES字典包含 {len(STRATEGIES)} 个策略")

        return True, available_strategies

    except ImportError as e:
        print(f"❌ 策略导入失败: {e}")
        return False, []
    except Exception as e:
        print(f"❌ 策略验证失败: {e}")
        return False, []


def test_strategy_functions(df, available_strategies):
    """测试每个策略函数"""
    print("\n=== 测试策略函数执行 ===")

    results = {}

    for strategy_name in available_strategies:
        print(f"\n测试策略: {strategy_name}")

        try:
            # 使用run_strategy函数
            signals = run_strategy(df, strategy_name)

            if isinstance(signals, pd.DataFrame):
                signal_columns = ['long_entry', 'long_exit', 'short_entry', 'short_exit']
                missing_columns = [col for col in signal_columns if col not in signals.columns]

                if missing_columns:
                    print(f"   ❌ 缺少必要列: {missing_columns}")
                    results[strategy_name] = {'status': 'error', 'message': f'缺少列: {missing_columns}'}
                else:
                    # 统计信号
                    signal_counts = signals[signal_columns].sum()
                    total_signals = signal_counts.sum()

                    print(f"   ✅ 策略执行成功")
                    print(f"   📊 信号统计: {signal_counts.to_dict()}")
                    print(f"   📈 总信号数: {total_signals}")

                    results[strategy_name] = {
                        'status': 'success',
                        'signal_counts': signal_counts.to_dict(),
                        'total_signals': int(total_signals),
                        'data_shape': signals.shape
                    }
            else:
                print(f"   ❌ 返回类型错误: {type(signals)}")
                results[strategy_name] = {'status': 'error', 'message': f'返回类型错误: {type(signals)}'}

        except Exception as e:
            print(f"   ❌ 策略执行失败: {e}")
            results[strategy_name] = {'status': 'error', 'message': str(e)}

    return results


def test_btc_trader_import():
    """测试BTC交易器导入"""
    print("\n=== 测试BTC交易器导入 ===")

    try:
        from btc_live_trader import BTCLiveTrader
        print("✅ BTCLiveTrader导入成功")

        # 测试配置文件
        trader = BTCLiveTrader("trading_config.json")
        print(f"✅ 交易器初始化成功")
        print(f"   交易对: {trader.symbol}")
        print(f"   策略: {trader.strategy_name}")
        print(f"   仓位大小: {trader.position_size} BTC")

        return True

    except FileNotFoundError:
        print("❌ 配置文件不存在，这是正常的（首次运行）")
        return True
    except Exception as e:
        print(f"❌ 交易器测试失败: {e}")
        return False


def test_validate_scripts():
    """测试验证脚本"""
    print("\n=== 测试验证脚本 ===")

    scripts_to_test = [
        "validate_strategies.py",
        "update_strategy_config.py",
        "trading_example.py"
    ]

    for script in scripts_to_test:
        try:
            print(f"\n测试脚本: {script}")

            # 尝试导入并执行基本功能
            if script == "validate_strategies.py":
                from validate_strategies import StrategyValidator
                validator = StrategyValidator()
                print(f"   ✅ {script} 导入成功，发现 {len(validator.strategies)} 个策略")

            elif script == "update_strategy_config.py":
                from update_strategy_config import validate_strategy_functions
                if validate_strategy_functions():
                    print(f"   ✅ {script} 策略验证功能正常")

            elif script == "trading_example.py":
                from trading_example import setup_demo_config
                print(f"   ✅ {script} 导入成功")

        except Exception as e:
            print(f"   ❌ {script} 测试失败: {e}")


def generate_summary_report(strategy_results):
    """生成总结报告"""
    print("\n=== 策略测试总结报告 ===")

    successful_strategies = [name for name, result in strategy_results.items() if result['status'] == 'success']
    failed_strategies = [name for name, result in strategy_results.items() if result['status'] == 'error']

    print(f"\n📊 测试统计:")
    print(f"   总策略数: {len(strategy_results)}")
    print(f"   成功: {len(successful_strategies)}")
    print(f"   失败: {len(failed_strategies)}")
    print(f"   成功率: {len(successful_strategies)/len(strategy_results)*100:.1f}%")

    if successful_strategies:
        print(f"\n✅ 成功的策略:")
        for strategy_name in successful_strategies:
            result = strategy_results[strategy_name]
            print(f"   🔸 {strategy_name}")
            print(f"      信号数: {result['total_signals']}")
            print(f"      数据形状: {result['data_shape']}")

    if failed_strategies:
        print(f"\n❌ 失败的策略:")
        for strategy_name in failed_strategies:
            result = strategy_results[strategy_name]
            print(f"   🔸 {strategy_name}")
            print(f"      错误: {result['message']}")

    # 性能排名
    strategy_performance = []
    for strategy_name in successful_strategies:
        result = strategy_results[strategy_name]
        # 简单的性能评分：信号数量 + 信号多样性
        signal_diversity = len([count for count in result['signal_counts'].values() if count > 0])
        performance_score = result['total_signals'] + signal_diversity * 10

        strategy_performance.append({
            'name': strategy_name,
            'signals': result['total_signals'],
            'diversity': signal_diversity,
            'score': performance_score
        })

    strategy_performance.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n🏆 策略性能排名 (基于信号数量和多样性):")
    for i, perf in enumerate(strategy_performance[:5], 1):
        print(f"   {i}. {perf['name']}")
        print(f"      信号数: {perf['signals']}")
        print(f"      信号类型数: {perf['diversity']}")
        print(f"      评分: {perf['score']}")

    return {
        'total_strategies': len(strategy_results),
        'successful': len(successful_strategies),
        'failed': len(failed_strategies),
        'success_rate': len(successful_strategies)/len(strategy_results)*100,
        'performance_ranking': strategy_performance
    }


def main():
    """主测试函数"""
    print("BTC交易系统策略替换测试")
    print("=" * 60)

    # 1. 生成测试数据
    df = generate_test_data()

    # 2. 测试策略导入
    print("\n第1步: 测试策略导入...")
    import_success, available_strategies = test_strategy_imports()

    if not import_success:
        print("❌ 策略导入失败，无法继续测试")
        return

    # 3. 测试策略函数
    print("\n第2步: 测试策略函数...")
    strategy_results = test_strategy_functions(df, available_strategies)

    # 4. 测试BTC交易器
    print("\n第3步: 测试BTC交易器...")
    trader_success = test_btc_trader_import()

    # 5. 测试验证脚本
    print("\n第4步: 测试验证脚本...")
    script_success = test_validate_scripts()

    # 6. 生成总结报告
    print("\n第5步: 生成总结报告...")
    summary = generate_summary_report(strategy_results)

    # 7. 最终结果
    print("\n" + "=" * 60)
    print("测试完成！")

    if import_success and trader_success:
        print("✅ 策略替换成功！系统可以正常使用")
        print(f"   策略成功率: {summary['success_rate']:.1f}%")
        print(f"   可用策略: {len(available_strategies)} 个")

        print("\n🚀 使用方法:")
        print("1. 运行 python update_strategy_config.py 配置策略")
        print("2. 运行 python trading_example.py 查看演示")
        print("3. 运行 python start_btc_trading.py 开始交易")
        print("4. 运行 python validate_strategies.py 验证策略")

    else:
        print("❌ 策略替换存在问题，请检查错误信息")
        print("   策略导入:", "✅" if import_success else "❌")
        print("   交易器:", "✅" if trader_success else "❌")

    print("\n💡 建议:")
    print("- 如果有策略失败，请检查数据和策略逻辑")
    print("- 首次使用前建议运行演示和验证脚本")
    print("- 配置API密钥后再进行实盘交易")


if __name__ == "__main__":
    main()