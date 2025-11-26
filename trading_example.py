#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC实盘交易使用示例
演示如何使用btc_live_trader.py进行实盘交易
"""

import os
import json
from btc_live_trader import BTCLiveTrader
from strategy import (
    STRATEGIES, get_strategy_list, run_strategy,
    trend_atr_signal, boll_rsi_signal, rsi_reversal_strategy,
    breakout_strategy, mean_reversion_strategy, momentum_strategy, macd_strategy,
    trend_volatility_stop_signal
)
import pandas as pd


def setup_demo_config():
    """创建演示配置文件"""
    config = {
        "api_key": "your_api_key_here",
        "secret_key": "your_secret_key_here",
        "passphrase": "your_passphrase_here",
        "symbol": "BTC-USDT-SWAP",
        "strategy": "trend_atr",
        "trade_mode": "cross",
        "position_size": 0.001,  # 0.001 BTC，适合小额测试
        "max_positions": 1,
        "leverage": 5,
        "timeframe": "5m",
        "data_limit": 100,
        "risk_management": {
            "max_loss_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_atr_multiplier": 2.0
        }
    }

    config_file = "demo_trading_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"已创建演示配置文件: {config_file}")
    print("请修改配置文件中的API密钥信息")
    return config_file


def demo_strategy_backtest():
    """演示策略回测"""
    print("=== 策略回测演示 ===")

    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    import numpy as np
    np.random.seed(42)

    # 模拟BTC价格走势
    base_price = 43000
    price_changes = np.random.normal(0, 0.002, 100).cumsum()
    close_prices = base_price * (1 + price_changes)

    df = pd.DataFrame({
        'Open': close_prices * (1 + np.random.normal(0, 0.001, 100)),
        'High': close_prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
        'Close': close_prices,
        'Volume': np.random.uniform(100, 1000, 100)
    }, index=dates)

    # 确保High >= Close >= Low
    df['High'] = df[['High', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Close']].min(axis=1)

    print(f"生成了 {len(df)} 条示例K线数据")
    print(f"价格范围: {df['Low'].min():.2f} - {df['High'].max():.2f}")

    # 测试不同策略
    strategies = {
        'trend_atr': lambda df: trend_atr_signal(df),
        'boll_rsi': lambda df: boll_rsi_signal(df),
        'rsi_reversal': lambda df: rsi_reversal_strategy(df),
        'trend_volatility_stop': lambda df: trend_volatility_stop_signal(df),
        'breakout': lambda df: breakout_strategy(df),
        'mean_reversion': lambda df: mean_reversion_strategy(df),
        'momentum': lambda df: momentum_strategy(df),
        'macd': lambda df: macd_strategy(df)
    }

    for strategy_name, strategy_func in strategies.items():
        print(f"\n--- 测试策略: {strategy_name} ---")
        try:
            signals = strategy_func(df)

            # 统计信号
            signal_counts = signals.sum()
            print("信号统计:")
            print(signal_counts)

            # 显示信号详情
            signal_dates = signals[signals.any(axis=1)].index
            if len(signal_dates) > 0:
                print(f"信号时间点数量: {len(signal_dates)}")
                print("前5个信号时间点:")
                for i, date in enumerate(signal_dates[:5]):
                    row_signals = signals.loc[date][signals.loc[date]].index.tolist()
                    price = df.loc[date, 'Close']
                    print(f"  {date}: {row_signals} @ {price:.2f}")
            else:
                print("未检测到交易信号")

        except Exception as e:
            print(f"策略测试失败: {e}")


def demo_trader_initialization():
    """演示交易器初始化"""
    print("\n=== 交易器初始化演示 ===")

    # 首先创建配置文件
    config_file = setup_demo_config()

    print("\n正在初始化交易器...")
    print("注意: 由于没有配置真实的API密钥，初始化会失败")
    print("这只是为了演示流程")

    try:
        trader = BTCLiveTrader(config_file)

        print(f"交易对: {trader.symbol}")
        print(f"策略: {trader.strategy_name}")
        print(f"仓位大小: {trader.position_size} BTC")
        print(f"杠杆: {trader.leverage}x")
        print(f"交易模式: {trader.trade_mode}")

        print("\n交易器初始化成功（演示完成）")

    except Exception as e:
        print(f"预期的初始化错误: {e}")
        print("这是正常的，因为API密钥未配置")


def demo_report_generation():
    """演示报告生成"""
    print("\n=== 报告生成演示 ===")

    # 创建模拟交易记录
    from btc_live_trader import TradeRecord
    from datetime import datetime, timedelta

    # 模拟一些交易记录
    demo_trades = [
        TradeRecord(
            timestamp=(datetime.now() - timedelta(hours=2)).isoformat(),
            trade_id="demo001",
            symbol="BTC-USDT-SWAP",
            side="buy",
            order_type="market",
            size=0.001,
            price=43250.0,
            amount=43.25,
            strategy="trend_atr",
            signal_type="long_entry",
            pnl=2.15,
            balance=1000.0,
            notes="做多开仓"
        ),
        TradeRecord(
            timestamp=(datetime.now() - timedelta(hours=1)).isoformat(),
            trade_id="demo002",
            symbol="BTC-USDT-SWAP",
            side="sell",
            order_type="market",
            size=0.001,
            price=43500.0,
            amount=43.5,
            strategy="trend_atr",
            signal_type="long_exit",
            pnl=2.5,
            balance=1002.5,
            notes="做多平仓，盈利"
        ),
        TradeRecord(
            timestamp=(datetime.now() - timedelta(minutes=30)).isoformat(),
            trade_id="demo003",
            symbol="BTC-USDT-SWAP",
            side="sell",
            order_type="market",
            size=0.001,
            price=43400.0,
            amount=43.4,
            strategy="boll_rsi",
            signal_type="short_entry",
            pnl=0.0,
            balance=1002.5,
            notes="做空开仓"
        )
    ]

    # 计算统计数据
    total_trades = len(demo_trades)
    total_pnl = sum(trade.pnl for trade in demo_trades)
    winning_trades = [t for t in demo_trades if t.pnl > 0]
    losing_trades = [t for t in demo_trades if t.pnl < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    print("=== 模拟交易报告 ===")
    print(f"总交易次数: {total_trades}")
    print(f"总盈亏: {total_pnl:.2f} USDT")
    print(f"盈利交易: {len(winning_trades)}")
    print(f"亏损交易: {len(losing_trades)}")
    print(f"胜率: {win_rate*100:.1f}%")

    print("\n=== 交易明细 ===")
    for i, trade in enumerate(demo_trades, 1):
        print(f"{i}. {trade.timestamp[:19]} {trade.side} {trade.size} BTC @ {trade.price} USDT")
        print(f"   策略: {trade.strategy}, 信号: {trade.signal_type}")
        print(f"   盈亏: {trade.pnl:+.2f} USDT, 余额: {trade.balance:.2f} USDT")
        print(f"   备注: {trade.notes}")
        print()


def main():
    """主演示函数"""
    print("BTC实盘交易系统演示")
    print("=" * 50)

    print("本演示包含以下内容:")
    print("1. 策略回测演示")
    print("2. 交易器初始化演示")
    print("3. 交易报告生成演示")
    print()

    try:
        # 1. 策略回测演示
        demo_strategy_backtest()

        # 2. 交易器初始化演示
        demo_trader_initialization()

        # 3. 报告生成演示
        demo_report_generation()

        print("\n" + "=" * 50)
        print("演示完成!")
        print()
        print("使用说明:")
        print("1. 修改 demo_trading_config.json 中的API密钥")
        print("2. 运行 python btc_live_trader.py 开始实盘交易")
        print("3. 交易记录将保存在相应的JSON和CSV文件中")
        print()
        print("风险提示:")
        print("- 数字货币交易存在高风险，请谨慎投资")
        print("- 建议先在测试环境或小额资金测试")
        print("- 请确保理解策略逻辑和风险控制措施")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()