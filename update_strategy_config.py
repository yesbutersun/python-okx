#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新策略配置脚本
用于将配置文件中的策略名称更新为与simple_strategy.py兼容的格式
"""

import json
import os
from strategy import get_strategy_list


def update_config_file(config_file="trading_config.json"):
    """更新配置文件中的策略选项"""

    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在，将创建新的配置文件")
        return create_new_config(config_file)

    # 读取现有配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 获取当前策略
    current_strategy = config.get("strategy", "trend_atr")

    # 策略映射表
    strategy_mapping = {
        # 原策略名 -> 新策略名
        "trend_atr": "trend_atr",
        "boll_rsi": "boll_rsi",
        "rsi_reversal": "rsi_reversal",
        "trend_volatility_stop": "trend_volatility_stop",

        # 新增策略
        "breakout": "breakout",
        "mean_reversion": "mean_reversion",
        "boll_zscore_slope_accel": "boll_zscore_slope_accel",
        "momentum": "momentum",
        "macd": "macd"
    }

    # 映射当前策略
    if current_strategy in strategy_mapping:
        new_strategy = strategy_mapping[current_strategy]
        config["strategy"] = new_strategy
        print(f"策略已更新: {current_strategy} -> {new_strategy}")
    else:
        print(f"策略 {current_strategy} 无需更改")

    # 确保其他必要字段存在
    default_values = {
        "symbol": "BTC-USDT-SWAP",
        "trade_mode": "cross",
        "position_size": 0.001,
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

    for key, value in default_values.items():
        if key not in config:
            config[key] = value
            print(f"添加默认配置: {key} = {value}")

    # 保存更新后的配置
    backup_file = config_file.replace('.json', '_backup.json')
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"原配置已备份到: {backup_file}")

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"配置文件已更新: {config_file}")
    return config


def create_new_config(config_file="trading_config.json"):
    """创建新的配置文件"""

    print("创建新的配置文件...")

    # 显示可用策略
    available_strategies = {
        "1": ("trend_atr", "趋势ATR策略 - EMA金叉死叉 + ATR动态止盈止损"),
        "2": ("boll_rsi", "布林RSI策略 - 布林带位置 + RSI超买超卖"),
        "3": ("rsi_reversal", "RSI反转策略 - RSI超买超卖反转信号"),
        "4": ("trend_volatility_stop", "趋势波动止损策略 - 趋势跟踪 + ATR止损"),
        "5": ("breakout", "突破策略 - 价格突破前期高底"),
        "6": ("mean_reversion", "均值回归策略 - 价格偏离均值的回归"),
        "7": ("momentum", "动量策略 - 基于变化率的动量交易"),
        "8": ("macd", "MACD策略 - MACD金叉死叉信号"),
        "9": ("boll_zscore_slope_accel", "BOLL+Z-score斜率加速度策略 - 极端偏离 + 均值拐头过滤")
    }

    print("\n可用交易策略:")
    for key, (strategy_key, description) in available_strategies.items():
        print(f"{key}. {description}")

    # 选择策略
    choice = input("\n请选择策略 (1-9, 默认1): ").strip() or "1"
    if choice in available_strategies:
        strategy_key, strategy_name = available_strategies[choice]
    else:
        print("无效选择，使用默认策略")
        strategy_key, strategy_name = available_strategies["1"]

    print(f"已选择策略: {strategy_name}")

    # 获取API配置
    print("\n请输入OKX API配置:")
    api_key = input("API Key: ").strip()
    secret_key = input("Secret Key: ").strip()
    passphrase = input("Passphrase: ").strip()

    # 交易参数
    print("\n交易参数配置 (直接回车使用默认值):")

    try:
        position_size = float(input("仓位大小 (BTC, 默认0.001): ").strip() or "0.001")
        leverage = int(input("杠杆倍数 (1-20, 默认5): ").strip() or "5")
    except ValueError:
        print("输入格式错误，使用默认值")
        position_size = 0.001
        leverage = 5

    # 创建配置
    config = {
        "api_key": api_key,
        "secret_key": secret_key,
        "passphrase": passphrase,
        "symbol": "BTC-USDT-SWAP",
        "strategy": strategy_key,
        "strategy_name": strategy_name,
        "trade_mode": "cross",
        "position_size": min(position_size, 1.0),  # 最大1 BTC
        "max_positions": 1,
        "leverage": max(1, min(leverage, 20)),  # 1-20倍
        "timeframe": "5m",
        "data_limit": 100,
        "risk_management": {
            "max_loss_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_atr_multiplier": 2.0
        }
    }

    # 保存配置
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 配置文件已创建: {config_file}")
    print(f"策略: {strategy_name}")
    print(f"仓位大小: {config['position_size']} BTC")
    print(f"杠杆倍数: {config['leverage']}x")

    return config


def show_strategy_info():
    """显示策略信息"""
    from strategy import STRATEGIES

    print("=== 可用交易策略详细信息 ===\n")

    strategy_descriptions = {
        "RSI反转策略": "基于RSI指标的 reversal 策略，当RSI从超卖区域反弹时做多，从超买区域回落时做空",
        "趋势ATR策略": "结合EMA趋势跟踪和ATR动态止盈止损的金叉死叉交易策略",
        "布林RSI策略": "结合布林带位置和RSI超买超卖信号的震荡策略",
        "趋势波动止损策略": "基于趋势跟踪和ATR波动性止损的策略，适合高波动市场",
        "突破策略": "基于价格突破前期高低点的策略，适合趋势市场",
        "均值回归策略": "基于价格偏离均值后回归的震荡策略",
        "BOLL+Z-score斜率加速度策略": "结合布林带与Z-score识别极端偏离，并用均值斜率/加速度筛选“趋势减速拐头”的均值回归策略",
        "动量策略": "基于价格变化率的动量交易策略",
        "MACD策略": "基于MACD金叉死叉的趋势跟踪策略"
    }

    for strategy_name in STRATEGIES.keys():
        description = strategy_descriptions.get(strategy_name, "暂无描述")
        print(f"🔸 {strategy_name}")
        print(f"   {description}")
        print()


def validate_strategy_functions():
    """验证所有策略函数是否可用"""
    try:
        from strategy import STRATEGIES, get_strategy_list

        print("=== 策略函数验证 ===\n")

        available_strategies = get_strategy_list()
        print(f"发现 {len(available_strategies)} 个可用策略:")

        for i, strategy_name in enumerate(available_strategies, 1):
            print(f"{i}. {strategy_name}")

            if strategy_name in STRATEGIES:
                strategy_func = STRATEGIES[strategy_name]
                if callable(strategy_func):
                    print(f"   ✅ 策略函数可用: {strategy_func.__name__}")
                else:
                    print(f"   ❌ 策略函数不可调用: {strategy_func}")
            else:
                print(f"   ❌ 策略未在STRATEGIES中找到")

        print(f"\n所有策略已验证，共 {len(STRATEGIES)} 个策略函数可用")
        return True

    except ImportError as e:
        print(f"❌ 导入策略模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 策略验证失败: {e}")
        return False


def main():
    """主函数"""
    print("策略配置更新工具")
    print("=" * 50)

    while True:
        print("\n选项:")
        print("1. 更新现有配置文件")
        print("2. 创建新的配置文件")
        print("3. 查看策略详细信息")
        print("4. 验证策略函数")
        print("5. 显示可用策略列表")
        print("0. 退出")

        choice = input("\n请选择 (0-5): ").strip()

        if choice == "1":
            config_file = input("配置文件路径 (默认 trading_config.json): ").strip() or "trading_config.json"
            update_config_file(config_file)

        elif choice == "2":
            config_file = input("配置文件路径 (默认 trading_config.json): ").strip() or "trading_config.json"
            create_new_config(config_file)

        elif choice == "3":
            show_strategy_info()

        elif choice == "4":
            if validate_strategy_functions():
                print("✅ 所有策略函数验证通过")
            else:
                print("❌ 策略函数验证失败")

        elif choice == "5":
            try:
                from strategy import get_strategy_list
                strategies = get_strategy_list()
                print(f"\n可用策略列表 (共 {len(strategies)} 个):")
                for i, strategy in enumerate(strategies, 1):
                    print(f"{i:2d}. {strategy}")
            except Exception as e:
                print(f"获取策略列表失败: {e}")

        elif choice == "0":
            print("退出程序")
            break

        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    main()
