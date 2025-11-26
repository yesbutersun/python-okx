#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX沙盒环境配置脚本
专门用于配置和测试OKX沙盒环境的交易功能
"""

import json


def get_sandbox_api_info():
    """获取沙盒API信息"""
    print("=== OKX沙盒环境API配置 ===\n")

    print("🔒 沙盒环境说明:")
    print("   - 使用测试资金，无真实资金风险")
    print("   - API接口与生产环境完全一致")
    print("   - 需要单独申请沙盒API密钥")
    print("   - 定期重置测试数据和余额")
    print()

    print("📋 获取沙盒API密钥步骤:")
    print("   1. 访问OKX官网 (www.okx.com)")
    print("   2. 登录账户，进入API管理")
    print("   3. 创建新的API Key")
    print("   4. 选择沙盒环境(Sandbox)")
    print("   5. 设置权限: 交易+读取")
    print("   6. 记录API Key、Secret Key、Passphrase")
    print()

    print("🌐 沙盒环境信息:")
    print("   - API域名: https://www.okx.com")
    print("   - 环境标记: flag = '0'")
    print("   - 测试代币会自动发放")
    print()

    # 获取用户输入
    api_key = input("请输入沙盒API Key: ").strip()
    secret_key = input("请输入沙盒Secret Key: ").strip()
    passphrase = input("请输入沙盒Passphrase: ").strip()

    if not all([api_key, secret_key, passphrase]):
        print("❌ API密钥信息不完整，请重新填写")
        return None

    return api_key, secret_key, passphrase


def get_sandbox_balance_info():
    """获取沙盒余额信息"""
    print("\n=== 沙盒环境资金信息 ===")

    print("💡 沙盒环境特点:")
    print("   - 自动提供100,000 USDT测试资金")
    print("   - 支持所有主流交易对")
    print("   - 杠杆倍数: 1-125倍")
    print("   - 交易手续费: 与生产环境相同")
    print("   - 数据延迟: 与生产环境相同")
    print()


def select_sandbox_strategy():
    """选择沙盒测试策略"""
    print("\n=== 选择沙盒测试策略 ===")

    strategies = {
        "1": {
            "name": "conservative_test",
            "display": "保守测试策略",
            "description": "RSI反转，小仓位，低杠杆",
            "config": {
                "strategy": "rsi_reversal",
                "position_size": 0.0001,  # 极小仓位
                "leverage": 2,           # 低杠杆
                "max_positions": 1
            }
        },
        "2": {
            "name": "trend_test",
            "display": "趋势测试策略",
            "description": "趋势ATR，中等仓位，中等杠杆",
            "config": {
                "strategy": "trend_atr",
                "position_size": 0.0005,  # 小仓位
                "leverage": 5,           # 中等杠杆
                "max_positions": 1
            }
        },
        "3": {
            "name": "balanced_test",
            "display": "均衡测试策略",
            "description": "布林RSI，中等仓位，中等杠杆",
            "config": {
                "strategy": "boll_rsi",
                "position_size": 0.001,   # 标准测试仓位
                "leverage": 3,           # 中等杠杆
                "max_positions": 1
            }
        },
        "4": {
            "name": "custom_test",
            "display": "自定义策略",
            "description": "自定义所有参数",
            "config": {}
        }
    }

    for key, info in strategies.items():
        print(f"{key}. {info['display']}")
        print(f"   {info['description']}")
        if 'position_size' in info['config']:
            print(f"   仓位: {info['config']['position_size']} BTC, 杠杆: {info['config']['leverage']}x")
        print()

    choice = input("请选择策略 (1-4, 推荐1): ").strip() or "1"

    if choice in strategies:
        selected = strategies[choice]
        print(f"✅ 已选择: {selected['display']}")

        if choice == "4":  # 自定义策略
            return customize_strategy()

        return selected['config']
    else:
        print("❌ 无效选择，使用默认保守策略")
        return strategies["1"]["config"]


def customize_strategy():
    """自定义策略配置"""
    print("\n=== 自定义策略配置 ===")

    print("可选策略:")
    print("1. rsi_reversal - RSI反转策略")
    print("2. trend_atr - 趋势ATR策略")
    print("3. boll_rsi - 布林RSI策略")
    print("4. trend_volatility_stop - 趋势波动止损策略")
    print("5. breakout - 突破策略")
    print("6. mean_reversion - 均值回归策略")
    print("7. momentum - 动量策略")
    print("8. macd - MACD策略")

    strategy_mapping = {
        "1": "rsi_reversal",
        "2": "trend_atr",
        "3": "boll_rsi",
        "4": "trend_volatility_stop",
        "5": "breakout",
        "6": "mean_reversion",
        "7": "momentum",
        "8": "macd"
    }

    strategy_choice = input("选择策略 (1-8): ").strip()
    strategy = strategy_mapping.get(strategy_choice, "rsi_reversal")

    try:
        position_size = float(input("仓位大小 (BTC, 推荐0.0001-0.001): ").strip() or "0.0001")
        leverage = int(input("杠杆倍数 (1-20, 推荐1-5): ").strip() or "2")

        # 限制参数范围
        position_size = min(max(position_size, 0.0001), 0.01)  # 0.0001-0.01 BTC
        leverage = max(1, min(leverage, 20))  # 1-20倍

        return {
            "strategy": strategy,
            "position_size": position_size,
            "leverage": leverage,
            "max_positions": 1
        }

    except ValueError:
        print("❌ 输入格式错误，使用默认值")
        return {
            "strategy": "rsi_reversal",
            "position_size": 0.0001,
            "leverage": 2,
            "max_positions": 1
        }


def create_sandbox_config():
    """创建沙盒配置文件"""
    print("🏗️  创建沙盒交易配置")
    print("=" * 50)

    # 获取API信息
    api_info = get_sandbox_api_info()
    if not api_info:
        return None

    api_key, secret_key, passphrase = api_info

    # 获取资金信息
    get_sandbox_balance_info()

    # 选择策略
    strategy_config = select_sandbox_strategy()

    # 创建配置
    config = {
        "api_key": api_key,
        "secret_key": secret_key,
        "passphrase": passphrase,
        "domain": "https://www.okx.com",
        "flag": "0",  # 0 = 沙盒环境
        "environment": "sandbox",
        "symbol": "BTC-USDT-SWAP",
        "strategy": strategy_config["strategy"],
        "trade_mode": "cross",
        "position_size": strategy_config["position_size"],
        "max_positions": strategy_config["max_positions"],
        "leverage": strategy_config["leverage"],
        "timeframe": "5m",
        "data_limit": 100,
        "risk_management": {
            "max_loss_per_trade": 0.001,  # 0.1% 单笔最大亏损
            "max_daily_loss": 0.005,   # 0.5% 每日最大亏损
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_atr_multiplier": 2.0
        },
        "sandbox_config": {
            "reset_balance_daily": True,
            "log_all_trades": True,
            "paper_trading": False,  # 真实API调用，但用测试资金
            "test_mode": True
        }
    }

    return config


def save_sandbox_config(config, config_file="sandbox_trading_config.json"):
    """保存沙盒配置文件"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"\n✅ 沙盒配置文件已保存: {config_file}")
        print(f"🔒 环境: 沙盒测试环境")
        print(f"💰 策略: {config['strategy']}")
        print(f"📊 仓位: {config['position_size']} BTC")
        print(f"⚡ 杠杆: {config['leverage']}x")
        print(f"🎯 单笔最大风险: {config['risk_management']['max_loss_per_trade']*100:.2f}%")

        # 安全提醒
        print(f"\n🔒 沙盒环境安全特性:")
        print(f"   - 使用测试资金，无真实风险")
        print(f"   - API权限限制在沙盒环境")
        print(f"   - 交易数据不真实记录")
        print(f"   - 定期重置账户余额")

        return config_file

    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")
        return None


def test_sandbox_config(config_file):
    """测试沙盒配置"""
    print(f"\n🧪 测试沙盒配置: {config_file}")

    try:
        from btc_live_trader import BTCLiveTrader

        # 初始化交易器
        trader = BTCLiveTrader(config_file)

        print("✅ 交易器初始化成功")
        print(f"   环境: {trader.environment}")
        print(f"   交易对: {trader.symbol}")
        print(f"   策略: {trader.strategy_name}")

        # 测试API连接
        current_price = trader.get_current_price()
        if current_price > 0:
            print(f"✅ API连接成功")
            print(f"   当前BTC价格: ${current_price:,.2f}")
        else:
            print("❌ API连接失败")
            return False

        # 测试市场数据获取
        df = trader.get_market_data(10)
        if not df.empty:
            print(f"✅ 市场数据获取成功")
            print(f"   数据条数: {len(df)}")
        else:
            print("❌ 市场数据获取失败")
            return False

        print("✅ 沙盒配置测试通过")
        return True

    except Exception as e:
        print(f"❌ 沙盒配置测试失败: {e}")
        return False


def main():
    """主函数"""
    print("OKX沙盒环境交易配置工具")
    print("=" * 60)
    print("🔒 专门用于安全测试的沙盒环境配置")
    print("=" * 60)

    # 创建配置
    config = create_sandbox_config()
    if not config:
        print("❌ 配置创建失败")
        return

    # 保存配置
    config_file = save_sandbox_config(config)
    if not config_file:
        print("❌ 配置保存失败")
        return

    # 测试配置
    test_result = test_sandbox_config(config_file)

    print("\n" + "=" * 60)
    print("🚀 配置完成!")

    if test_result:
        print("✅ 沙盒环境配置成功，可以开始测试交易")
        print("\n📋 下一步操作:")
        print(f"1. 运行沙盒交易: python btc_live_trader.py --config {config_file}")
        print("2. 查看策略演示: python trading_example.py")
        print("3. 验证策略: python validate_strategies.py")
        print("4. 使用管理界面: python start_btc_trading.py")
    else:
        print("❌ 配置测试失败，请检查API密钥和网络连接")

    print("\n💡 沙盒环境使用建议:")
    print("   - 先用极小仓位测试策略有效性")
    print("   - 验证止盈止损功能是否正常")
    print("   - 测试网络连接稳定性")
    print("   - 熟悉交易界面和操作流程")
    print("   - 确认策略在沙盒环境表现良好后再考虑生产环境")


if __name__ == "__main__":
    main()
