#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC实盘交易启动脚本
提供简单易用的界面来启动和配置BTC交易系统
"""

import os
import sys
import json
import subprocess
from datetime import datetime


def print_banner():
    """打印欢迎横幅"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    BTC实盘交易系统                            ║
    ║                    基于OKX API的智能交易                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """检查依赖包"""
    required_packages = ['pandas', 'okx', 'loguru']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ 所有依赖包已安装")
        return True


def setup_config():
    """设置配置文件"""
    config_file = "trading_config.json"

    if os.path.exists(config_file):
        print(f"\n配置文件 '{config_file}' 已存在")
        choice = input("是否重新配置? (y/n): ").lower()
        if choice != 'y':
            return config_file

    print("\n=== 配置OKX交易参数 ===")

    print("请获取OKX API密钥:")
    print("1. 登录OKX官网 -> API管理 -> 创建API")
    print("2. 需要开启交易权限")
    print("3. 建议设置IP白名单")

    # 环境选择
    print("\n=== 选择交易环境 ===")
    print("1. 生产环境 (实盘交易)")
    print("2. 沙盒环境 (测试环境)")

    env_choice = input("请选择环境 (1-2，推荐2): ").strip() or "2"

    if env_choice == "1":
        domain = "https://www.okx.com"  # 生产环境
        flag = "1"
        env_note = "生产环境 - 真实资金交易"
        print("⚠️  您选择了生产环境，将进行真实的资金交易!")
        confirm = input("确认选择生产环境? 输入 'PROD' 确认: ")
        if confirm != "PROD":
            print("取消生产环境配置")
            return None
    else:
        domain = "https://www.okx.com"  # 沙盒环境，域名相同但flag不同
        flag = "0"  # 沙盒环境标记
        env_note = "沙盒环境 - 测试资金交易"
        print("✅ 您选择了沙盒环境，将使用测试资金")

    print(f"\n当前配置: {env_note}")
    print(f"API域名: {domain}")
    print(f"环境标记: {flag}")

    config = {
        "api_key": input("API Key: ").strip(),
        "secret_key": input("Secret Key: ").strip(),
        "passphrase": input("Passphrase: ").strip(),
        "symbol": "BTC-USDT-SWAP",
        "strategy": "trend_atr",
        "trade_mode": "cross",
        "position_size": 0.001,
        "max_positions": 1,
        "leverage": 5,
        "timeframe": "5m",
        "data_limit": 100,
        "domain": domain,
        "flag": flag,
        "environment": "production" if env_choice == "1" else "sandbox"
    }

    # 策略选择
    print("\n可选交易策略:")
    strategies = {
        "1": "trend_atr",
        "2": "boll_rsi",
        "3": "rsi_reversal",
        "4": "trend_volatility_stop"
    }

    for key, value in strategies.items():
        print(f"{key}. {value}")

    strategy_choice = input("选择策略 (1-4，默认1): ").strip() or "1"
    if strategy_choice in strategies:
        config["strategy"] = strategies[strategy_choice]

    # 风险管理参数
    print("\n=== 风险管理设置 ===")

    try:
        position_size = float(input("仓位大小 (BTC, 默认0.001): ").strip() or "0.001")
        leverage = int(input("杠杆倍数 (1-20, 默认5): ").strip() or "5")

        config["position_size"] = min(position_size, 1.0)  # 最大1 BTC
        config["leverage"] = max(1, min(leverage, 20))  # 1-20倍
    except ValueError:
        print("输入格式错误，使用默认值")

    # 保存配置
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 配置已保存到 '{config_file}'")
    print("⚠️  请确保API密钥正确，资金充足")

    return config_file


def run_demo():
    """运行演示"""
    print("\n=== 运行交易系统演示 ===")

    try:
        result = subprocess.run([sys.executable, "trading_example.py"],
                              capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(result.stdout)
            print("✅ 演示运行成功")
        else:
            print("❌ 演示运行失败:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("⏰ 演示运行超时")
    except FileNotFoundError:
        print("❌ 找不到演示文件 trading_example.py")


def validate_strategies():
    """验证策略"""
    print("\n=== 验证交易策略 ===")

    try:
        result = subprocess.run([sys.executable, "validate_strategies.py"],
                              capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(result.stdout)
            print("✅ 策略验证完成")

            # 检查是否生成了报告文件
            if os.path.exists("strategy_validation_report.json"):
                print("📊 详细验证报告: strategy_validation_report.json")
        else:
            print("❌ 策略验证失败:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("⏰ 策略验证超时")
    except FileNotFoundError:
        print("❌ 找不到验证脚本 validate_strategies.py")


def start_paper_trading():
    """启动模拟交易"""
    print("\n=== 启动模拟交易 ===")
    print("⚠️  模拟交易仅用于测试，不会进行真实交易")

    # 修改配置为模拟模式
    config_file = setup_config()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 添加模拟交易标记
        config["paper_trading"] = True
        config["simulation_mode"] = True

        # 保存修改后的配置
        paper_config_file = "paper_trading_config.json"
        with open(paper_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"📝 模拟交易配置已保存到: {paper_config_file}")
        print("⚠️  这只是配置演示，实际模拟需要修改代码")

    except Exception as e:
        print(f"❌ 配置模拟交易失败: {e}")


def start_live_trading():
    """启动实盘交易"""
    print("\n=== 启动实盘交易 ===")
    print("⚠️  实盘交易涉及真实资金，请谨慎操作!")

    # 安全检查
    confirm = input("确认要进行实盘交易? 输入 'LIVE_TRADING' 确认: ")
    if confirm != "LIVE_TRADING":
        print("❌ 确认失败，取消实盘交易")
        return

    # 检查配置文件
    config_file = "trading_config.json"
    if not os.path.exists(config_file):
        print("❌ 配置文件不存在，请先配置")
        config_file = setup_config()

    # 验证配置
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if not all([config.get("api_key"), config.get("secret_key"), config.get("passphrase")]):
            print("❌ API密钥配置不完整，请检查配置文件")
            return

        print(f"✅ 配置验证通过")
        print(f"交易对: {config['symbol']}")
        print(f"策略: {config['strategy']}")
        print(f"仓位: {config['position_size']} BTC")
        print(f"杠杆: {config['leverage']}x")

    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        return

    # 启动交易
    print("\n🚀 启动实盘交易...")
    print("按 Ctrl+C 可以安全停止交易")

    try:
        # 这里可以调用实际的交易代码
        print("实盘交易功能需要额外的安全验证")
        print("请直接运行: python btc_live_trader.py")

        # subprocess.run([sys.executable, "btc_live_trader.py"])

    except KeyboardInterrupt:
        print("\n🛑 交易已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")


def show_reports():
    """显示交易报告"""
    print("\n=== 交易报告 ===")

    reports = []

    # 查找报告文件
    for file in os.listdir('.'):
        if file.endswith('.json') and any(keyword in file for keyword in ['report', 'record']):
            reports.append(file)

    if not reports:
        print("📝 暂无交易报告")
    else:
        print("📊 找到以下报告文件:")
        for i, report in enumerate(reports, 1):
            file_size = os.path.getsize(report)
            mod_time = datetime.fromtimestamp(os.path.getmtime(report))
            print(f"{i}. {report}")
            print(f"   大小: {file_size} 字节")
            print(f"   修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")


def show_help():
    """显示帮助信息"""
    print("""
📖 帮助信息

1. 首次使用:
   - 运行演示: python trading_example.py
   - 验证策略: python validate_strategies.py
   - 配置交易: 选择菜单选项 2

2. 重要提醒:
   - 实盘交易前请充分测试
   - 建议使用小额资金开始
   - 设置合理的止损止盈
   - 定期监控交易表现

3. 文件说明:
   - btc_live_trader.py: 主要交易逻辑
   - strategy.py: 交易策略实现
   - trading_example.py: 使用示例
   - validate_strategies.py: 策略验证工具

4. 安全建议:
   - 妥善保管API密钥
   - 设置IP白名单
   - 定期更换密码
   - 监控异常活动

5. 技术支持:
   - 查看日志文件: btc_trading.log
   - 检查配置文件: trading_config.json
   - 验证网络连接和API权限
    """)


def main():
    """主函数"""
    print_banner()

    # 检查依赖
    if not check_dependencies():
        return

    while True:
        print("\n" + "="*60)
        print("BTC实盘交易系统 - 主菜单")
        print("="*60)
        print("1. 🎬 运行演示")
        print("2. ⚙️  配置交易参数")
        print("3. 🔍 验证交易策略")
        print("4. 📈 启动模拟交易")
        print("5. 🚀 启动实盘交易")
        print("6. 📊 查看交易报告")
        print("7. ❓ 帮助信息")
        print("0. 🚪 退出")

        choice = input("\n请选择 (0-7): ").strip()

        if choice == "1":
            run_demo()
        elif choice == "2":
            setup_config()
        elif choice == "3":
            validate_strategies()
        elif choice == "4":
            start_paper_trading()
        elif choice == "5":
            start_live_trading()
        elif choice == "6":
            show_reports()
        elif choice == "7":
            show_help()
        elif choice == "0":
            print("\n👋 感谢使用BTC交易系统!")
            print("⚠️  交易有风险，投资需谨慎")
            break
        else:
            print("❌ 无效选择，请重新输入")

        input("\n按回车键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        print("请检查依赖包和文件是否完整")
