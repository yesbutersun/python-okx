#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC实盘交易系统
集成strategy.py中的交易策略，通过OKX API执行实盘交易
并记录交易明细和生成总结报告
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict

from okx import Trade, MarketData, Account
from strategy import (
    STRATEGIES, run_strategy, get_strategy_list,
    trend_atr_signal, boll_rsi_signal, rsi_reversal_strategy,
    trend_volatility_stop_signal, breakout_strategy,
    mean_reversion_strategy, boll_zscore_slope_accel_strategy, momentum_strategy, macd_strategy
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录数据类"""
    timestamp: str
    trade_id: str
    symbol: str
    side: str  # 'buy'/'sell'
    order_type: str
    size: float
    price: float
    amount: float
    strategy: str
    signal_type: str  # 'long_entry'/'long_exit'/'short_entry'/'short_exit'
    fee: float = 0.0
    pnl: float = 0.0
    balance: float = 0.0
    notes: str = ""


@dataclass
class PositionInfo:
    """持仓信息数据类"""
    symbol: str
    side: str  # 'long'/'short'/'flat'
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    entry_time: str


class BTCLiveTrader:
    """BTC实盘交易器"""

    def __init__(self, config_file: str = "trading_config.json"):
        """
        初始化交易器

        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.symbol = self.config.get("symbol", "BTC-USDT-SWAP")
        self.strategy_name = self.config.get("strategy", "trend_atr")
        self.trade_mode = self.config.get("trade_mode", "cross")  # cross/isolated
        self.position_size = self.config.get("position_size", 0.01)  # BTC数量
        self.max_positions = self.config.get("max_positions", 1)
        self.leverage = self.config.get("leverage", 10)

        # 初始化API客户端
        self._init_api_clients()

        # 交易状态
        self.current_position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0.0
        self.entry_time = None

        # 交易记录
        self.trades: List[TradeRecord] = []
        self.positions: List[PositionInfo] = []

        # 数据存储
        self.trade_records_file = f"trade_records_{datetime.now().strftime('%Y%m%d')}.json"
        self.balance_file = f"balance_history_{datetime.now().strftime('%Y%m%d')}.csv"

        logger.info(f"BTC交易器初始化完成，交易对: {self.symbol}, 策略: {self.strategy_name}")

    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        default_config = {
            "api_key": "",
            "secret_key": "",
            "passphrase": "",
            "symbol": "BTC-USDT-SWAP",
            "strategy": "trend_atr",
            "trade_mode": "cross",
            "position_size": 0.01,
            "max_positions": 1,
            "leverage": 10,
            "timeframe": "5m",
            "data_limit": 100,
            "risk_management": {
                "max_loss_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 2.0
            }
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        else:
            # 创建默认配置文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            logger.info(f"已创建默认配置文件: {config_file}")

        return default_config

    def _init_api_clients(self):
        """初始化API客户端"""
        try:
            api_key = self.config["api_key"]
            secret_key = self.config["secret_key"]
            passphrase = self.config["passphrase"]
            domain = self.config.get("domain", "https://www.okx.com")
            flag = self.config.get("flag", "1")  # 1: 生产环境, 0: 沙盒环境

            if not all([api_key, secret_key, passphrase]):
                raise ValueError("API密钥配置不完整")

            # 根据环境配置API客户端
            self.trade_api = Trade.TradeAPI(
                api_key, secret_key, passphrase,
                debug=False, domain=domain
            )
            self.market_api = MarketData.MarketAPI(
                api_key, secret_key, passphrase,
                debug=False, domain=domain
            )
            self.account_api = Account.AccountAPI(
                api_key, secret_key, passphrase,
                debug=False, domain=domain
            )

            # 设置环境标记
            self.is_sandbox = (flag == "0")
            self.environment = "沙盒环境" if self.is_sandbox else "生产环境"

            logger.info(f"API客户端初始化成功 - {self.environment}")
            logger.info(f"API域名: {domain}")

            # 设置杠杆
            self._set_leverage()

        except Exception as e:
            logger.error(f"API客户端初始化失败: {e}")
            raise

    def _set_leverage(self):
        """设置杠杆倍数"""
        try:
            result = self.account_api.set_leverage(
                instId=self.symbol,
                lever=str(self.leverage),
                mgnMode=self.trade_mode
            )
            if result.get("code") == "0":
                logger.info(f"杠杆设置成功: {self.leverage}x")
            else:
                logger.warning(f"杠杆设置失败: {result}")
        except Exception as e:
            logger.warning(f"设置杠杆时出错: {e}")

    def get_market_data(self, limit: int = None) -> pd.DataFrame:
        """获取K线数据"""
        try:
            limit = limit or self.config.get("data_limit", 100)
            timeframe = self.config.get("timeframe", "5m")

            result = self.market_api.get_candlesticks(
                instId=self.symbol,
                bar=timeframe,
                limit=str(limit)
            )

            if result.get("code") == "0" and result.get("data"):
                data = result["data"]
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'volume_currency', 'volume_currency_quote', 'confirm'
                ])

                # 转换数据类型
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)

                # 重命名列以匹配策略要求
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close',
                    'volume': 'Volume'
                })

                logger.info(f"获取到 {len(df)} 条K线数据")
                return df
            else:
                logger.error(f"获取市场数据失败: {result}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取市场数据时出错: {e}")
            return pd.DataFrame()

    def get_strategy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据配置的策略获取交易信号"""
        try:
            # 策略名称映射
            strategy_mapping = {
                "trend_atr": "趋势ATR策略",
                "boll_rsi": "布林RSI策略",
                "rsi_reversal": "RSI反转策略",
                "trend_volatility_stop": "趋势波动止损策略",
                "breakout": "突破策略",
                "mean_reversion": "均值回归策略",
                "boll_zscore_slope_accel": "BOLL+Z-score斜率加速度策略",
                "momentum": "动量策略",
                "macd": "MACD策略"
            }

            # 如果策略名称直接在STRATEGIES中，使用run_strategy函数
            if self.strategy_name in strategy_mapping:
                strategy_cn_name = strategy_mapping[self.strategy_name]
                if strategy_cn_name in STRATEGIES:
                    signals = run_strategy(df, strategy_cn_name)
                    logger.info(f"策略信号生成完成: {strategy_cn_name}")
                    return signals

            # 尝试直接使用策略函数
            if self.strategy_name == "trend_atr":
                signals = trend_atr_signal(df)
            elif self.strategy_name == "boll_rsi":
                signals = boll_rsi_signal(df)
            elif self.strategy_name == "rsi_reversal":
                signals = rsi_reversal_strategy(df)
            elif self.strategy_name == "trend_volatility_stop":
                signals = trend_volatility_stop_signal(df)
            elif self.strategy_name == "breakout":
                signals = breakout_strategy(df)
            elif self.strategy_name == "mean_reversion":
                signals = mean_reversion_strategy(df)
            elif self.strategy_name == "boll_zscore_slope_accel":
                signals = boll_zscore_slope_accel_strategy(df)
            elif self.strategy_name == "momentum":
                signals = momentum_strategy(df)
            elif self.strategy_name == "macd":
                signals = macd_strategy(df)
            else:
                # 显示可用策略
                available_strategies = list(strategy_mapping.keys())
                logger.error(f"未知策略: {self.strategy_name}")
                logger.error(f"可用策略: {available_strategies}")
                return pd.DataFrame()

            logger.info(f"策略信号生成完成: {self.strategy_name}")
            return signals

        except Exception as e:
            logger.error(f"生成策略信号时出错: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """获取当前价格"""
        try:
            result = self.market_api.get_ticker(self.symbol)
            if result.get("code") == "0" and result.get("data"):
                return float(result["data"][0]["last"])
            else:
                logger.error(f"获取当前价格失败: {result}")
                return 0.0
        except Exception as e:
            logger.error(f"获取当前价格时出错: {e}")
            return 0.0

    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        try:
            result = self.account_api.get_balance()
            if result.get("code") == "0" and result.get("data"):
                for currency_info in result["data"][0]["details"]:
                    if currency_info["ccy"] == "USDT":
                        return {
                            "usdt_balance": float(currency_info["eq"]),
                            "available_balance": float(currency_info["availBal"])
                        }
            return {"usdt_balance": 0.0, "available_balance": 0.0}
        except Exception as e:
            logger.error(f"获取账户余额时出错: {e}")
            return {"usdt_balance": 0.0, "available_balance": 0.0}

    def place_order(self, side: str, order_type: str = "market", price: float = None,
                   size: float = None, signal_type: str = "") -> Dict:
        """下单"""
        try:
            size = size or self.position_size
            size_str = str(size)

            # 订单参数
            params = {
                "instId": self.symbol,
                "tdMode": self.trade_mode,
                "side": side,
                "ordType": order_type,
                "sz": size_str
            }

            # 限价单需要价格
            if order_type == "limit" and price:
                params["px"] = str(price)

            # 仓位方向
            if self.symbol.endswith("-SWAP"):
                if self.current_position == 1:
                    params["posSide"] = "long"
                elif self.current_position == -1:
                    params["posSide"] = "short"
                else:
                    params["posSide"] = "long" if side == "buy" else "short"

            logger.info(f"下单参数: {params}")

            result = self.trade_api.place_order(**params)

            if result.get("code") == "0":
                order_info = result["data"][0]
                logger.info(f"下单成功: {order_info}")

                # 记录交易
                self._record_trade(order_info, side, order_type, size, price, signal_type)

                return {"success": True, "order_info": order_info}
            else:
                logger.error(f"下单失败: {result}")
                return {"success": False, "error": result}

        except Exception as e:
            logger.error(f"下单时出错: {e}")
            return {"success": False, "error": str(e)}

    def _record_trade(self, order_info: Dict, side: str, order_type: str,
                     size: float, price: float, signal_type: str):
        """记录交易"""
        try:
            current_price = price or self.get_current_price()
            amount = size * current_price

            trade = TradeRecord(
                timestamp=datetime.now().isoformat(),
                trade_id=order_info.get("ordId", ""),
                symbol=self.symbol,
                side=side,
                order_type=order_type,
                size=size,
                price=current_price,
                amount=amount,
                strategy=self.strategy_name,
                signal_type=signal_type,
                balance=self.get_account_balance()["available_balance"]
            )

            self.trades.append(trade)

            # 保存到文件
            self._save_trade_records()

            logger.info(f"交易记录已保存: {side} {size} BTC @ {current_price}")

        except Exception as e:
            logger.error(f"记录交易时出错: {e}")

    def _save_trade_records(self):
        """保存交易记录到文件"""
        try:
            records = [asdict(trade) for trade in self.trades]
            with open(self.trade_records_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存交易记录时出错: {e}")

    def execute_signal(self, signal_type: str, current_price: float):
        """执行交易信号"""
        try:
            # 检查信号类型和当前持仓
            if signal_type == "long_entry" and self.current_position != 1:
                logger.info(f"执行做多信号，当前价格: {current_price}")
                result = self.place_order("buy", "market", signal_type="long_entry")
                if result["success"]:
                    self.current_position = 1
                    self.entry_price = current_price
                    self.entry_time = datetime.now()

            elif signal_type == "short_entry" and self.current_position != -1:
                logger.info(f"执行做空信号，当前价格: {current_price}")
                result = self.place_order("sell", "market", signal_type="short_entry")
                if result["success"]:
                    self.current_position = -1
                    self.entry_price = current_price
                    self.entry_time = datetime.now()

            elif signal_type == "long_exit" and self.current_position == 1:
                logger.info(f"执行平多仓信号，当前价格: {current_price}")
                result = self.place_order("sell", "market", signal_type="long_exit")
                if result["success"]:
                    self.current_position = 0
                    self._calculate_and_record_pnl(current_price)

            elif signal_type == "short_exit" and self.current_position == -1:
                logger.info(f"执行平空仓信号，当前价格: {current_price}")
                result = self.place_order("buy", "market", signal_type="short_exit")
                if result["success"]:
                    self.current_position = 0
                    self._calculate_and_record_pnl(current_price)

        except Exception as e:
            logger.error(f"执行交易信号时出错: {e}")

    def _calculate_and_record_pnl(self, exit_price: float):
        """计算并记录盈亏"""
        try:
            if self.entry_price > 0 and self.entry_time:
                if self.current_position == 0:  # 平仓
                    # 更新最后一笔交易的PNL
                    if self.trades:
                        last_trade = self.trades[-1]
                        if self.entry_price > 0:
                            if last_trade.side == "buy":  # 平多仓
                                pnl = (exit_price - self.entry_price) * self.position_size
                            else:  # 平空仓
                                pnl = (self.entry_price - exit_price) * self.position_size

                            last_trade.pnl = pnl
                            self._save_trade_records()

                            logger.info(f"平仓盈亏: {pnl:.2f} USDT")

        except Exception as e:
            logger.error(f"计算盈亏时出错: {e}")

    def run_trading_cycle(self):
        """运行一次交易周期"""
        try:
            logger.info("=== 开始新的交易周期 ===")

            # 获取市场数据
            df = self.get_market_data()
            if df.empty:
                logger.error("无法获取市场数据，跳过本次周期")
                return

            # 生成交易信号
            signals = self.get_strategy_signals(df)
            if signals.empty:
                logger.error("无法生成交易信号，跳过本次周期")
                return

            # 获取最新信号
            latest_signals = signals.iloc[-1]
            current_price = self.get_current_price()

            logger.info(f"当前价格: {current_price}, 信号: {latest_signals.to_dict()}")

            # 检查并执行信号
            for signal_type, is_signal in latest_signals.items():
                if is_signal and signal_type in ['long_entry', 'long_exit', 'short_entry', 'short_exit']:
                    self.execute_signal(signal_type, current_price)
                    time.sleep(1)  # 避免频繁交易

            # 记录余额
            self._record_balance()

            logger.info("=== 交易周期完成 ===")

        except Exception as e:
            logger.error(f"交易周期执行时出错: {e}")

    def _record_balance(self):
        """记录账户余额"""
        try:
            balance_info = self.get_account_balance()
            balance_record = {
                "timestamp": datetime.now().isoformat(),
                "usdt_balance": balance_info["usdt_balance"],
                "available_balance": balance_info["available_balance"],
                "position": self.current_position,
                "entry_price": self.entry_price,
                "current_price": self.get_current_price()
            }

            # 保存到CSV文件
            df_balance = pd.DataFrame([balance_record])
            if os.path.exists(self.balance_file):
                df_existing = pd.read_csv(self.balance_file)
                df_balance = pd.concat([df_existing, df_balance], ignore_index=True)

            df_balance.to_csv(self.balance_file, index=False)

        except Exception as e:
            logger.error(f"记录余额时出错: {e}")

    def generate_daily_report(self) -> Dict:
        """生成每日交易报告"""
        try:
            if not self.trades:
                return {"error": "暂无交易记录"}

            # 当日交易
            today = datetime.now().date()
            today_trades = [t for t in self.trades if datetime.fromisoformat(t.timestamp).date() == today]

            if not today_trades:
                return {"error": "今日暂无交易"}

            # 统计数据
            total_trades = len(today_trades)
            total_pnl = sum(t.pnl for t in today_trades)
            winning_trades = [t for t in today_trades if t.pnl > 0]
            losing_trades = [t for t in today_trades if t.pnl < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

            # 当前持仓
            current_balance = self.get_account_balance()
            current_price = self.get_current_price()

            report = {
                "date": today.isoformat(),
                "summary": {
                    "total_trades": total_trades,
                    "total_pnl": round(total_pnl, 2),
                    "win_rate": round(win_rate * 100, 2),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                },
                "current_status": {
                    "position": self.current_position,
                    "entry_price": self.entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": self._calculate_unrealized_pnl(current_price),
                    "balance": current_balance
                },
                "trades": [asdict(t) for t in today_trades]
            }

            # 保存报告
            report_file = f"daily_report_{today.strftime('%Y%m%d')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"每日报告已生成: {report_file}")
            return report

        except Exception as e:
            logger.error(f"生成每日报告时出错: {e}")
            return {"error": str(e)}

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        try:
            if self.current_position == 0 or self.entry_price == 0:
                return 0.0

            if self.current_position == 1:  # 多仓
                return (current_price - self.entry_price) * self.position_size
            else:  # 空仓
                return (self.entry_price - current_price) * self.position_size

        except Exception as e:
            logger.error(f"计算未实现盈亏时出错: {e}")
            return 0.0

    def start_continuous_trading(self, interval_seconds: int = 300):
        """开始连续交易"""
        logger.info(f"开始连续交易，检查间隔: {interval_seconds}秒")
        logger.info(f"交易对: {self.symbol}, 策略: {self.strategy_name}")
        logger.info(f"仓位大小: {self.position_size} BTC, 杠杆: {self.leverage}x")

        try:
            while True:
                self.run_trading_cycle()
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("交易被用户中断")
            self.generate_daily_report()
        except Exception as e:
            logger.error(f"连续交易时出错: {e}")
            self.generate_daily_report()


def main():
    """主函数"""
    trader = BTCLiveTrader()

    print("BTC实盘交易系统")
    print("=" * 50)
    print(f"交易环境: {trader.environment}")
    print(f"API域名: {trader.config.get('domain', 'https://www.okx.com')}")
    print(f"交易对: {trader.symbol}")
    print(f"策略: {trader.strategy_name}")
    print(f"仓位大小: {trader.position_size} BTC")
    print(f"杠杆: {trader.leverage}x")

    # 环境警告
    if trader.is_sandbox:
        print("🔒 沙盒环境: 使用测试资金，无真实风险")
    else:
        print("⚠️  生产环境: 使用真实资金，存在资金风险")
        print("   建议先在沙盒环境测试!")

    print("=" * 50)

    try:
        # 运行一次测试
        print("运行一次交易测试...")
        trader.run_trading_cycle()

        # 生成报告
        print("\n生成每日报告...")
        report = trader.generate_daily_report()

        if "error" not in report:
            print("\n=== 每日交易报告 ===")
            print(f"交易次数: {report['summary']['total_trades']}")
            print(f"总盈亏: {report['summary']['total_pnl']} USDT")
            print(f"胜率: {report['summary']['win_rate']}%")
            print(f"当前仓位: {report['current_status']['position']}")

        # 询问是否开始连续交易
        response = input("\n是否开始连续交易? (y/n): ").lower()
        if response == 'y':
            interval = int(input("请输入检查间隔(秒，默认300): ") or 300)
            trader.start_continuous_trading(interval)

    except Exception as e:
        logger.error(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()
