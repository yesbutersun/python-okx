# ==============================
# OKX真实模拟盘交易系统（均值回归策略）
# ==============================
import json
import logging
import time
from datetime import datetime, timedelta

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('okx_real_simulation_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入策略和API
from simple_strategy import mean_reversion_strategy
from okx.Trade import TradeAPI
from okx.Account import AccountAPI
from okx.MarketData import MarketAPI


class OKXRealSimulationTrader:
    """OKX真实模拟盘交易器"""

    def __init__(self, api_config_file='okx_simulation_config.json', trading_config_file='trading_config.json'):
        """初始化交易器"""
        self.load_config(api_config_file, trading_config_file)
        self.reset_trading_state()
        self.connect_okx()
        # 连接成功后立即从API获取真实账户状态
        self.initialize_account_state()

    def load_config(self, api_config_file, trading_config_file):
        """加载配置文件"""
        try:
            # 加载API配置
            with open(api_config_file, 'r') as f:
                config = json.load(f)

            self.api_key = config['api_key']
            self.secret_key = config['secret_key']
            self.passphrase = config['passphrase']

            # 加载交易配置
            with open(trading_config_file, 'r') as f:
                trading_config = json.load(f)

            self.symbol = trading_config.get('symbol', 'BTC-USDT-SWAP')
            self.position_size_usdt = trading_config.get('position_size_usdt', 100.0)
            self.leverage = trading_config.get('leverage', 5)
            self.strategy_params = trading_config.get('strategy_params', {})

            logger.info(f"✅ 配置加载成功: {self.symbol}")
            logger.info(f"🔐 API配置: API Key前4位 {self.api_key[:4]}...")
            logger.info(f"💰 仓位大小: {self.position_size_usdt} USDT")
            logger.info(f"📊 杠杆倍数: {self.leverage}x")

        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise

    def reset_trading_state(self):
        """重置交易状态"""
        self.initial_balance = 0
        self.current_balance = 0
        self.trades = []
        self.position = 0  # 当前持仓：0=无持仓，>0=多仓，<0=空仓
        self.entry_price = 0
        self.entry_time = None
        self.unrealized_pnl = 0
        self.equity_history = []

        # 策略参数
        self.lookback = self.strategy_params.get('lookback', 20)
        self.std_dev = self.strategy_params.get('std_dev', 2.0)

        # 风险管理
        self.max_drawdown = 0
        self.peak_equity = 0
        self.daily_loss_limit = 0.05  # 日亏损限制5%
        self.daily_start_balance = 0

        logger.info(f"🔄 交易状态重置完成")

    def initialize_account_state(self):
        """初始化账户状态，从OKX API获取真实数据"""
        try:
            logger.info("🔄 正在从OKX API获取账户初始状态...")

            # 1. 获取账户余额
            self.current_balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {self.current_balance:.2f} USDT")

            # 2. 设置初始余额（如果是第一次运行）
            if self.initial_balance == 0:
                self.initial_balance = self.current_balance
                logger.info(f"🎯 设置初始余额: {self.initial_balance:.2f} USDT")

            # 3. 获取当前持仓信息
            positions = self.get_positions()
            self.position = positions['position']
            self.entry_price = positions['entry_price']
            self.unrealized_pnl = positions['unrealized_pnl']

            # 4. 初始化风险管理参数
            self.peak_equity = self.current_balance + self.unrealized_pnl
            self.daily_start_balance = self.current_balance

            # 5. 记录初始权益
            total_equity = self.current_balance + self.unrealized_pnl
            self.equity_history.append(total_equity)

            logger.info(f"📊 初始化完成:")
            logger.info(f"   - 初始余额: {self.initial_balance:.2f} USDT")
            logger.info(f"   - 当前持仓: {self.position:.6f}")
            logger.info(f"   - 入场价格: {self.entry_price:.2f}")
            logger.info(f"   - 未实现盈亏: {self.unrealized_pnl:+.2f} USDT")
            logger.info(f"   - 总权益: {total_equity:.2f} USDT")

            return True

        except Exception as e:
            logger.error(f"❌ 账户状态初始化失败: {e}")
            # 使用默认值
            self.current_balance = 10000.0
            self.initial_balance = 10000.0
            self.peak_equity = 10000.0
            self.daily_start_balance = 10000.0
            logger.warning(f"⚠️ 使用默认值: {self.current_balance:.2f} USDT")
            return False

    def connect_okx(self):
        """连接OKX API"""
        try:
            # 使用沙盒标志
            flag = '1'  # 1 = 沙盒模式

            # 创建OKX API客户端
            self.trade_api = TradeAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                debug=True
            )

            self.account_api = AccountAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                debug=True
            )

            self.market_api = MarketAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=flag,
                debug=True
            )

            logger.info("✅ OKX沙盒API连接成功")

        except Exception as e:
            logger.error(f"❌ OKX连接失败: {e}")
            raise

    def get_account_balance(self):
        """获取账户余额"""
        try:
            # 使用OKX SDK获取账户余额
            result = self.account_api.get_account_balance()

            if result and result.get('code') == '0':
                details = result.get('data', [])
                for detail in details:
                    for asset in detail.get('details', []):
                        if asset.get('ccy') == 'USDT':
                            return float(asset.get('availEq', 0))
                        elif asset.get('ccy') == 'BTC' and float(asset.get('availEq', 0)) > 0:
                            # 如果有BTC余额，转换为USDT（粗略估计）
                            return float(asset.get('availEq', 0)) * self.get_current_btc_price()

            # 如果API失败，返回模拟余额
            logger.warning("⚠️ API余额获取失败，使用模拟余额: 10000 USDT")
            return 10000.0

        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            logger.warning("⚠️ 使用模拟余额: 10000 USDT")
            return 10000.0

    def get_current_btc_price(self):
        """获取当前BTC价格（从K线数据）"""
        try:
            # 使用OKX MarketAPI获取最新K线数据
            klines_data = self.market_api.get_candlesticks(instId='BTC-USDT-SWAP', bar='15m', limit='1')

            if klines_data and klines_data.get('code') == '0':
                klines = klines_data.get('data', [])
                if klines:
                    return float(klines[0][4])  # Close price
            return 0

        except Exception as e:
            logger.error(f"获取BTC价格失败: {e}")
            return 0

    def get_positions(self):
        """获取当前持仓"""
        try:
            # 使用OKX SDK获取持仓信息
            result = self.account_api.get_positions(instType='SWAP', instId=self.symbol)

            if result and result.get('code') == '0':
                positions = result.get('data', [])
                for pos in positions:
                    if pos.get('instId') == self.symbol and float(pos.get('pos', 0)) != 0:
                        return {
                            'position': float(pos['pos']),
                            'entry_price': float(pos['avgPx']) if pos['avgPx'] else 0,
                            'unrealized_pnl': float(pos['upl']) if pos['upl'] else 0,
                            'side': pos.get('posSide', '')
                        }

            return {'position': 0, 'entry_price': 0, 'unrealized_pnl': 0, 'side': ''}

        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return {'position': 0, 'entry_price': 0, 'unrealized_pnl': 0, 'side': ''}

    def place_order(self, side, size, order_type='market', price=None):
        """下单（真实模拟盘）"""
        try:
            logger.info(f"🔄 发送订单: {side} {size:.6f} {self.symbol}")

            # 准备订单参数
            order_params = {
                'instId': self.symbol,
                'tdMode': 'cross',
                'side': side,
                'ordType': order_type,
                'sz': str(size),
                'clOrdId': str(int(time.time() * 1000))
            }

            if order_type == 'limit' and price:
                order_params['px'] = str(price)

            # 使用OKX SDK下单
            result = self.trade_api.place_order(**order_params)

            logger.info(f"📊 订单响应: {result}")

            # 检查订单状态
            if result and result.get('code') == '0':
                order_id = result.get('data', [{}])[0].get('ordId') if result.get('data') else None
                logger.info(f"✅ 订单提交成功: {order_id}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'response': result
                }
            else:
                logger.error(f"❌ 订单提交失败: {result}")
                return {'success': False, 'response': result}

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'success': False, 'error': str(e)}

    def get_current_data(self):
        """获取最新市场数据"""
        try:
            # 使用OKX SDK获取最新100条K线数据用于策略计算
            result = self.market_api.get_candlesticks(instId=self.symbol, bar='15m', limit='100')

            if not result or result.get('code') != '0':
                raise Exception("无法获取K线数据")

            klines = result.get('data', [])

            if not klines:
                raise Exception("K线数据为空")

            # 转换为DataFrame - OKX API返回9个字段
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'quote_volume', 'trades_count', 'taker_buy_volume'
            ])

            # 转换数据类型
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])

            # 设置时间索引 - 修复FutureWarning
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            raise

    def calculate_position_size(self, current_price):
        """计算仓位大小"""
        try:
            # 使用固定USDT仓位，添加安全检查
            if current_price <= 0:
                logger.error(f"当前价格异常: {current_price}")
                return 0
            position_size = self.position_size_usdt / current_price
            return position_size
        except Exception as e:
            logger.error(f"仓位计算失败: {e}")
            return 0

    def generate_signals(self, df):
        """生成交易信号"""
        try:
            if len(df) < self.lookback:
                logger.warning(f"⚠️ 数据不足，需要至少{self.lookback}条，当前{len(df)}条")
                return None

            # 使用均值回归策略
            signals = mean_reversion_strategy(df, lookback=self.lookback, std_dev=self.std_dev)

            # 手动计算指标
            df['mean_price'] = df['Close'].rolling(self.lookback).mean()
            df['std_price'] = df['Close'].rolling(self.lookback).std()
            df['upper_band'] = df['mean_price'] + self.std_dev * df['std_price']
            df['lower_band'] = df['mean_price'] - self.std_dev * df['std_price']

            return signals

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return None

    def check_risk_limits(self):
        """检查风险限制"""
        try:
            if self.daily_start_balance == 0:
                self.daily_start_balance = self.current_balance
                return True

            daily_pnl_pct = (self.current_balance - self.daily_start_balance) / self.daily_start_balance

            if daily_pnl_pct < -self.daily_loss_limit:
                logger.warning(f"⚠️ 触发日亏损限制: {daily_pnl_pct:.2%}")
                return False

            return True

        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return False

    def run_trading_cycle(self):
        """执行一个交易周期"""
        try:
            logger.info("🔄 开始新的交易周期...")

            # 1. 获取市场数据
            df = self.get_current_data()
            if len(df) < self.lookback + 10:
                logger.warning(f"⚠️ 数据不足，需要至少{self.lookback + 10}条，当前{len(df)}条")
                return

            # 2. 更新账户状态（从API获取最新数据）
            positions = self.get_positions()
            self.current_balance = self.get_account_balance()
            self.position = positions['position']
            self.entry_price = positions['entry_price']
            self.unrealized_pnl = positions['unrealized_pnl']

            # 3. 计算总权益
            total_equity = self.current_balance + self.unrealized_pnl
            self.equity_history.append(total_equity)

            # 4. 更新最大回撤
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # 5. 风险检查
            if not self.check_risk_limits():
                logger.warning("⚠️ 触发风险限制，暂停交易")
                return

            # 6. 生成交易信号
            signals = self.generate_signals(df)
            if signals is None:
                logger.warning("⚠️ 信号生成失败，跳过此周期")
                return

            # 7. 检查最新信号
            latest_signal = signals.iloc[-1]
            latest_price = df['Close'].iloc[-1]
            mean_price = df['mean_price'].iloc[-1]

            logger.info(f"📊 当前价格: ${latest_price:,.2f}")
            logger.info(f"📈 均值线: ${mean_price:,.2f}")
            logger.info(f"💰 当前余额: ${self.current_balance:.2f}")
            logger.info(f"📈 当前持仓: {self.position:.6f}")
            logger.info(f"💹 未实现盈亏: ${self.unrealized_pnl:+.2f}")

            # 9. 执行交易逻辑
            if self.position == 0:  # 无持仓
                if latest_signal['long_entry']:
                    # 开多仓
                    position_size = self.calculate_position_size(latest_price)
                    result = self.place_order('buy', position_size, 'market')

                    if result.get('success'):
                        self.position = position_size
                        self.entry_price = latest_price
                        self.entry_time = datetime.now()

                        trade = {
                            'time': self.entry_time,
                            'action': 'BUY',
                            'price': latest_price,
                            'position': position_size,
                            'balance': self.current_balance,
                            'type': 'open_long'
                        }
                        self.trades.append(trade)

                        logger.info(f"📈 开多仓成功: ${latest_price:.2f}, 仓位: {position_size:.6f}")

                elif latest_signal['short_entry']:
                    # 开空仓
                    position_size = self.calculate_position_size(latest_price)
                    result = self.place_order('sell', position_size, 'market')

                    if result.get('success'):
                        self.position = -position_size
                        self.entry_price = latest_price
                        self.entry_time = datetime.now()

                        trade = {
                            'time': self.entry_time,
                            'action': 'SELL',
                            'price': latest_price,
                            'position': -position_size,
                            'balance': self.current_balance,
                            'type': 'open_short'
                        }
                        self.trades.append(trade)

                        logger.info(f"📉 开空仓成功: ${latest_price:.2f}, 仓位: {position_size:.6f}")

            elif self.position > 0:  # 持有多仓
                if latest_signal['long_exit'] or latest_price >= mean_price:
                    # 平多仓
                    result = self.place_order('sell', abs(self.position), 'market')

                    if result.get('success'):
                        execution_price = latest_price  # 模拟市价平仓
                        pnl = (execution_price - self.entry_price) * self.position
                        self.current_balance += pnl

                        trade = {
                            'time': datetime.now(),
                            'action': 'SELL',
                            'price': execution_price,
                            'position': -self.position,
                            'pnl': pnl,
                            'balance': self.current_balance,
                            'type': 'close_long'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 平多仓: ${execution_price:.2f}, 盈亏: ${pnl:+.2f} USDT")
                        self.position = 0
                        self.entry_price = 0

            elif self.position < 0:  # 持有空仓
                if latest_signal['short_exit'] or latest_price <= mean_price:
                    # 平空仓
                    result = self.place_order('buy', abs(self.position), 'market')

                    if result.get('success'):
                        execution_price = latest_price  # 模拟市价平仓
                        pnl = (self.entry_price - execution_price) * abs(self.position)
                        self.current_balance += pnl

                        trade = {
                            'time': datetime.now(),
                            'action': 'BUY',
                            'price': execution_price,
                            'position': abs(self.position),
                            'pnl': pnl,
                            'balance': self.current_balance,
                            'type': 'close_short'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 平空仓: ${execution_price:.2f}, 盈亏: ${pnl:+.2f} USDT")
                        self.position = 0
                        self.entry_price = 0

            logger.info(f"✅ 交易周期完成")

        except Exception as e:
            logger.error(f"❌ 交易周期执行失败: {e}")

    def print_status(self):
        """打印当前状态"""
        try:
            total_equity = self.current_balance + self.unrealized_pnl
            total_return = (total_equity - self.initial_balance) / self.initial_balance * 100

            print(f"\n{'='*60}")
            print(f"📊 OKX模拟盘交易状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"交易品种: {self.symbol}")
            print(f"初始资金: ${self.initial_balance:.2f} USDT")
            print(f"当前余额: ${self.current_balance:.2f} USDT")
            print(f"未实现盈亏: ${self.unrealized_pnl:+.2f} USDT")
            print(f"总权益: ${total_equity:.2f} USDT")
            print(f"总收益率: {total_return:+.2f}%")
            print(f"最大回撤: {self.max_drawdown:.2%}")
            print(f"当前持仓: {self.position:.6f}")
            print(f"交易次数: {len(self.trades)}")

            if self.position != 0:
                pnl_pct = (self.unrealized_pnl / (abs(self.position) * self.entry_price)) * 100
                position_type = "多仓" if self.position > 0 else "空仓"
                print(f"持仓类型: {position_type}")
                print(f"入场价格: ${self.entry_price:.2f}")
                print(f"持仓盈亏: {pnl_pct:+.2f}%")

            print(f"{'='*60}")

        except Exception as e:
            logger.error(f"状态打印失败: {e}")

    def save_results(self):
        """保存交易结果"""
        try:
            # 保存交易记录
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv('okx_simulation_trades.csv', index=False)
                logger.info("✅ 交易记录已保存到 okx_simulation_trades.csv")

            # 保存权益曲线
            if self.equity_history:
                equity_df = pd.DataFrame({
                    'equity': self.equity_history,
                    'timestamp': pd.date_range(start=datetime.now(), periods=len(self.equity_history), freq='15T')
                })
                equity_df.to_csv('okx_simulation_equity.csv', index=False)
                logger.info("✅ 权益曲线已保存到 okx_simulation_equity.csv")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def print_final_report(self):
        """打印最终报告"""
        try:
            total_equity = self.current_balance + self.unrealized_pnl
            total_return = (total_equity - self.initial_balance) / self.initial_balance * 100

            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                profitable_trades = trades_df[trades_df.get('pnl', 0) > 0]
                win_rate = len(profitable_trades) / len(trades_df) * 100

                avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
                losing_trades = trades_df[trades_df.get('pnl', 0) <= 0]
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            else:
                win_rate = 0
                avg_profit = 0
                avg_loss = 0

            print(f"\n🎯 OKX模拟盘交易完成")
            print(f"{'='*60}")
            print(f"最终资金: ${total_equity:.2f} USDT")
            print(f"总收益: {total_return:+.2f}%")
            print(f"最大回撤: {self.max_drawdown:.2%}")
            print(f"总交易次数: {len(self.trades)}")
            print(f"胜率: {win_rate:.1f}%")
            print(f"平均盈利: ${avg_profit:.2f} USDT")
            print(f"平均亏损: ${avg_loss:.2f} USDT")

            if avg_loss != 0:
                profit_loss_ratio = abs(avg_profit / avg_loss)
                print(f"盈亏比: {profit_loss_ratio:.2f}")

            print(f"{'='*60}")
            print(f"📁 详细结果文件:")
            print(f"   - okx_simulation_trades.csv")
            print(f"   - okx_simulation_equity.csv")
            print(f"   - okx_real_simulation_trading.log")

        except Exception as e:
            logger.error(f"最终报告生成失败: {e}")

    def run_continuous_trading(self, duration_minutes=60):
        """运行连续交易"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        logger.info(f"🚀 开始OKX模拟盘交易，持续 {duration_minutes} 分钟")
        logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            cycle_count = 0
            while datetime.now() < end_time:
                cycle_start = time.time()

                # 执行交易周期
                self.run_trading_cycle()
                cycle_count += 1

                # 每5分钟执行一次（15分钟K线，5分钟检查一次）
                cycle_time = time.time() - cycle_start
                wait_time = max(0, 300 - cycle_time)  # 5分钟间隔

                if wait_time > 0:
                    logger.info(f"⏳ 等待 {wait_time:.0f} 秒后进行下一周期 (已完成 {cycle_count} 个周期)...")
                    time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("⏹️ 用户中断交易")
        except Exception as e:
            logger.error(f"❌ 连续交易失败: {e}")
        finally:
            # 保存结果
            self.save_results()
            self.print_final_report()


def main(sandbox=True, trading_duration=30):
    """主函数"""
    try:
        trader = OKXRealSimulationTrader()

        # 设置交易时长（分钟）
        trading_duration = trading_duration  # 默认30分钟

        logger.info(f"🚀 开始OKX{'模拟盘' if sandbox else '实盘'}交易，持续 {trading_duration} 分钟")

        # 开始连续交易
        trader.run_continuous_trading(duration_minutes=trading_duration)

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return False

    return True


if __name__ == "__main__":
    import sys

    try:
        # 支持命令行参数
        sandbox = True  # 默认使用沙盒
        trading_duration = 600  # 默认60分钟

        if len(sys.argv) > 1:
            if sys.argv[1].lower() == '--production':
                sandbox = False
                print("⚠️ 使用实盘环境 - 请谨慎操作！")
            elif sys.argv[1].lower() == '--sandbox':
                sandbox = True
                print("🏖️ 使用沙盒环境")

        if len(sys.argv) > 2:
            try:
                trading_duration = int(sys.argv[2])
                print(f"📊 设置交易时长: {trading_duration} 分钟")
            except ValueError:
                print("⚠️ 无效的交易时长，使用默认值 60 分钟")

        success = main(sandbox=sandbox, trading_duration=trading_duration)
        exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⏹️ 程序被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
