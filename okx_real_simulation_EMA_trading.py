# ==============================
# OKX真实模拟盘交易系统（均值回归策略）
# ==============================
import json
import os
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('okx_real_simulation_EMA_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入策略和API
from simple_strategy import ema_mean_reversion_strategy
from okx.Trade import TradeAPI
from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx_contract_specs import get_contract_spec, validate_order_size
from stop_loss import LossPriceDiffStopLoss
try:
    from trading_visualizer import TradingVisualizer
except Exception:
    TradingVisualizer = None


class OKXRealSimulationTrader:
    """OKX真实模拟盘交易器"""

    def __init__(self, secrets_file='secrets.json', trading_config_file='trading_config.json'):
        """初始化交易器"""
        self.load_config(secrets_file, trading_config_file)
        self.reset_trading_state()
        self.connect_okx()
        self.last_signal_ts = None  # 记录上一次处理的信号时间，避免漏单

        # 确保所有必要属性都被正确初始化
        # 风险管理属性
        self.daily_loss_limit = None  # 日亏损限制5% (None=disable)
        self.daily_start_balance = 0
        self.max_drawdown = 0
        self.peak_equity = 0

        # 交易状态属性
        self.initial_balance = 0
        self.current_balance = 0
        self.trades = []
        self.position = 0  # 当前持仓：0=无持仓，>0=多仓，<0=空仓
        self.entry_price = 0
        self.entry_time = None
        self.unrealized_pnl = 0
        self.equity_history = []
        self.equity_timestamps = []

        # 策略参数
        self.lookback = 20  # 默认值
        self.std_dev = 2.0  # 默认值

        # 杠杆设置
        self.leverage = 5  # 默认杠杆

        # 可视化
        self.visualizer_enabled = TradingVisualizer is not None
        self.visualizer = TradingVisualizer(figsize=(18, 10), dpi=110) if self.visualizer_enabled else None
        self.last_price_df = None  # 保存最近一批价格数据便于画图

        # 连接成功后立即从API获取真实账户状态
        self.initialize_account_state()

    def load_config(self, secrets_file, trading_config_file):
        """加载配置文件"""
        try:
            secrets_file = self._resolve_config_path(secrets_file)
            trading_config_file = self._resolve_config_path(trading_config_file)

            # 加载交易配置
            with open(trading_config_file, 'r', encoding='utf-8') as f:
                trading_config = json.load(f)

            # 加载密钥配置（支持按环境分组）
            with open(secrets_file, 'r', encoding='utf-8') as f:
                secrets = json.load(f)

            environment = trading_config.get('environment', 'sandbox')
            env_block = secrets.get('environments', {}).get(environment, {})
            self.api_key = env_block.get('api_key') or secrets.get('api_key')
            self.secret_key = env_block.get('secret_key') or secrets.get('secret_key')
            self.passphrase = env_block.get('passphrase') or secrets.get('passphrase')
            self.okx_flag = str(
                env_block.get('flag')
                or secrets.get('flag')
                or trading_config.get('flag', '1')
            )

            if not self.api_key or not self.secret_key or not self.passphrase:
                raise ValueError(f"{secrets_file} 缺少必要的API密钥 (environment={environment})")

            symbol, instrument_config = self._select_instrument_config(trading_config)
            self.symbol = symbol
            self.position_size_usdt = instrument_config.get(
                'position_size_usdt',
                trading_config.get('position_size_usdt', 100.0)
            )
            self.leverage = instrument_config.get('leverage', trading_config.get('leverage', 5))
            self.strategy_params = instrument_config.get('strategy_params', trading_config.get('strategy_params', {}))

            self.stop_loss_threshold = instrument_config.get(
                'stop_loss_threshold',
                trading_config.get('stop_loss_threshold', 50.0)
            )
            self.stop_loss_policy = LossPriceDiffStopLoss(self.stop_loss_threshold)

            logger.info(f"✅ 配置加载成功: {self.symbol}")
            logger.info(f"🔐 API配置: API Key前4位 {self.api_key[:4]}...")
            logger.info(f"🧪 OKX环境: flag={self.okx_flag} (1=沙盒, 0=实盘)")
            logger.info(f"💰 仓位大小: {self.position_size_usdt} USDT")
            logger.info(f"📊 杠杆倍数: {self.leverage}x")

        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise

    @staticmethod
    def _resolve_config_path(config_path):
        if os.path.isabs(config_path):
            return config_path
        return os.path.join(os.path.dirname(__file__), config_path)

    @staticmethod
    def _select_instrument_config(trading_config):
        instruments = trading_config.get('instruments', {})
        script_symbols = trading_config.get('script_symbols', {})
        script_symbol = script_symbols.get(os.path.basename(__file__))
        symbol = script_symbol or trading_config.get('symbol') or trading_config.get('default_symbol')
        if instruments:
            if not symbol:
                symbol = next(iter(instruments.keys()))
            base = trading_config.get('defaults', {})
            instrument_config = instruments.get(symbol, {})
            return symbol, {**base, **instrument_config}
        return symbol or 'BTC-USDT-SWAP', trading_config

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
        self.equity_timestamps = []

        # 策略参数
        self.lookback = self.strategy_params.get('lookback', 20)
        self.std_dev = self.strategy_params.get('std_dev', 2.0)

        self.stop_loss_threshold = getattr(self, 'stop_loss_threshold', 50.0)
        self.stop_loss_policy = LossPriceDiffStopLoss(self.stop_loss_threshold)

        # 风险管理
        self.max_drawdown = 0
        self.peak_equity = 0
        self.daily_loss_limit = None  # 日亏损限制5% (None=disable)
        self.daily_start_balance = 0
        self.leverage = 5  # 交易杠杆倍数

        logger.info(f"🔄 交易状态重置完成")

    def initialize_account_state(self):
        """初始化账户状态，从OKX API获取真实数据"""
        try:
            logger.info("🔄 正在从OKX API获取账户初始状态...")

            # 1. 设置交易杠杆（在获取账户信息前）
            logger.info(f"🔧 准备设置交易杠杆: {self.leverage}x")
            try:
                result = self.account_api.set_leverage(instId=self.symbol, lever=str(self.leverage), mgnMode='cross')

                if result and result.get('code') == '0':
                    logger.info(f"✅ 杠杆设置完成: {self.leverage}x")
                else:
                    logger.error(f"❌ 杠杆设置失败: {result}")
                    logger.warning(f"💡 建议检查杠杆倍数限制和账户权限")
            except Exception as e:
                logger.error(f"杠杆设置失败: {e}")
                logger.warning(f"⚠️ 将使用默认杠杆进行交易")

            # 2. 获取账户余额
            self.current_balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {self.current_balance:.2f} USDT")

            # 3. 设置初始余额（如果是第一次运行）
            if self.initial_balance == 0:
                self.initial_balance = self.current_balance
                logger.info(f"🎯 设置初始余额: {self.initial_balance:.2f} USDT")

            # 4. 获取当前持仓信息
            positions = self.get_positions()
            self.position = positions['position']
            self.entry_price = positions['entry_price']
            self.unrealized_pnl = positions['unrealized_pnl']

            # 5. 初始化风险管理参数
            self.peak_equity = self.current_balance + self.unrealized_pnl
            self.daily_start_balance = self.current_balance

            # 6. 记录初始权益
            total_equity = self.current_balance + self.unrealized_pnl
            self._record_equity(total_equity)

            logger.info(f"📊 初始化完成:")
            logger.info(f"   - 初始余额: {self.initial_balance:.2f} USDT")
            logger.info(f"   - 当前持仓: {self.position:.6f}")
            logger.info(f"   - 入场价格: {self.entry_price:.2f}")
            logger.info(f"   - 未实现盈亏: {self.unrealized_pnl:+.2f} USDT")
            logger.info(f"   - 总权益: {total_equity:.2f} USDT")
            logger.info(f"   - 交易杠杆: {self.leverage}x")

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
            # 创建OKX API客户端
            self.trade_api = TradeAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.okx_flag,
                debug=True
            )

            self.account_api = AccountAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.okx_flag,
                debug=True
            )

            self.market_api = MarketAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.okx_flag,
                debug=True
            )

            logger.info("✅ OKX沙盒API连接成功")

            # 杠杆设置将在initialize_account_state方法中处理，确保API已正确初始化

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
        """下单（真实模拟盘，适配OKX合约规格）"""
        try:
            logger.info(f"🔄 准备订单: {side} {size:.6f} {self.symbol}")

            # 获取合约规格，按品种动态调整
            spec = get_contract_spec(self.symbol)
            if not spec:
                logger.warning(f"未找到 {self.symbol} 的合约规格，使用默认BTC规格")
                spec = {
                    'min_lot_size': 0.001,
                    'lot_size_multiple': 0.001
                }
            min_lot_size = Decimal(str(spec['min_lot_size']))
            lot_size_multiple = Decimal(str(spec['lot_size_multiple']))

            # 使用统一的规格验证逻辑
            try:
                adjusted_size, validation_logs = validate_order_size(
                    self.symbol,
                    size,
                    price if price else 0,
                    self.position_size_usdt
                )
                for log in validation_logs:
                    logger.info(f"📏 {log}")
            except Exception as e:
                logger.warning(f"规格验证失败，使用默认规格调整: {e}")
                adjusted_size = size

            # 确保数量严格对齐到最小单位和倍数，避免浮点误差
            size_decimal = Decimal(str(adjusted_size))
            multiples = (size_decimal / lot_size_multiple).to_integral_value(rounding=ROUND_DOWN)
            aligned_size = multiples * lot_size_multiple
            if aligned_size < min_lot_size:
                aligned_size = min_lot_size

            if aligned_size != size_decimal:
                logger.info(f"📏 数量重新对齐: {size_decimal:.6f} → {aligned_size:.6f} (步长 {lot_size_multiple})")

            size_str = format(aligned_size.normalize(), 'f')
            size = float(aligned_size)

            # 准备订单参数
            order_params = {
                'instId': self.symbol,
                'tdMode': 'cross',  # 全仓模式
                'side': side,
                'ordType': order_type,
                'sz': size_str,
                'clOrdId': str(int(time.time() * 1000))
            }

            # 如果是限价单，添加价格
            if order_type == 'limit' and price:
                order_params['px'] = str(price)

            # OKX永续合约使用'lever'参数设置杠杆，在订单前需要单独设置
            # 注意：杠杆不是通过订单参数设置，而是通过单独的API设置
            # 这里先注释掉，避免参数错误
            # order_params['lever'] = str(self.leverage) if hasattr(self, 'leverage') else '5'

            logger.info(f"📋 订单参数: {order_params}")

            # 使用OKX SDK下单
            result = self.trade_api.place_order(**order_params)

            logger.info(f"📊 订单响应: {result}")

            # 检查订单状态
            if result and result.get('code') == '0':
                order_data = result.get('data', [{}])
                if order_data:
                    order_id = order_data[0].get('ordId')
                    client_order_id = order_data[0].get('clOrdId')
                    logger.info(f"✅ 订单提交成功:")
                    logger.info(f"   - 订单ID: {order_id}")
                    logger.info(f"   - 客户端ID: {client_order_id}")
                    logger.info(f"   - 交易方向: {side}")
                    logger.info(f"   - 交易数量: {size}")
                    logger.info(f"   - 合约代码: {self.symbol}")

                    return {
                        'success': True,
                        'order_id': order_id,
                        'client_order_id': client_order_id,
                        'size': size,
                        'side': side,
                        'response': result
                    }
                else:
                    logger.error(f"❌ 订单响应数据为空: {result}")
                    return {'success': False, 'response': result}
            else:
                error_msg = result.get('msg', 'Unknown error')
                error_data = result.get('data', [])
                if error_data and len(error_data) > 0:
                    error_code = error_data[0].get('sCode', 'Unknown')
                    error_detail = error_data[0].get('sMsg', 'No detail')
                    logger.error(f"❌ 订单提交失败:")
                    logger.error(f"   - 错误代码: {error_code}")
                    logger.error(f"   - 错误详情: {error_detail}")
                    logger.error(f"   - 完整响应: {result}")

                    # 如果是lot size错误，提供具体建议
                    if 'lot size' in error_detail.lower() or 'multiple' in error_detail.lower():
                        logger.error(f"💡 建议检查:")
                        logger.error(f"   - 当前数量: {size_str}")
                        logger.error(f"   - 最小单位: {min_lot_size}")
                        suggestion_size = aligned_size + lot_size_multiple
                        logger.error(f"   - 建议数量: {format(suggestion_size.normalize(), 'f')}")

                return {'success': False, 'response': result}

        except Exception as e:
            logger.error(f"下单失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def get_current_data(self):
        """获取最新市场数据"""
        try:
            # 使用OKX SDK获取最新500条K线数据用于策略计算
            result = self.market_api.get_candlesticks(instId=self.symbol, bar='15m', limit='500')

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

            # 保存实时数据到CSV，便于后续分析/排查
            self._save_live_data(df)

            return df

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            raise

    def calculate_position_size(self, current_price):
        """计算仓位大小（使用合约规格验证）"""
        try:
            # 使用固定USDT仓位，添加安全检查
            if current_price <= 0:
                logger.error(f"当前价格异常: {current_price}")
                return 0

            spec = get_contract_spec(self.symbol)
            if not spec:
                logger.warning(f"未找到 {self.symbol} 的合约规格，使用默认张数1")
                return 1

            contract_value = Decimal(str(spec.get('contract_value', 1)))
            notional_per_contract = contract_value * Decimal(str(current_price))

            # 将目标USDT换算为张数（OKX永续下单数量单位为张数）
            base_contracts = Decimal(str(self.position_size_usdt)) / (contract_value * Decimal(str(current_price)))
            logger.info(
                f"🎯 目标仓位: {self.position_size_usdt} USDT → 预计下单 {base_contracts:.4f} 张 "
                f"(每张名义约 {float(notional_per_contract):.2f} USDT)"
            )

            # 使用合约规格验证和调整
            try:
                adjusted_size, validation_logs = validate_order_size(
                    self.symbol, float(base_contracts), current_price, self.position_size_usdt
                )

                # 输出验证日志
                for log in validation_logs:
                    logger.info(f"📏 {log}")

                return adjusted_size

            except Exception as e:
                logger.error(f"订单规格验证失败: {e}")
                # 使用fallback逻辑
                spec = get_contract_spec(self.symbol)
                if spec:
                    min_size = spec['min_lot_size']
                    logger.warning(f"⚠️ 使用fallback最小单位: {min_size}")
                    return min_size
                else:
                    logger.error(f"❌ 无法获取合约规格: {self.symbol}")
                    return 1  # 返回1张而不是0

        except Exception as e:
            logger.error(f"仓位计算失败: {e}")
            return 1  # 返回最小张数而不是0

    def generate_signals(self, df):
        """生成交易信号"""
        try:
            if len(df) < self.lookback:
                logger.warning(f"⚠️ 数据不足，需要至少{self.lookback}条，当前{len(df)}条")
                return None

            # 手动计算指标
            df['mean_price'] = df['Close'].ewm(span=self.lookback, adjust=False).mean()
            df['std_price'] = df['Close'].rolling(self.lookback).std()
            df['upper_band'] = df['mean_price'] + self.std_dev * df['std_price']
            df['lower_band'] = df['mean_price'] - self.std_dev * df['std_price']

            # 使用均值回归策略
            signals = ema_mean_reversion_strategy(df, lookback=self.lookback, std_dev=self.std_dev)

            # 捕获本周期新增信号，避免只看最后一根导致的漏单
            if self.last_signal_ts is not None:
                recent_signals = signals.loc[signals.index > self.last_signal_ts]
            else:
                recent_signals = signals

            if not recent_signals.empty:
                latest_trigger_row = recent_signals[recent_signals.any(axis=1)].tail(1)
                if not latest_trigger_row.empty:
                    # 将最新触发行推送到信号末尾，便于后续统一读取
                    signals = signals.copy()
                    signals = signals[signals.index <= latest_trigger_row.index[-1]]

            # 详细的信号调试信息
            latest_signal = signals.iloc[-1]
            latest_price = df['Close'].iloc[-1]
            latest_mean = df['mean_price'].iloc[-1]
            latest_upper = df['upper_band'].iloc[-1]
            latest_lower = df['lower_band'].iloc[-1]
            latest_std = df['std_price'].iloc[-1]

            logger.info(f"📊 信号分析 - 价格: ${latest_price:.2f}")
            logger.info(f"📈 均值: ${latest_mean:.2f} (±${latest_std:.2f})")
            logger.info(f"📊 上轨: ${latest_upper:.2f} | 下轨: ${latest_lower:.2f}")
            logger.info(f"🚨 信号状态: 多头={latest_signal['long_entry']}, 空头={latest_signal['short_entry']}")
            logger.info(f"🚨 平仓信号: 多平={latest_signal['long_exit']}, 空平={latest_signal['short_exit']}")

            # 计算当前价格在布林带中的位置
            if latest_std > 0:
                z_score = (latest_price - latest_mean) / latest_std
                logger.info(f"📊 Z-Score: {z_score:.2f} (价格距离均值{abs(z_score):.2f}个标准差)")

            return signals

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return None

    def check_risk_limits(self):
        """检查风险限制"""
        try:
            # 确保有有效的初始余额，避免除零错误
            if self.daily_start_balance == 0 or self.daily_start_balance <= 0:
                self.daily_start_balance = self.current_balance
                logger.info(f"🎯 设置日初余额: ${self.daily_start_balance:.2f}")
                return True

            # 计算日收益率
            daily_pnl_pct = (self.current_balance - self.daily_start_balance) / self.daily_start_balance

            # 添加调试信息，仅在余额有显著变化时记录
            if abs(daily_pnl_pct) > 0.001:  # 超过0.1%才记录
                logger.info(f"📊 日收益率: {daily_pnl_pct:+.2%} (当前: ${self.current_balance:.2f}, 日初: ${self.daily_start_balance:.2f})")

            # 检查亏损限制
            if (
                self.daily_loss_limit is not None
                and self.daily_loss_limit > 0
                and daily_pnl_pct < -self.daily_loss_limit
            ):
                logger.warning(f"⚠️ 触发日亏损限制: {daily_pnl_pct:+.2%}")
                logger.warning(f"💰 日初余额: ${self.daily_start_balance:.2f}")
                logger.warning(f"💰 当前余额: ${self.current_balance:.2f}")
                logger.warning(f"📉 亏损金额: ${self.current_balance - self.daily_start_balance:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return True  # 出错时允许继续交易，避免误停

    def run_trading_cycle(self):
        """执行一个交易周期"""
        try:
            logger.info("🔄 开始新的交易周期...")

            # 1. 获取市场数据
            df = self.get_current_data()
            self.last_price_df = df
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
            equity_time = df.index[-1].to_pydatetime() if not df.empty else datetime.now()
            self._record_equity(total_equity, equity_time)

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

            # 7. 检查最新信号（对齐触发信号对应的K线）
            signal_time = signals.index[-1]
            latest_signal = signals.loc[signal_time]
            latest_price = df.loc[signal_time, 'Close']
            mean_price = df.loc[signal_time, 'mean_price']
            upper_band = df.loc[signal_time, 'upper_band']
            lower_band = df.loc[signal_time, 'lower_band']

            logger.info(f"📊 当前价格: ${latest_price:,.2f}")
            logger.info(f"🕒 信号时间: {signal_time}")
            logger.info(f"📈 均值线: ${mean_price:,.2f}")
            logger.info(f"📊 上轨/下轨: ${upper_band:,.2f} / ${lower_band:,.2f}")
            logger.info(f"💰 当前余额: ${self.current_balance:.2f}")
            logger.info(f"📈 当前持仓: {self.position:.6f}")
            logger.info(f"💹 未实现盈亏: ${self.unrealized_pnl:+.2f}")

            # 记录已处理的最新信号时间
            self.last_signal_ts = signal_time

            if self.position != 0 and self.entry_price:
                decision = self.stop_loss_policy.decide_exit(
                    1 if self.position > 0 else -1,
                    self.entry_price,
                    latest_price,
                )
                if decision.should_exit:
                    close_reason = f"止损触发({decision.reason})"
                    logger.warning(
                        f"ð {close_reason}: å½å=${latest_price:.2f}, å¥åº=${self.entry_price:.2f}"
                    )
                    side = 'sell' if self.position > 0 else 'buy'
                    result = self.place_order(side, abs(self.position), 'market')
                    if result.get('success'):
                        execution_price = latest_price
                        if self.position > 0:
                            pnl = (execution_price - self.entry_price) * self.position
                            trade = {
                                'time': datetime.now(),
                                'action': 'SELL',
                                'price': execution_price,
                                'position': -self.position,
                                'pnl': pnl,
                                'balance': self.current_balance + pnl,
                                'reason': close_reason,
                                'type': 'close_long'
                            }
                        else:
                            pnl = (self.entry_price - execution_price) * abs(self.position)
                            trade = {
                                'time': datetime.now(),
                                'action': 'BUY',
                                'price': execution_price,
                                'position': abs(self.position),
                                'pnl': pnl,
                                'balance': self.current_balance + pnl,
                                'reason': close_reason,
                                'type': 'close_short'
                            }
                        self.current_balance += pnl
                        self.trades.append(trade)
                        logger.info(
                            f"✅ 止损平仓: ${execution_price:.2f}, 盈亏: ${pnl:+.2f} USDT"
                        )
                        self.position = 0
                        self.entry_price = 0
                    else:
                        logger.error(f"❌ 止损平仓失败: {result}")
                    return

            # 9. 执行交易逻辑
            if self.position == 0:  # 无持仓
                if latest_signal['long_entry']:
                    # 开多仓
                    position_size = self.calculate_position_size(latest_price)
                    open_reason = "价格触及下轨"
                    logger.info(f"🎯 开多理由: {open_reason}")
                    logger.info(f"🎯 尝试开多仓: 价格=${latest_price:.2f}, 仓位={position_size:.6f}, 原因={open_reason}")
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
                            'reason': open_reason,
                            'type': 'open_long'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 开多仓成功: ${latest_price:.2f}, 仓位: {position_size:.6f}")
                    else:
                        logger.error(f"❌ 开多仓失败: {result}")

                elif latest_signal['short_entry']:
                    # 开空仓
                    position_size = self.calculate_position_size(latest_price)
                    open_reason = "价格触及上轨"
                    logger.info(f"🎯 开空理由: {open_reason}")
                    logger.info(f"🎯 尝试开空仓: 价格=${latest_price:.2f}, 仓位={position_size:.6f}, 原因={open_reason}")
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
                            'reason': open_reason,
                            'type': 'open_short'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 开空仓成功: ${latest_price:.2f}, 仓位: {position_size:.6f}")
                    else:
                        logger.error(f"❌ 开空仓失败: {result}")
                else:
                    logger.info(f"⚪ 无持仓且无开仓信号，继续观察")

            elif self.position > 0:  # 持有多仓
                # 检查平仓条件
                should_close = latest_signal['long_exit'] or latest_price >= mean_price
                if should_close:
                    close_reason = "信号触发" if latest_signal['long_exit'] else "价格回归均值"
                    logger.info(f"🎯 平多理由: {close_reason}")
                    # 平多仓
                    logger.info(f"🎯 尝试平多仓: 当前=${latest_price:.2f}, 入场=${self.entry_price:.2f}, 原因={close_reason}")
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
                            'reason': close_reason,
                            'type': 'close_long'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 平多仓: ${execution_price:.2f}, 盈亏: ${pnl:+.2f} USDT ({pnl/(self.position*self.entry_price)*100:+.2f}%)")
                        self.position = 0
                        self.entry_price = 0
                    else:
                        logger.error(f"❌ 平多仓失败: {result}")
                else:
                    logger.info(f"📈 持有多仓观望: 当前=${latest_price:.2f}, 入场=${self.entry_price:.2f}, 盈亏=${(latest_price-self.entry_price)*self.position:+.2f}")

            elif self.position < 0:  # 持有空仓
                # 检查平仓条件
                should_close = latest_signal['short_exit'] or latest_price <= mean_price
                if should_close:
                    close_reason = "信号触发" if latest_signal['short_exit'] else "价格回归均值"
                    logger.info(f"🎯 平空理由: {close_reason}")
                    # 平空仓
                    logger.info(f"🎯 尝试平空仓: 当前=${latest_price:.2f}, 入场=${self.entry_price:.2f}, 原因={close_reason}")
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
                            'reason': close_reason,
                            'type': 'close_short'
                        }
                        self.trades.append(trade)

                        logger.info(f"✅ 平空仓: ${execution_price:.2f}, 盈亏: ${pnl:+.2f} USDT ({pnl/(abs(self.position)*self.entry_price)*100:+.2f}%)")
                        self.position = 0
                        self.entry_price = 0
                    else:
                        logger.error(f"❌ 平空仓失败: {result}")
                else:
                    logger.info(f"📉 持有空仓观望: 当前=${latest_price:.2f}, 入场=${self.entry_price:.2f}, 盈亏=${(self.entry_price-latest_price)*abs(self.position):+.2f}")

            logger.info(f"✅ 交易周期完成")

        except Exception as e:
            logger.error(f"❌ 交易周期执行失败: {e}")
        finally:
            if self.visualizer_enabled:
                self._export_charts()

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

    def _record_equity(self, equity, timestamp=None):
        """记录权益及对应时间戳"""
        ts = pd.to_datetime(timestamp) if timestamp is not None else datetime.now()
        self.equity_history.append(float(equity))
        self.equity_timestamps.append(ts)

    def _build_equity_dataframe(self):
        """构建带时间戳的权益DataFrame"""
        if not self.equity_history:
            return pd.DataFrame(columns=['datetime', 'equity'])

        if self.equity_timestamps and len(self.equity_timestamps) == len(self.equity_history):
            timestamps = pd.to_datetime(self.equity_timestamps)
        else:
            logger.warning("⚠️ 权益时间戳缺失，使用当前时间序列补全")
            timestamps = pd.date_range(end=datetime.now(), periods=len(self.equity_history), freq='15T')

        equity_df = pd.DataFrame({
            'datetime': timestamps,
            'equity': self.equity_history
        })
        return equity_df.sort_values('datetime')

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
                equity_df = self._build_equity_dataframe()
                export_df = equity_df.rename(columns={'datetime': 'timestamp'})[['equity', 'timestamp']]
                export_df.to_csv('okx_simulation_equity.csv', index=False)
                logger.info("✅ 权益曲线已保存到 okx_simulation_equity.csv")

            # 保存图表
            if self.visualizer_enabled:
                self._export_charts()

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def _export_charts(self):
        """生成并保存K线和权益曲线图"""
        if not self.visualizer_enabled or self.last_price_df is None:
            return

        try:
            import os

            os.makedirs('charts', exist_ok=True)

            # 准备价格数据
            price_export = self.last_price_df.reset_index().rename(columns={'index': 'datetime'})
            cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            price_export = price_export[[c for c in cols if c in price_export.columns]]

            # 准备交易数据
            trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
            if not trades_df.empty:
                # 兼容 time / datetime 字段
                if 'time' in trades_df.columns and 'datetime' not in trades_df.columns:
                    trades_df['datetime'] = trades_df['time']
                trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])

            # K线 + 交易点
            kline_path = os.path.join('charts', 'live_kline.png')
            self.visualizer.plot_kline_with_trades(
                price_data=price_export,
                trades_data=trades_df,
                title="实盘/模拟 - K线与交易点",
                save_path=kline_path
            )

            # 权益曲线
            equity_df = self._build_equity_dataframe()
            if not equity_df.empty:
                equity_path = os.path.join('charts', 'live_equity.png')
                self.visualizer.plot_equity_curve(
                    equity_data=equity_df,
                    title="实盘/模拟 - 权益曲线",
                    save_path=equity_path
                )

        except Exception as e:
            logger.error(f"生成图表失败: {e}")

    def _save_live_data(self, df):
        """将实时K线数据写入CSV，便于复盘"""
        try:
            import os
            os.makedirs('data_capture', exist_ok=True)
            export = df.reset_index().rename(columns={'timestamp': 'datetime'})
            export.to_csv('data_capture/live_kline.csv', index=False)
            logger.info("📝 已保存最新K线数据到 data_capture/live_kline.csv")
        except Exception as e:
            logger.error(f"实时数据保存失败: {e}")

    def print_final_report(self):
        """打印最终报告"""
        try:
            total_equity = self.current_balance + self.unrealized_pnl
            total_return = (total_equity - self.initial_balance) / self.initial_balance * 100

            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                closed_trades = trades_df[trades_df.get('pnl').notna()]
                profitable_trades = closed_trades[closed_trades['pnl'] > 0]
                win_rate = (len(profitable_trades) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0

                avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
                losing_trades = closed_trades[closed_trades['pnl'] <= 0]
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
            total_closed = len(closed_trades) if self.trades else 0
            print(f"已平仓交易次数: {total_closed}")
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

                # 改为1分钟间隔，提高响应速度（15分钟K线，1分钟检查）
                cycle_time = time.time() - cycle_start
                wait_time = max(0, 60 - cycle_time)  # 1分钟间隔

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
        trading_duration = 3600  # 默认60分钟

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
