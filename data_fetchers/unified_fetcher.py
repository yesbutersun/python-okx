# ==============================
# 统一数据获取接口
# ==============================
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

try:
    from .okx_fetcher import OKXDataFetcher
    from .binance_fetcher import BinanceDataFetcher
except ImportError:
    # 处理相对导入失败的情况
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from okx_fetcher import OKXDataFetcher
    from binance_fetcher import BinanceDataFetcher

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../unified_fetcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedDataFetcher:
    """统一数据获取接口，支持多个交易所"""

    def __init__(self, exchange: str = 'binance', **kwargs):
        """
        初始化统一数据获取器

        Args:
            exchange: 交易所名称 ('okx' 或 'binance')
            **kwargs: 传递给具体获取器的参数
        """
        self.exchange = exchange.lower()
        self.data_dir = '../stock_data'
        self._ensure_data_dir()

        # 初始化具体的获取器
        if self.exchange == 'okx':
            self.fetcher = OKXDataFetcher(**kwargs)
        elif self.exchange == 'binance':
            self.fetcher = BinanceDataFetcher(**kwargs)
        else:
            raise ValueError(f"不支持的交易所: {exchange}")

        logger.info(f"初始化{exchange.upper()}数据获取器成功")

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def fetch_historical_data(self, symbol: str, interval: str = '1h',
                            start_time: datetime = None, end_time: datetime = None,
                            **kwargs) -> List[Dict]:
        """
        获取历史K线数据

        Args:
            symbol: 交易对/产品ID
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数

        Returns:
            K线数据列表
        """
        logger.info(f"使用{self.exchange.upper()}获取{symbol}历史数据")
        return self.fetcher.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            **kwargs
        )

    def fetch_and_save(self, symbol: str, interval: str = '1h',
                      start_time: datetime = None, end_time: datetime = None,
                      filename: str = None, **kwargs) -> str:
        """
        获取并保存历史数据

        Args:
            symbol: 交易对/产品ID
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            filename: 文件名
            **kwargs: 其他参数

        Returns:
            保存的文件路径
        """
        logger.info(f"使用{self.exchange.upper()}获取并保存{symbol}数据")
        return self.fetcher.fetch_and_save(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            filename=filename,
            **kwargs
        )

    def get_available_symbols(self) -> List[str]:
        """获取可用的交易对"""
        if hasattr(self.fetcher, 'get_available_symbols'):
            return self.fetcher.get_available_symbols()
        elif hasattr(self.fetcher, 'get_available_instruments'):
            return self.fetcher.get_available_instruments()
        else:
            return []

    def get_time_intervals(self) -> Dict[str, str]:
        """获取支持的时间间隔"""
        if hasattr(self.fetcher, 'get_time_intervals'):
            return self.fetcher.get_time_intervals()
        elif hasattr(self.fetcher, 'get_time_ranges'):
            return self.fetcher.get_time_ranges()
        else:
            return {}

    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        if hasattr(self.fetcher, 'fetch_latest_price'):
            return self.fetcher.fetch_latest_price(symbol)
        else:
            # 如果没有直接获取最新价格的方法，尝试通过ticker获取
            try:
                if self.exchange == 'okx':
                    ticker = self.fetcher.client.get_ticker(symbol)
                    return float(ticker['data'][0]['last'])
                elif self.exchange == 'binance':
                    ticker = self.fetcher.client.get_ticker(symbol)
                    return float(ticker['lastPrice'])
            except Exception as e:
                logger.error(f"获取最新价格失败: {e}")
            return 0.0

    def test_connection(self) -> bool:
        """测试连接"""
        return self.fetcher.client.test_connection()


class MultiExchangeDataFetcher:
    """多交易所数据获取器"""

    def __init__(self, exchanges_config: Dict[str, Dict]):
        """
        初始化多交易所数据获取器

        Args:
            exchanges_config: 交易所配置字典
                格式: {'binance': {'api_key': 'xxx', 'secret_key': 'yyy', 'testnet': True},
                       'okx': {'api_key': 'xxx', 'secret_key': 'yyy', 'passphrase': 'zzz', 'sandbox': True}}
        """
        self.fetchers = {}
        self.data_dir = '../stock_data'
        self._ensure_data_dir()

        for exchange, config in exchanges_config.items():
            try:
                self.fetchers[exchange] = UnifiedDataFetcher(exchange, **config)
                logger.info(f"成功初始化{exchange.upper()}数据获取器")
            except Exception as e:
                logger.error(f"初始化{exchange.upper()}数据获取器失败: {e}")

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def fetch_from_all_exchanges(self, symbol_map: Dict[str, str],
                                interval: str = '1h', start_time: datetime = None,
                                end_time: datetime = None) -> Dict[str, List[Dict]]:
        """
        从所有交易所获取数据

        Args:
            symbol_map: 交易对映射字典 {'binance': 'BTCUSDT', 'okx': 'BTC-USDT-SWAP'}
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            各交易所的数据字典
        """
        results = {}
        for exchange, symbol in symbol_map.items():
            if exchange in self.fetchers:
                try:
                    data = self.fetchers[exchange].fetch_historical_data(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    results[exchange] = data
                    logger.info(f"从{exchange.upper()}获取到{len(data)}条数据")
                except Exception as e:
                    logger.error(f"从{exchange.upper()}获取数据失败: {e}")
            else:
                logger.warning(f"未初始化{exchange.upper()}获取器")
        return results

    def save_from_all_exchanges(self, symbol_map: Dict[str, str],
                               interval: str = '1h', start_time: datetime = None,
                               end_time: datetime = None) -> Dict[str, str]:
        """
        从所有交易所获取并保存数据

        Args:
            symbol_map: 交易对映射字典
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            各交易所保存的文件路径字典
        """
        filepaths = {}
        for exchange, symbol in symbol_map.items():
            if exchange in self.fetchers:
                try:
                    filepath = self.fetchers[exchange].fetch_and_save(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    if filepath:
                        filepaths[exchange] = filepath
                except Exception as e:
                    logger.error(f"从{exchange.upper()}获取并保存数据失败: {e}")
        return filepaths

    def compare_prices(self, symbol_map: Dict[str, str]) -> Dict[str, float]:
        """
        比较各交易所的当前价格

        Args:
            symbol_map: 交易对映射字典

        Returns:
            各交易所的价格字典
        """
        prices = {}
        for exchange, symbol in symbol_map.items():
            if exchange in self.fetchers:
                try:
                    price = self.fetchers[exchange].get_latest_price(symbol)
                    prices[exchange] = price
                    logger.info(f"{exchange.upper()} {symbol} 价格: ${price}")
                except Exception as e:
                    logger.error(f"获取{exchange.upper()}价格失败: {e}")
        return prices

    def get_all_available_symbols(self) -> Dict[str, List[str]]:
        """获取所有交易所的可用交易对"""
        symbols = {}
        for exchange, fetcher in self.fetchers.items():
            try:
                symbols[exchange] = fetcher.get_available_symbols()
            except Exception as e:
                logger.error(f"获取{exchange.upper()}交易对失败: {e}")
        return symbols


def main():
    """主函数，用于测试"""
    try:
        # 测试单一交易所数据获取
        print("=== 测试单一交易所数据获取 ===")
        fetcher = UnifiedDataFetcher('binance', testnet=True)

        if fetcher.test_connection():
            print("✅ 连接成功")

            # 设置时间范围（最近3天）
            end_time = datetime.now()
            start_time = end_time - timedelta(days=3)

            # 获取并保存数据
            filepath = fetcher.fetch_and_save(
                symbol='BTCUSDT',
                interval='1h',
                start_time=start_time,
                end_time=end_time
            )

            if filepath:
                print(f"✅ 数据保存成功: {filepath}")

        # 测试多交易所数据获取
        print("\n=== 测试多交易所数据获取 ===")
        exchanges_config = {
            'binance': {'testnet': True},
            # 'okx': {'sandbox': True}  # 如果有OKX配置可以启用
        }

        multi_fetcher = MultiExchangeDataFetcher(exchanges_config)

        # 获取价格比较
        symbol_map = {'binance': 'BTCUSDT'}
        prices = multi_fetcher.compare_prices(symbol_map)

        print("各交易所价格:")
        for exchange, price in prices.items():
            print(f"  {exchange.upper()}: ${price}")

    except Exception as e:
        logger.error(f"主程序执行出错: {e}")


if __name__ == "__main__":
    main()