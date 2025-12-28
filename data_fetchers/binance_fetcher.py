# ==============================
# Binance 历史K线数据获取器
# ==============================
import csv
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd

from providers.binance_api import BinanceAPI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../binance_data_fetcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Binance历史数据获取器"""

    def __init__(self, api_key: str = None, secret_key: str = None,
                 testnet: bool = True):
        """
        初始化Binance数据获取器

        Args:
            api_key: API密钥
            secret_key: 密钥
            testnet: 是否使用测试网环境
        """
        self.client = BinanceAPI(api_key, secret_key, testnet)
        self.data_dir = '../stock_data'
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def _timestamp_to_datetime(self, timestamp: int) -> datetime:
        """将时间戳转换为datetime对象"""
        return datetime.fromtimestamp(timestamp / 1000)

    def _datetime_to_timestamp(self, dt: datetime) -> int:
        """将datetime对象转换为时间戳"""
        return int(dt.timestamp() * 1000)

    def _parse_kline_data(self, raw_data: List, symbol: str = None) -> List[Dict]:
        """
        解析K线数据

        Args:
            raw_data: Binance返回的原始K线数据
            symbol: 交易对（用于回填，避免API末尾字段误用）

        Returns:
            解析后的K线数据列表
        """
        parsed_data = []
        for kline in raw_data:
            # Binance K线数据格式: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            parsed_data.append({
                'timestamp': int(kline[0]),
                'datetime': self._timestamp_to_datetime(kline[0]).strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_timestamp': int(kline[6]),
                'close_datetime': self._timestamp_to_datetime(kline[6]).strftime('%Y-%m-%d %H:%M:%S'),
                'quote_volume': float(kline[7]),
                'trades_count': int(kline[8]),
                'taker_buy_volume': float(kline[9]),
                'taker_buy_quote_volume': float(kline[10]),
                # Binance candles 返回末尾字段通常是 ignore，并不包含 symbol；
                # 这里优先使用传入的 symbol 以避免保存成 "0_kline_..."。
                'symbol': symbol if symbol is not None else (kline[11] if len(kline) > 11 else '')
            })
        return parsed_data

    def fetch_historical_data(self, symbol: str = 'BTCUSDT',
                            interval: str = '1h', start_time: datetime = None,
                            end_time: datetime = None, limit: int = 500) -> List[Dict]:
        """
        获取历史K线数据

        Args:
            symbol: 交易对 (如 'BTCUSDT')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            start_time: 开始时间
            end_time: 结束时间
            limit: 单次请求数量限制 (最大1500)

        Returns:
            K线数据列表
        """
        all_data = []

        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        current_start = start_time

        logger.info(f"开始获取 {symbol} {interval} K线数据")
        logger.info(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        while current_start < end_time:
            try:
                # 转换时间为Binance格式
                start_timestamp = self._datetime_to_timestamp(current_start)
                end_timestamp = min(self._datetime_to_timestamp(end_time), start_timestamp + 1500 * self._get_interval_ms(interval))

                # 请求数据
                response = self.client.get_candles(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_timestamp,
                    end_time=end_timestamp,
                    limit=min(limit, 1500)
                )

                if not response:
                    logger.warning("没有获取到数据，停止获取")
                    break

                # 解析数据
                candles = self._parse_kline_data(response, symbol=symbol)

                if not candles:
                    logger.info("没有更多数据，停止获取")
                    break

                # 添加到总数据
                all_data.extend(candles)

                # 更新开始时间为最后一条数据的时间
                last_time = datetime.strptime(candles[-1]['datetime'], '%Y-%m-%d %H:%M:%S')
                current_start = last_time + timedelta(minutes=1)  # 留1分钟间隔避免重复

                logger.info(f"已获取 {len(candles)} 条数据，总计 {len(all_data)} 条")

                # 避免API频率限制
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"获取数据时出错: {e}")
                break

        logger.info(f"总共获取 {len(all_data)} 条K线数据")
        return all_data

    def _get_interval_ms(self, interval: str) -> int:
        """
        获取时间间隔对应的毫秒数

        Args:
            interval: 时间间隔字符串

        Returns:
            毫秒数
        """
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,  # 近似值
        }
        return interval_map.get(interval, 60 * 60 * 1000)  # 默认1小时

    def save_to_csv(self, data: List[Dict], filename: str = None, symbol: str = None) -> str:
        """
        将数据保存为CSV文件

        Args:
            data: K线数据列表
            filename: 文件名（可选）

        Returns:
            保存的文件路径
        """
        if not data:
            logger.warning("没有数据可保存")
            return None

        # 生成文件名
        if filename is None:
            start_time = datetime.strptime(data[0]['datetime'], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(data[-1]['datetime'], '%Y-%m-%d %H:%M:%S')

            file_symbol = symbol if symbol is not None else data[0].get('symbol', 'BTCUSDT')
            file_symbol = str(file_symbol).strip()
            if not file_symbol or file_symbol.lower() == 'none' or file_symbol == '0':
                file_symbol = 'BTCUSDT'

            filename = f"{file_symbol}_kline_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"

        filepath = os.path.join(self.data_dir, filename)

        # 保存为CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'datetime', 'open', 'high', 'low', 'close',
                         'volume', 'close_timestamp', 'close_datetime', 'quote_volume',
                         'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'symbol']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"数据已保存到: {filepath}")
        return filepath

    def save_to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """
        将数据转换为DataFrame

        Args:
            data: K线数据列表

        Returns:
            pandas DataFrame
        """
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['close_datetime'] = pd.to_datetime(df['close_datetime'])
        df.set_index('datetime', inplace=True)

        # 确保数值列的数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def fetch_and_save(self, symbol: str = 'BTCUSDT', interval: str = '1h',
                      start_time: datetime = None, end_time: datetime = None,
                      filename: str = None) -> str:
        """
        获取并保存历史数据

        Args:
            symbol: 交易对
            interval: K线间隔
            start_time: 开始时间
            end_time: 结束时间
            filename: 文件名

        Returns:
            保存的文件路径
        """
        # 获取数据
        data = self.fetch_historical_data(symbol, interval, start_time, end_time)

        if not data:
            logger.error("没有获取到数据")
            return None

        # 保存数据
        return self.save_to_csv(data, filename, symbol=symbol)

    def get_available_symbols(self) -> List[str]:
        """
        获取可用的交易对

        Returns:
            交易对列表
        """
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = []
            for symbol_info in exchange_info.get('symbols', []):
                if symbol_info.get('status') == 'TRADING':
                    symbols.append(symbol_info['symbol'])
            return sorted(symbols)
        except Exception as e:
            logger.error(f"获取交易对列表失败: {e}")
            # 返回一些常见的交易对
            return [
                'BTCUSDT',
                'ETHUSDT',
                'SOLUSDT',
                'DOGEUSDT',
                'ADAUSDT',
                'BNBUSDT',
                'XRPUSDT',
                'DOTUSDT'
            ]

    def get_time_intervals(self) -> Dict[str, str]:
        """
        获取支持的时间间隔

        Returns:
            时间间隔字典
        """
        return {
            '1m': '1分钟',
            '3m': '3分钟',
            '5m': '5分钟',
            '15m': '15分钟',
            '30m': '30分钟',
            '1h': '1小时',
            '2h': '2小时',
            '4h': '4小时',
            '6h': '6小时',
            '8h': '8小时',
            '12h': '12小时',
            '1d': '1天',
            '3d': '3天',
            '1w': '1周',
            '1M': '1月'
        }

    def fetch_latest_price(self, symbol: str = 'BTCUSDT') -> float:
        """
        获取最新价格

        Args:
            symbol: 交易对

        Returns:
            最新价格
        """
        try:
            ticker = self.client.get_ticker(symbol)
            return float(ticker.get('lastPrice', 0))
        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            return 0.0


def main():
    """主函数，用于测试"""
    try:
        # 创建数据获取器
        fetcher = BinanceDataFetcher(testnet=True)

        # 测试连接
        if not fetcher.client.test_connection():
            logger.error("API连接失败")
            return

        # 设置时间范围（最近7天）
        symbol="ETHUSDT"
        interval ='15m'
        end_date_str = "2025-12-31"
        start_date_str = "2025-01-01"
        end_time = datetime.fromisoformat(end_date_str)
        start_time = datetime.fromisoformat(start_date_str)


        # 获取BTCUSDT 1小时K线数据
        filepath = fetcher.fetch_and_save(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        logger.info(f"开始获取{symbol}-{start_time}-{end_time}-{interval}K线数据...")

        if filepath:
            print(f"✅ 数据获取成功！")
            print(f"📁 文件路径: {filepath}")

            # 读取并显示前几行数据
            df = pd.read_csv(filepath)
            print(f"📊 数据形状: {df.shape}")
            print("\n前5行数据:")
            print(df.head())

    except Exception as e:
        logger.error(f"主程序执行出错: {e}")


if __name__ == "__main__":
    main()
