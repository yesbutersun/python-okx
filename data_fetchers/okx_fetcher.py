# ==============================
# OKX 历史K线数据获取器
# ==============================
import csv
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd

try:
    from ..providers.okx_api import OKXAPI
except ImportError:
    # 处理相对导入失败的情况
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from providers.okx_api import OKXAPI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../okx_data_fetcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OKXDataFetcher:
    """OKX历史数据获取器"""

    def __init__(self, api_key: str = None, secret_key: str = None,
                 passphrase: str = None, sandbox: bool = True):
        """
        初始化OKX数据获取器

        Args:
            api_key: API密钥
            secret_key: 密钥
            passphrase: 口令
            sandbox: 是否使用沙盒环境
        """
        self.client = OKXAPI(api_key, secret_key, passphrase, sandbox)
        self.data_dir = '../stock_data'
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def _timestamp_to_datetime(self, timestamp: str) -> datetime:
        """将时间戳转换为datetime对象"""
        return datetime.fromtimestamp(int(timestamp) / 1000)

    def _datetime_to_timestamp(self, dt: datetime) -> str:
        """将datetime对象转换为时间戳"""
        return str(int(dt.timestamp() * 1000))

    def _parse_candle_data(self, raw_data: List) -> List[Dict]:
        """
        解析K线数据

        Args:
            raw_data: OKX返回的原始K线数据

        Returns:
            解析后的K线数据列表
        """
        parsed_data = []
        for candle in raw_data:
            # OKX K线数据格式: [timestamp, open, high, low, close, volume, volume_currency, volume_currency_quote, confirm]
            parsed_data.append({
                'timestamp': int(candle[0]),
                'datetime': self._timestamp_to_datetime(candle[0]).strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),  # 成交量（以交易货币计）
                'volume_currency': float(candle[6]),  # 成交量（以计价货币计）
                'volume_currency_quote': float(candle[7]),  # 成交额
                'confirm': candle[8] == '1'  # 是否确认
            })
        return parsed_data

    def fetch_historical_data(self, inst_id: str = 'BTC-USDT-SWAP',
                            bar: str = '1H', start_time: datetime = None,
                            end_time: datetime = None, limit: int = 100) -> List[Dict]:
        """
        获取历史K线数据

        Args:
            inst_id: 产品ID (如 'BTC-USDT-SWAP')
            bar: K线周期 (如 '1m', '5m', '15m', '1H', '4H', '1D')
            start_time: 开始时间
            end_time: 结束时间
            limit: 单次请求数量限制 (最大100)

        Returns:
            K线数据列表
        """
        all_data = []

        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        current_end = end_time

        logger.info(f"开始获取 {inst_id} {bar} K线数据")
        logger.info(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        while current_end > start_time:
            try:
                # 转换时间为OKX格式
                after = self._datetime_to_timestamp(current_end)

                # 请求数据
                response = self.client.get_candles(
                    inst_id=inst_id,
                    bar=bar,
                    after=after,
                    limit=min(limit, 100)
                )

                if 'data' not in response or not response['data']:
                    logger.warning("没有获取到数据，停止获取")
                    break

                # 解析数据
                candles = self._parse_candle_data(response['data'])

                # 过滤时间范围
                filtered_candles = [
                    candle for candle in candles
                    if datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S') >= start_time
                ]

                if not filtered_candles:
                    logger.info("已达到开始时间，停止获取")
                    break

                # 添加到总数据（按时间正序）
                all_data = filtered_candles + all_data

                # 更新结束时间为最早的数据时间
                earliest_time = datetime.strptime(filtered_candles[0]['datetime'], '%Y-%m-%d %H:%M:%S')
                current_end = earliest_time - timedelta(minutes=1)  # 留1分钟间隔避免重复

                logger.info(f"已获取 {len(filtered_candles)} 条数据，总计 {len(all_data)} 条")

                # 避免API频率限制
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"获取数据时出错: {e}")
                break

        logger.info(f"总共获取 {len(all_data)} 条K线数据")
        return all_data

    def save_to_csv(self, data: List[Dict], filename: str = None) -> str:
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
            inst_id = data[0].get('inst_id', 'BTC-USDT-SWAP').replace('-', '_')
            filename = f"{inst_id}_kline_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"

        filepath = os.path.join(self.data_dir, filename)

        # 保存为CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'datetime', 'open', 'high', 'low', 'close',
                         'volume', 'volume_currency', 'volume_currency_quote', 'confirm']
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
        df.set_index('datetime', inplace=True)

        # 确保数值列的数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'volume_currency', 'volume_currency_quote']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def fetch_and_save(self, inst_id: str = 'BTC-USDT-SWAP', bar: str = '1H',
                      start_time: datetime = None, end_time: datetime = None,
                      filename: str = None) -> str:
        """
        获取并保存历史数据

        Args:
            inst_id: 产品ID
            bar: K线周期
            start_time: 开始时间
            end_time: 结束时间
            filename: 文件名

        Returns:
            保存的文件路径
        """
        # 获取数据
        data = self.fetch_historical_data(inst_id, bar, start_time, end_time)

        if not data:
            logger.error("没有获取到数据")
            return None

        # 保存数据
        return self.save_to_csv(data, filename)

    def get_available_instruments(self) -> List[str]:
        """
        获取可用的交易对（需要实现OKX公共API调用）

        Returns:
            交易对列表
        """
        # 这里可以添加获取交易对列表的逻辑
        # 暂时返回一些常见的交易对
        return [
            'BTC-USDT-SWAP',
            'ETH-USDT-SWAP',
            'SOL-USDT-SWAP',
            'DOGE-USDT-SWAP',
            'ADA-USDT-SWAP'
        ]

    def get_time_ranges(self) -> Dict[str, str]:
        """
        获取支持的时间周期

        Returns:
            时间周期字典
        """
        return {
            '1m': '1分钟',
            '3m': '3分钟',
            '5m': '5分钟',
            '15m': '15分钟',
            '30m': '30分钟',
            '1H': '1小时',
            '2H': '2小时',
            '4H': '4小时',
            '6H': '6小时',
            '12H': '12小时',
            '1D': '1天',
            '1W': '1周',
            '1M': '1月'
        }


def main():
    """主函数，用于测试"""
    try:
        # 创建数据获取器
        fetcher = OKXDataFetcher(sandbox=True)

        # 测试连接
        if not fetcher.client.test_connection():
            logger.error("API连接失败")
            return

        # 设置时间范围（最近7天）
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        # 获取BTC-USDT-SWAP 1小时K线数据
        logger.info("开始获取BTC 1小时K线数据...")
        filepath = fetcher.fetch_and_save(
            inst_id='BTC-USDT-SWAP',
            bar='1H',
            start_time=start_time,
            end_time=end_time
        )

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