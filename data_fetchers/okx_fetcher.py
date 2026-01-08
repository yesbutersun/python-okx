# ==============================
# OKX 历史K线数据获取器（使用 okx SDK）
# ==============================
import csv
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

try:
    from okx.MarketData import MarketAPI
    from okx.PublicData import PublicAPI
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from okx.MarketData import MarketAPI
    from okx.PublicData import PublicAPI


LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "okx_data_fetcher.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class OKXDataFetcher:
    """OKX历史数据获取器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        sandbox: bool = True,
        data_dir: Optional[str] = None,
    ):
        """
        初始化OKX数据获取器

        Args:
            api_key: API密钥（公共接口可不填）
            secret_key: 密钥（公共接口可不填）
            passphrase: 口令（公共接口可不填）
            sandbox: 是否使用沙盒环境
            data_dir: 数据保存目录
        """
        flag = "1" if sandbox else "0"
        api_key = api_key or "-1"
        secret_key = secret_key or "-1"
        passphrase = passphrase or "-1"

        self.market_api = MarketAPI(api_key, secret_key, passphrase, flag=flag)
        self.public_api = PublicAPI(api_key, secret_key, passphrase, flag=flag)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "stock_data"
        )
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = self.public_api.get_system_time()
            if response and response.get("code") == "0":
                ts = response.get("data", [{}])[0].get("ts")
                logger.info(f"API连接成功，服务器时间: {ts}")
                return True
            logger.error(f"API连接失败: {response}")
            return False
        except Exception as e:
            logger.error(f"API连接失败: {e}")
            return False

    def _timestamp_to_datetime(self, timestamp: str) -> datetime:
        """将时间戳转换为datetime对象"""
        return datetime.fromtimestamp(int(timestamp) / 1000)

    def _datetime_to_timestamp(self, dt: datetime) -> str:
        """将datetime对象转换为时间戳"""
        return str(int(dt.timestamp() * 1000))

    def _parse_candle_data(
        self,
        inst_id: str,
        raw_data: List,
        *,
        symbol: str,
        bar_ms: int,
    ) -> List[Dict]:
        """
        解析K线数据

        Args:
            inst_id: 产品ID
            raw_data: OKX返回的原始K线数据

        Returns:
            解析后的K线数据列表
        """
        parsed_data: List[Dict] = []
        for candle in raw_data:
            if len(candle) < 9:
                continue
            open_ts = int(candle[0])
            close_ts = open_ts + bar_ms
            parsed_data.append(
                {
                    "timestamp": open_ts,
                    "datetime": self._timestamp_to_datetime(candle[0]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "close_timestamp": close_ts,
                    "close_datetime": self._timestamp_to_datetime(close_ts).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "quote_volume": float(candle[7]),
                    "trades_count": 0,
                    "taker_buy_volume": 0.0,
                    "taker_buy_quote_volume": 0.0,
                    "symbol": symbol,
                }
            )
        return parsed_data

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """将简化symbol转换为OKX instId。"""
        if not symbol:
            return "BTC-USDT-SWAP"
        if "-" in symbol:
            return symbol
        upper = symbol.upper()
        if upper.endswith("USDT"):
            base = upper[:-4]
            return f"{base}-USDT-SWAP"
        return upper

    @staticmethod
    def _normalize_symbol_for_binance(symbol: str) -> str:
        """将OKX instId 转换为 Binance 风格 symbol。"""
        if not symbol:
            return "BTCUSDT"
        upper = symbol.upper()
        if "-" in upper:
            parts = upper.split("-")
            if len(parts) >= 2:
                return f"{parts[0]}{parts[1]}"
        return upper

    @staticmethod
    def _bar_to_milliseconds(bar: str) -> int:
        """将OKX bar 转为毫秒数。"""
        if not bar:
            return 60 * 60 * 1000
        val = str(bar).strip()
        unit = val[-1]
        num = int(val[:-1])
        if unit in ("m", "M") and val.endswith("m"):
            return num * 60 * 1000
        if unit in ("H", "h"):
            return num * 60 * 60 * 1000
        if unit in ("D", "d"):
            return num * 24 * 60 * 60 * 1000
        if unit in ("W", "w"):
            return num * 7 * 24 * 60 * 60 * 1000
        if unit in ("M", "m") and val.endswith("M"):
            return num * 30 * 24 * 60 * 60 * 1000
        return 60 * 60 * 1000

    def fetch_historical_data(
        self,
        inst_id: str = "BTC-USDT-SWAP",
        bar: str = "1H",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
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
        all_data: List[Dict] = []

        inst_id = self._normalize_symbol(inst_id)
        symbol = self._normalize_symbol_for_binance(inst_id)
        bar_ms = self._bar_to_milliseconds(bar)

        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        current_end = end_time

        logger.info(f"开始获取 {inst_id} {bar} K线数据")
        logger.info(
            f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        while current_end > start_time:
            try:
                after = self._datetime_to_timestamp(current_end)

                response = self.market_api.get_candlesticks(
                    instId=inst_id, after=after, bar=bar, limit=str(min(limit, 100))
                )

                if not response or response.get("code") != "0":
                    logger.warning(f"API返回异常: {response}")
                    break

                raw_candles = response.get("data", [])
                if not raw_candles:
                    logger.warning("没有获取到数据，停止获取")
                    break

                candles = self._parse_candle_data(
                    inst_id,
                    raw_candles,
                    symbol=symbol,
                    bar_ms=bar_ms,
                )
                candles.sort(key=lambda x: x["timestamp"])

                filtered_candles = [
                    candle
                    for candle in candles
                    if datetime.strptime(candle["datetime"], "%Y-%m-%d %H:%M:%S")
                    >= start_time
                ]

                if not filtered_candles:
                    logger.info("已达到开始时间，停止获取")
                    break

                all_data = filtered_candles + all_data

                earliest_time = datetime.strptime(
                    filtered_candles[0]["datetime"], "%Y-%m-%d %H:%M:%S"
                )
                current_end = earliest_time - timedelta(minutes=1)

                logger.info(f"已获取 {len(filtered_candles)} 条数据，总计 {len(all_data)} 条")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"获取数据时出错: {e}")
                break

        logger.info(f"总共获取 {len(all_data)} 条K线数据")
        return all_data

    def save_to_csv(
        self,
        data: List[Dict],
        filename: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[str]:
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

        if filename is None:
            start_time = datetime.strptime(data[0]["datetime"], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(data[-1]["datetime"], "%Y-%m-%d %H:%M:%S")
            file_symbol = symbol if symbol is not None else data[0].get("symbol", "BTCUSDT")
            file_symbol = str(file_symbol).strip() or "BTCUSDT"
            filename = f"{file_symbol}_kline_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"

        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "timestamp",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_timestamp",
                "close_datetime",
                "quote_volume",
                "trades_count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "symbol",
            ]
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
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        df["close_datetime"] = pd.to_datetime(df["close_datetime"])
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def fetch_and_save(
        self,
        inst_id: str = "BTC-USDT-SWAP",
        bar: str = "1H",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """获取并保存历史数据"""
        data = self.fetch_historical_data(inst_id, bar, start_time, end_time)
        if not data:
            logger.error("没有获取到数据")
            return None
        symbol = self._normalize_symbol_for_binance(inst_id)
        return self.save_to_csv(data, filename, symbol=symbol)

    def fetch_by_date_range(
        self,
        symbol: str,
        interval: str,
        start_date_str: str,
        end_date_str: str,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """按字符串日期范围获取并保存数据。"""
        inst_id = self._normalize_symbol(symbol)
        start_time = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_time = datetime.strptime(end_date_str, "%Y-%m-%d")
        return self.fetch_and_save(
            inst_id=inst_id,
            bar=interval,
            start_time=start_time,
            end_time=end_time,
            filename=filename,
        )

    def get_available_instruments(self) -> List[str]:
        """获取可用的交易对（示例列表）"""
        return [
            "BTC-USDT-SWAP",
            "ETH-USDT-SWAP",
            "SOL-USDT-SWAP",
            "DOGE-USDT-SWAP",
            "ADA-USDT-SWAP",
        ]

    def get_time_ranges(self) -> Dict[str, str]:
        """获取支持的时间周期"""
        return {
            "1m": "1分钟",
            "3m": "3分钟",
            "5m": "5分钟",
            "15m": "15分钟",
            "30m": "30分钟",
            "1H": "1小时",
            "2H": "2小时",
            "4H": "4小时",
            "6H": "6小时",
            "12H": "12小时",
            "1D": "1天",
            "1W": "1周",
            "1M": "1月",
        }


def main():
    """主函数，用于测试"""
    try:
        fetcher = OKXDataFetcher(sandbox=True)

        if not fetcher.test_connection():
            logger.error("API连接失败")
            return

        symbol = "ETHUSDT"
        interval = "15m"
        start_date_str = "2025-12-20"
        end_date_str = "2026-01-31"

        logger.info("开始获取OKX历史K线数据...")
        filepath = fetcher.fetch_by_date_range(
            symbol=symbol,
            interval=interval,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )

        if filepath:
            print("✅ 数据获取成功！")
            print(f"📁 文件路径: {filepath}")

            df = pd.read_csv(filepath)
            print(f"📊 数据形状: {df.shape}")
            print("\n前5行数据:")
            print(df.head())

    except Exception as e:
        logger.error(f"主程序执行出错: {e}")


if __name__ == "__main__":
    main()
