# ==============================
# Binance API 连接和认证模块
# ==============================
import hashlib
import hmac
import json
import logging
import time
from typing import Dict

import requests

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../binance_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceAPI:
    """Binance API客户端"""

    def __init__(self, api_key: str = None, secret_key: str = None, testnet: bool = True):
        """
        初始化Binance API客户端

        Args:
            api_key: API密钥
            secret_key: 密钥
            testnet: 是否使用测试网环境 (默认True，模拟交易)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet

        # 根据环境选择域名
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"  # 测试网环境
            logger.info("使用Binance测试网环境")
        else:
            self.base_url = "https://fapi.binance.com"  # 实盘环境
            logger.warning("使用Binance实盘环境 - 请谨慎操作！")

        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-MBX-APIKEY': self.api_key,
        })

    def _generate_signature(self, query_string: str) -> str:
        """
        生成签名
        """
        mac = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        )
        return mac.hexdigest()

    def _make_request(self, method: str, endpoint: str, params: Dict = None,
                     signed: bool = False, weight: int = 1) -> Dict:
        """
        发送API请求

        Args:
            method: HTTP方法
            endpoint: API端点
            params: 请求参数
            signed: 是否需要签名
            weight: 请求权重（用于频率限制）
        """
        if signed and not all([self.api_key, self.secret_key]):
            raise ValueError("签名请求需要API密钥和密钥")

        url = f"{self.base_url}{endpoint}"

        # 准备查询参数
        if params is None:
            params = {}

        # 对于签名请求，添加时间戳
        if signed:
            params['timestamp'] = str(int(time.time() * 1000))

            # 创建查询字符串并生成签名
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = self._generate_signature(query_string)
            params['signature'] = signature

        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=10)

            response.raise_for_status()
            result = response.json()

            # 检查API错误响应
            if 'code' in result and result['code'] != 200:
                error_msg = result.get('msg', '未知错误')
                logger.error(f"API错误 {result['code']}: {error_msg}")
                raise Exception(f"API错误 {result['code']}: {error_msg}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            raise

    def get_server_time(self) -> Dict:
        """获取服务器时间"""
        response = requests.get(f"{self.base_url}/fapi/v1/time", timeout=10)
        response.raise_for_status()
        return response.json()

    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = self.get_server_time()
            server_time = response.get('serverTime')
            logger.info(f"API连接成功，服务器时间: {server_time}")
            return True
        except Exception as e:
            logger.error(f"API连接失败: {e}")
            return False

    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return self._make_request('GET', '/fapi/v2/account', signed=True)

    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        return self.get_account_info()

    def get_positions(self) -> Dict:
        """获取持仓信息"""
        return self._make_request('GET', '/fapi/v2/positionRisk', signed=True)

    def get_ticker(self, symbol: str = 'BTCUSDT') -> Dict:
        """获取ticker信息"""
        return self._make_request('GET', '/fapi/v1/ticker/24hr', {'symbol': symbol})

    def get_orderbook(self, symbol: str = 'BTCUSDT', limit: int = 100) -> Dict:
        """获取订单簿"""
        return self._make_request('GET', '/fapi/v1/depth', {'symbol': symbol, 'limit': str(limit)})

    def get_candles(self, symbol: str = 'BTCUSDT', interval: str = '15m',
                    start_time: int = None, end_time: int = None, limit: int = 500) -> Dict:
        """
        获取K线数据

        Args:
            symbol: 交易对 (如 'BTCUSDT')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            start_time: 开始时间戳
            end_time: 结束时间戳
            limit: 返回结果的数量，最大值1500，默认值500
        """
        params = {'symbol': symbol, 'interval': interval, 'limit': str(limit)}
        if start_time:
            params['startTime'] = str(start_time)
        if end_time:
            params['endTime'] = str(end_time)

        return self._make_request('GET', '/fapi/v1/klines', params)

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                    price: float = None, time_in_force: str = 'GTC',
                    reduce_only: bool = False, close_position: bool = False,
                    client_order_id: str = None) -> Dict:
        """
        下单

        Args:
            symbol: 交易对，如 'BTCUSDT'
            side: 订单方向 'BUY' 或 'SELL'
            order_type: 订单类型 'MARKET', 'LIMIT', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
            quantity: 委托数量
            price: 委托价格，仅限价单必填
            time_in_force: 订单有效期 'GTC', 'IOC', 'FOK'
            reduce_only: 是否只减仓
            close_position: 是否平仓
            client_order_id: 客户自定义订单ID
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': f"{quantity:.6f}".rstrip('0').rstrip('.'),
        }

        if price is not None:
            params['price'] = f"{price:.6f}".rstrip('0').rstrip('.')
        if order_type == 'LIMIT':
            params['timeInForce'] = time_in_force
        if reduce_only:
            params['reduceOnly'] = 'true'
        if close_position:
            params['closePosition'] = 'true'
        if client_order_id:
            params['newClientOrderId'] = client_order_id

        response = self._make_request('POST', '/fapi/v1/order', params, signed=True)

        # 记录交易日志
        trade_type = "买入" if side == 'BUY' else "卖出"
        order_type_cn = "市价" if order_type == 'MARKET' else "限价"
        price_str = f" @ {price}" if price else " @ 市价"
        logger.info(f"下单成功: {trade_type} {order_type_cn} {quantity} {symbol}{price_str}")

        return response

    def cancel_order(self, symbol: str, order_id: int = None,
                     orig_client_order_id: str = None) -> Dict:
        """
        撤单

        Args:
            symbol: 交易对
            order_id: 订单ID
            orig_client_order_id: 客户端订单ID
        """
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = str(order_id)
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("必须提供order_id或orig_client_order_id中的一个")

        response = self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
        logger.info(f"撤单成功: 订单ID {order_id or orig_client_order_id}")

        return response

    def cancel_all_orders(self, symbol: str) -> Dict:
        """撤销指定交易对的所有订单"""
        params = {'symbol': symbol}
        response = self._make_request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)
        logger.info(f"撤销{symbol}所有订单成功")
        return response

    def get_order_info(self, symbol: str, order_id: int = None,
                      orig_client_order_id: str = None) -> Dict:
        """
        获取订单信息

        Args:
            symbol: 交易对
            order_id: 订单ID
            orig_client_order_id: 客户端订单ID
        """
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = str(order_id)
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("必须提供order_id或orig_client_order_id中的一个")

        return self._make_request('GET', '/fapi/v1/order', params, signed=True)

    def get_open_orders(self, symbol: str = None) -> Dict:
        """
        获取当前挂单

        Args:
            symbol: 交易对，如果不提供则返回所有交易对的挂单
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)

    def get_order_history(self, symbol: str = None, order_id: int = None,
                         start_time: int = None, end_time: int = None,
                         limit: int = 500) -> Dict:
        """
        获取历史订单记录

        Args:
            symbol: 交易对
            order_id: 订单ID
            start_time: 开始时间戳
            end_time: 结束时间戳
            limit: 返回结果的数量，最大值1000，默认值500
        """
        params = {'limit': str(limit)}
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = str(order_id)
        if start_time:
            params['startTime'] = str(start_time)
        if end_time:
            params['endTime'] = str(end_time)

        return self._make_request('GET', '/fapi/v1/allOrders', params, signed=True)

    def get_fills(self, symbol: str = None, order_id: int = None,
                  start_time: int = None, end_time: int = None,
                  from_id: int = None, limit: int = 500) -> Dict:
        """
        获取成交明细

        Args:
            symbol: 交易对
            order_id: 订单ID
            start_time: 开始时间戳
            end_time: 结束时间戳
            from_id: 从此ID开始查询
            limit: 返回结果的数量，最大值1000，默认值500
        """
        params = {'limit': str(limit)}
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = str(order_id)
        if start_time:
            params['startTime'] = str(start_time)
        if end_time:
            params['endTime'] = str(end_time)
        if from_id:
            params['fromId'] = str(from_id)

        return self._make_request('GET', '/fapi/v1/userTrades', params, signed=True)

    def get_leverage_bracket(self, symbol: str = None) -> Dict:
        """
        获取杠杆分层标准

        Args:
            symbol: 交易对
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/leverageBracket', params, signed=True)

    def change_initial_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        调整杠杆倍数

        Args:
            symbol: 交易对
            leverage: 杠杆倍数，1-125倍
        """
        params = {
            'symbol': symbol,
            'leverage': str(leverage)
        }

        response = self._make_request('POST', '/fapi/v1/leverage', params, signed=True)
        logger.info(f"设置杠杆成功: {symbol} {leverage}倍")

        return response

    def change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        调整保证金模式

        Args:
            symbol: 交易对
            margin_type: 'ISOLATED'(逐仓) 或 'CROSSED'(全仓)
        """
        params = {
            'symbol': symbol,
            'marginType': margin_type
        }

        response = self._make_request('POST', '/fapi/v1/marginType', params, signed=True)
        margin_type_cn = "逐仓" if margin_type == 'ISOLATED' else "全仓"
        logger.info(f"设置保证金模式成功: {symbol} {margin_type_cn}")

        return response

    def get_position_mode(self) -> Dict:
        """获取当前持仓模式"""
        return self._make_request('GET', '/fapi/v1/positionSide/dual', signed=True)

    def change_position_mode(self, dual_side_position: bool) -> Dict:
        """
        调整持仓模式

        Args:
            dual_side_position: true: 双向持仓模式，false: 单向持仓模式
        """
        params = {'dualSidePosition': str(dual_side_position).lower()}

        response = self._make_request('POST', '/fapi/v1/positionSide/dual', params, signed=True)
        mode = "双向持仓" if dual_side_position else "单向持仓"
        logger.info(f"设置持仓模式成功: {mode}")

        return response

    def get_exchange_info(self) -> Dict:
        """获取交易规则和交易对信息"""
        return self._make_request('GET', '/fapi/v1/exchangeInfo')

    def get_mark_price(self, symbol: str = None) -> Dict:
        """
        获取标记价格

        Args:
            symbol: 交易对
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('GET', '/fapi/v1/premiumIndex', params)

    def get_funding_rate(self, symbol: str = None, start_time: int = None,
                        end_time: int = None, limit: int = 100) -> Dict:
        """
        获取资金费率历史

        Args:
            symbol: 交易对
            start_time: 开始时间戳
            end_time: 结束时间戳
            limit: 返回结果的数量，最大值1000，默认值100
        """
        params = {'limit': str(limit)}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = str(start_time)
        if end_time:
            params['endTime'] = str(end_time)

        return self._make_request('GET', '/fapi/v1/fundingRate', params)


def create_binance_client(config_file: str = 'binance_config.json') -> BinanceAPI:
    """
    从配置文件创建Binance客户端

    Args:
        config_file: 配置文件路径
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return BinanceAPI(
            api_key=config.get('api_key'),
            secret_key=config.get('secret_key'),
            testnet=config.get('testnet', True)
        )
    except FileNotFoundError:
        logger.error(f"配置文件 {config_file} 不存在")
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件 {config_file} 格式错误")
        raise


if __name__ == "__main__":
    # 测试API连接
    try:
        client = create_binance_client()
        if client.test_connection():
            print("✅ Binance API连接成功！")

            # 获取BTC价格信息
            ticker = client.get_ticker()
            if 'price' in ticker:
                price = ticker['price']
                print(f"📈 BTC当前价格: ${price}")

            # 获取账户信息
            account = client.get_account_info()
            print(f"💰 账户信息: {account}")

    except Exception as e:
        print(f"❌ 连接失败: {e}")