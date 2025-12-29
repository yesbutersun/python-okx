# ==============================
# OKX API 连接和认证模块
# ==============================
import base64
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
    logging.FileHandler('/okx_trading.log', encoding='utf-8'),
    logging.StreamHandler()
  ]
)
logger = logging.getLogger(__name__)


class OKXAPI:
  """OKX API客户端"""

  def __init__(self, api_key: str = None, secret_key: str = None, passphrase: str = None, sandbox: bool = True):
    """
    初始化OKX API客户端

    Args:
        api_key: API密钥
        secret_key: 密钥
        passphrase: 口令
        sandbox: 是否使用沙盒环境 (默认True，模拟交易)
    """
    self.api_key = api_key
    self.secret_key = secret_key
    self.passphrase = passphrase
    self.sandbox = sandbox

    # 根据环境选择域名
    if sandbox:
      # OKX沙盒环境使用相同的域名，但需要特殊的API密钥和头部
      self.base_url = "https://www.okx.com"
      logger.info("使用OKX沙盒环境")
    else:
      self.base_url = "https://www.okx.com"  # 实盘环境
      logger.warning("使用OKX实盘环境 - 请谨慎操作！")

    self.session = requests.Session()
    self.session.headers.update({
      'Content-Type': 'application/json',
      'OK-ACCESS-KEY': self.api_key,
      'OK-ACCESS-PASSPHRASE': self.passphrase,
      'OK-ACCESS-SIGN': '',
      'OK-ACCESS-TIMESTAMP': '',
    })

  def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
    """
    生成签名
    """
    message = timestamp + method + request_path + body
    mac = hmac.new(
      self.secret_key.encode('utf-8'),
      message.encode('utf-8'),
      hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode('utf-8')

  def _generate_signature_new(self, timestamp, method, request_path, body):
        print(f"timestamp={timestamp}, method={method}, request_path={request_path}, body={body}")
        if not body:
            body = ""
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        mac = hmac.new(
            bytes(self.secret_key, "utf-8"),
            bytes(message, "utf-8"),
            digestmod="sha256",
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

  def _make_request(self, method: str, endpoint: str, params: Dict = None, body: Dict = None) -> Dict:
    """
    发送API请求
    """
    if not all([self.api_key, self.secret_key, self.passphrase]):
      raise ValueError("API密钥、密钥和口令不能为空")

    # 使用ISO 8601格式的时间戳
    timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    request_path = endpoint

    # 生成签名
    if method in ['GET', 'DELETE']:
      query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())]) if params else ''
      request_path += f"?{query_string}" if query_string else ''
      # GET请求也需要包含空的body参数进行签名
      signature = self._generate_signature_new(timestamp, method, request_path, '')
    else:  # POST
      body_str = json.dumps(params or {}) if params else '{}'
      signature = self._generate_signature_new(timestamp, method, request_path, body_str)

    # 设置请求头
    headers = {
      'OK-ACCESS-KEY': self.api_key,
      'OK-ACCESS-SIGN': signature,
      'OK-ACCESS-TIMESTAMP': timestamp,
      'OK-ACCESS-PASSPHRASE': self.passphrase,
      'Content-Type': 'application/json'
    }

    # 如果是沙盒环境，添加模拟交易标志
    if self.sandbox:
      headers['x-simulated-trading'] = '1'

    url = f"{self.base_url}{request_path}"

    try:
      if method == 'GET':
        response = requests.get(url, headers=headers, params=params, timeout=10)
      elif method == 'POST':
        response = requests.post(url, headers=headers, json=params, timeout=10)
      elif method == 'DELETE':
        response = requests.delete(url, headers=headers, params=params, timeout=10)

      response.raise_for_status()
      return response.json()

    except requests.exceptions.RequestException as e:
      logger.error(f"API请求失败: {e}")
      raise

  def get_server_time(self) -> Dict:
    """获取服务器时间"""
    return self._make_request('GET', '/api/v5/public/time')

  def test_connection(self) -> bool:
    """测试API连接"""
    try:
      response = self.get_server_time()
      logger.info(f"API连接成功，服务器时间: {response.get('data', [{}])[0].get('ts')}")
      return True
    except Exception as e:
      logger.error(f"API连接失败: {e}")
      return False

  def get_account_balance(self) -> Dict:
    """获取账户余额"""
    return self._make_request('GET', '/api/v5/account/balance')

  def get_positions(self) -> Dict:
    """获取持仓信息"""
    return self._make_request('GET', '/api/v5/account/positions')

  def get_ticker(self, inst_id: str = 'BTC-USDT-SWAP') -> Dict:
    """获取ticker信息"""
    return self._make_request('GET', '/api/v5/market/ticker', {'instId': inst_id})

  def get_orderbook(self, inst_id: str = 'BTC-USDT-SWAP', sz: int = 100) -> Dict:
    """获取订单簿"""
    return self._make_request('GET', '/api/v5/market/books', {'instId': inst_id, 'sz': str(sz)})

  def get_candles(self, inst_id: str = 'BTC-USDT-SWAP', bar: str = '15m',
                  before: str = None, after: str = None, limit: int = 100) -> Dict:
    """
    获取K线数据（公共API，无需认证）

    Args:
        inst_id: 产品ID (如 'BTC-USDT-SWAP')
        bar: K线周期 (如 '1m', '5m', '15m', '1H', '4H', '1D')
        before: 请求此时间戳之前的分页内容
        after: 请求此时间戳之后的分页内容
        limit: 返回结果的数量，最大值100，默认值100
    """
    try:
      # 公共API不需要认证
      url = f"{self.base_url}/api/v5/market/candles"
      params = {'instId': inst_id, 'bar': bar, 'limit': str(limit)}
      if before:
        params['before'] = before
      if after:
        params['after'] = after

      response = self.session.get(url, params=params, timeout=10)
      response.raise_for_status()
      return response.json()

    except requests.exceptions.RequestException as e:
      logger.error(f"K线数据获取失败: {e}")
      return {'error': str(e)}

  def place_order(self, inst_id: str, td_mode: str, side: str, ord_type: str, sz: str,
                  px: str = None, reduce_only: bool = False, cl_ord_id: str = None) -> Dict:
    """
    下单

    Args:
        inst_id: 产品ID，如 'BTC-USDT-SWAP'
        td_mode: 交易模式 'isolated'(逐仓) 或 'cross'(全仓)
        side: 订单方向 'buy' 或 'sell'
        ord_type: 订单类型 'market'(市价) 或 'limit'(限价)
        sz: 委托数量
        px: 委托价格，仅限价单必填
        reduce_only: 是否只减仓，'true'或'false'，默认'false'
        cl_ord_id: 客户自定义订单ID
    """
    params = {
      'instId': inst_id,
      'tdMode': td_mode,
      'side': side,
      'ordType': ord_type,
      'sz': sz,
    }

    if px is not None:
      params['px'] = px
    if reduce_only:
      params['reduceOnly'] = 'true'
    if cl_ord_id:
      params['clOrdId'] = cl_ord_id

    response = self._make_request('POST', '/api/v5/trade/order', params)

    # 记录交易日志
    trade_type = "买入" if side == 'buy' else "卖出"
    order_type = "市价" if ord_type == 'market' else "限价"
    logger.info(f"下单成功: {trade_type} {order_type} {sz} {inst_id} @ {px or '市价'}")

    return response

  def cancel_order(self, inst_id: str, ord_id: str) -> Dict:
    """
    撤单

    Args:
        inst_id: 产品ID
        ord_id: 订单ID
    """
    params = {'instId': inst_id, 'ordId': ord_id}

    response = self._make_request('POST', '/api/v5/trade/cancel-order', params)
    logger.info(f"撤单成功: 订单ID {ord_id}")

    return response

  def get_order_info(self, inst_id: str, ord_id: str) -> Dict:
    """
    获取订单信息

    Args:
        inst_id: 产品ID
        ord_id: 订单ID
    """
    params = {'instId': inst_id, 'ordId': ord_id}
    return self._make_request('GET', '/api/v5/trade/order', params)

  def get_order_history(self, inst_id: str = 'BTC-USDT-SWAP', ord_type: str = '',
                        state: str = '', category: str = '', after: str = '',
                        before: str = '', limit: str = '100') -> Dict:
    """
    获取历史订单记录

    Args:
        inst_id: 产品ID
        ord_type: 订单类型 'limit'/'market'/'post_only'/'fok'/'ioc'
        state: 订单状态 'live'/'partially_filled'/'filled'/'canceled'
        category: 订单种类 'simple'/'twice'/'oco'/'conditional'
        after: 请求此时间戳之后的分页内容
        before: 请求此时间戳之前的分页内容
        limit: 返回结果的数量，最大值100，默认值100
    """
    params = {'instId': inst_id, 'limit': limit}
    if ord_type:
      params['ordType'] = ord_type
    if state:
      params['state'] = state
    if category:
      params['category'] = category
    if after:
      params['after'] = after
    if before:
      params['before'] = before

    return self._make_request('GET', '/api/v5/trade/orders-history-archive', params)

  def get_fills(self, inst_id: str = 'BTC-USDT-SWAP', ord_id: str = '',
                after: str = '', before: str = '', limit: str = '100') -> Dict:
    """
    获取成交明细

    Args:
        inst_id: 产品ID
        ord_id: 订单ID，如果指定，则返回该订单的成交明细
        after: 请求此时间戳之后的分页内容
        before: 请求此时间戳之前的分页内容
        limit: 返回结果的数量，最大值100，默认值100
    """
    params = {'instId': inst_id, 'limit': limit}
    if ord_id:
      params['ordId'] = ord_id
    if after:
      params['after'] = after
    if before:
      params['before'] = before

    return self._make_request('GET', '/api/v5/trade/fills-history', params)

  def get_leverage_info(self, inst_id: str = 'BTC-USDT-SWAP', mgnMode: str = 'isolated') -> Dict:
    """
    获取杠杆倍数信息

    Args:
        inst_id: 产品ID
        mgnMode: 保证金模式 'isolated'(逐仓) 或 'cross'(全仓)
    """
    params = {'instId': inst_id, 'mgnMode': mgnMode}
    return self._make_request('GET', '/api/v5/account/max-avail-size', params)

  def set_leverage(self, inst_id: str = 'BTC-USDT-SWAP', lever: str = '5', mgnMode: str = 'isolated',
                   pos_side: str = 'long', ccy: str = 'USDT', posId: str = '') -> Dict:
    """
    设置杠杆倍数

    Args:
        inst_id: 产品ID
        lever: 杠杆倍数，1-125倍
        mgnMode: 保证金模式 'isolated'(逐仓) 或 'cross'(全仓)
        posSide: 持仓方向 'long'/'short'/'net'
        ccy: 保证金币种，仅适用于单向持仓模式下的全仓杠杆币种调整
        posId: 持仓ID，仅适用于全仓持仓模式下的杠杆倍数调整
    """
    params = {
      'instId': inst_id,
      'lever': lever,
      'mgnMode': mgnMode,
      'posSide': pos_side,
    }
    if ccy:
      params['ccy'] = ccy
    if posId:
      params['posId'] = posId

    response = self._make_request('POST', '/api/v5/account/set-leverage', params)
    logger.info(f"设置杠杆成功: {inst_id} {lever}倍")

    return response


def create_okx_client(sandbox: bool = True) -> OKXAPI:
  """
  创建OKX客户端

  Args:
      sandbox: 是否使用沙盒环境
          - True: 使用 okx_simulation_config.json
          - False: 使用 okx_config.json
  """
  try:
    # 根据sandbox参数选择配置文件
    if sandbox:
      config_file = 'config/okx_simulation_config.json'
      logger.info("使用沙盒配置文件: okx_simulation_config.json")
    else:
      config_file = 'config/okx_config.json'
      logger.info("使用实盘配置文件: okx_config.json")

    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
      config = json.load(f)

    # 获取API配置参数
    api_key = config.get('api_key')
    secret_key = config.get('secret_key')
    passphrase = config.get('passphrase')

    # 验证必要参数
    if not all([api_key, secret_key, passphrase]):
      raise ValueError(f"配置文件 {config_file} 缺少必要的API参数")

    logger.info(f"创建OKX客户端: {config_file}, 沙盒模式: {sandbox}")

    return OKXAPI(
      api_key=api_key,
      secret_key=secret_key,
      passphrase=passphrase,
      sandbox=sandbox  # 直接使用传入的sandbox参数
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
    client = create_okx_client()
    if client.test_connection():
      print("✅ OKX API连接成功！")

      # 获取BTC价格信息
      ticker = client.get_ticker()
      if 'data' in ticker and ticker['data']:
        price = ticker['data'][0]['last']
        print(f"📈 BTC当前价格: ${price}")

      # 获取账户余额
      balance = client.get_account_balance()
      print(f"💰 账户余额: {balance}")

  except Exception as e:
    print(f"❌ 连接失败: {e}")
