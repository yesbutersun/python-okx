#!/usr/bin/env python3
"""
OKX合约规格配置
用于处理不同合约的交易规格要求
"""

# OKX BTC永续合约规格
from decimal import Decimal, ROUND_DOWN, getcontext

getcontext().prec = 18
OKX_CONTRACT_SPECS = {
    'BTC-USDT-SWAP': {
        # sz单位：张数，BTC USDT本位永续每张面值0.01 BTC
        'min_lot_size': 1,          # 最小下单张数
        'lot_size_multiple': 1,     # 张数步长
        'max_leverage': 125,         # 最大杠杆倍数
        'tick_size': 0.1,            # 最小价格变动单位
        'contract_value': 0.01,       # 合约面值（每张合约对应的BTC数量）
        'settlement_currency': 'USDT', # 结算货币
        'quote_currency': 'USDT',      # 计价货币
        'base_currency': 'BTC',        # 基础货币
    },

    'ETH-USDT-SWAP': {
        # sz单位：张数，ETH USDT本位永续每张面值0.1 ETH
        'min_lot_size': 1,
        'lot_size_multiple': 1,
        'max_leverage': 100,
        'tick_size': 0.01,
        'contract_value': 0.1,
        'settlement_currency': 'USDT',
        'quote_currency': 'USDT',
        'base_currency': 'ETH',
    },

    # 可以继续添加其他合约规格
    'DOGE-USDT-SWAP': {
        # sz单位：张数，面值1 DOGE
        'min_lot_size': 10,
        'lot_size_multiple': 1,
        'max_leverage': 75,
        'tick_size': 0.0001,
        'contract_value': 1,
        'settlement_currency': 'USDT',
        'quote_currency': 'USDT',
        'base_currency': 'DOGE',
    }
}

def get_contract_spec(inst_id):
    """获取指定合约的规格信息"""
    return OKX_CONTRACT_SPECS.get(inst_id.upper(), None)

def validate_order_size(inst_id, size, current_price=0, target_usdt=0):
    """验证并调整订单数量以符合合约规格"""

    spec = get_contract_spec(inst_id)
    if not spec:
        raise ValueError(f"未找到合约 {inst_id} 的规格信息")

    min_size = Decimal(str(spec['min_lot_size']))
    size_multiple = Decimal(str(spec['lot_size_multiple']))
    contract_value = Decimal(str(spec.get('contract_value', 1)))

    logger_info = []
    size_decimal = Decimal(str(size))

    # 1. 检查最小数量
    if size_decimal < min_size:
        adjusted_size = min_size
        logger_info.append(f"数量过小: {size_decimal:.6f} → 最小单位 {min_size:.6f}")
    else:
        # 2. 调整为lot size的倍数
        multiples = (size_decimal / size_multiple).to_integral_value(rounding=ROUND_DOWN)
        adjusted_size = multiples * size_multiple

        if adjusted_size == 0:
            adjusted_size = min_size
            logger_info.append(f"调整后为0，使用最小单位: {min_size:.6f}")

        if adjusted_size != size_decimal:
            logger_info.append(f"调整为倍数: {size_decimal:.6f} → {adjusted_size:.6f} (倍数: {size_multiple})")

    # 3. 计算实际USDT价值
    if current_price > 0:
        actual_usdt_value = float(adjusted_size * contract_value) * current_price
        if target_usdt > 0:
            deviation = abs(actual_usdt_value - target_usdt)
            deviation_pct = deviation / target_usdt * 100
            logger_info.append(f"目标价值: {target_usdt:.2f} USDT")
            logger_info.append(f"实际价值: {actual_usdt_value:.2f} USDT (偏差: {deviation_pct:.2f}%)")
        else:
            logger_info.append(f"交易价值: {actual_usdt_value:.2f} USDT")

    return float(adjusted_size), logger_info

def print_contract_spec(inst_id):
    """打印合约规格信息"""
    spec = get_contract_spec(inst_id)
    if not spec:
        print(f"❌ 未找到合约 {inst_id} 的规格信息")
        return

    print(f"\n📋 {inst_id} 合约规格:")
    print(f"   - 最小交易单位: {spec['min_lot_size']}")
    print(f"   - 数量倍数: {spec['lot_size_multiple']}")
    print(f"   - 最大杠杆: {spec['max_leverage']}x")
    print(f"   - 价格精度: {spec['tick_size']}")
    print(f"   - 合约面值: {spec['contract_value']} {spec['base_currency']}")
    print(f"   - 结算货币: {spec['settlement_currency']}")
    print(f"   - 基础货币: {spec['base_currency']}")
    print(f"   - 计价货币: {spec['quote_currency']}")

def validate_leverage(inst_id, leverage):
    """验证杠杆倍数是否符合合约要求"""
    spec = get_contract_spec(inst_id)
    if not spec:
        return False, "未找到合约规格信息"

    max_leverage = spec['max_leverage']

    if leverage > max_leverage:
        return False, f"杠杆 {leverage}x 超过最大值 {max_leverage}x"

    if leverage <= 0:
        return False, f"杠杆 {leverage}x 无效"

    return True, f"杠杆 {leverage}x 有效"

# 使用示例和测试
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 测试BTC合约规格
    symbol = 'BTC-USDT-SWAP'
    print_contract_spec(symbol)

    # 测试订单数量验证
    test_cases = [
        {'size': 0.0005, 'price': 90000, 'target_usdt': 100},
        {'size': 0.0011, 'price': 90000, 'target_usdt': 100},
        {'size': 0.0025, 'price': 90000, 'target_usdt': 100},
        {'size': 0.01, 'price': 90000, 'target_usdt': 100},
    ]

    print(f"\n🧪 订单数量验证测试:")
    for i, test_case in enumerate(test_cases, 1):
        size = test_case['size']
        price = test_case['price']
        target_usdt = test_case['target_usdt']

        print(f"\n测试 {i}:")
        print(f"   原数量: {size:.6f} {symbol}")
        print(f"   当前价格: ${price:.2f}")
        print(f"   目标价值: ${target_usdt:.2f}")

        try:
            adjusted_size, logs = validate_order_size(symbol, size, price, target_usdt)
            print(f"   调整后数量: {adjusted_size:.6f}")
            for log in logs:
                print(f"   ⚠️  {log}")

        except Exception as e:
            print(f"   ❌ 验证失败: {e}")

    # 测试杠杆验证
    print(f"\n🔧 杠杆验证测试:")
    leverage_tests = [1, 5, 10, 50, 100, 130]

    for leverage in leverage_tests:
        is_valid, message = validate_leverage(symbol, leverage)
        status = "✅" if is_valid else "❌"
        print(f"   {status} 杠杆 {leverage}x: {message}")
