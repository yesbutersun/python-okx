# ==============================
# 优化策略模块 - 高性能量化交易策略
# ==============================
import pandas as pd
import numpy as np


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(prices, period):
    """计算EMA指标"""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_atr(high, low, close, period=14):
    """计算ATR指标"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """计算布林带"""
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算随机指标"""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    return k_percent, d_percent


def prepare_dataframe(df):
    """准备DataFrame"""
    df = df.copy()
    # 标准化列名
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # 确保数值类型
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


def adaptive_mean_reversion(df, fast_period=10, slow_period=30, vol_threshold=0.02):
    """
    自适应均值回归策略
    根据市场波动性动态调整参数
    """
    df = prepare_dataframe(df)

    # 计算波动率
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()

    # 根据波动率选择周期
    df['adaptive_period'] = np.where(df['volatility'] > vol_threshold, fast_period, slow_period)

    # 计算自适应布林带
    df['sma'] = df['Close'].rolling(window=30, min_periods=1).mean()
    df['std'] = df['Close'].rolling(window=30, min_periods=1).std()
    df['upper_band'] = df['sma'] + (df['std'] * 1.5)  # 降低阈值
    df['lower_band'] = df['sma'] - (df['std'] * 1.5)

    # RSI确认
    df['rsi'] = calculate_rsi(df['Close'], 14)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    entry_price = 0

    for i in range(1, len(df)):
        if pd.isna(df['upper_band'].iloc[i]) or pd.isna(df['rsi'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        rsi = df['rsi'].iloc[i]

        # 多重确认条件
        oversold = rsi < 30
        overbought = rsi > 70

        # 开仓条件
        if price < lower and oversold and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price
        elif price > upper and overbought and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            entry_price = price

        # 止盈止损（动态）
        if position == 1:
            atr = calculate_atr(df['High'][:i+1], df['Low'][:i+1], df['Close'][:i+1], 14).iloc[-1]
            if pd.notna(atr):
                tp_price = entry_price + (2 * atr)
                sl_price = entry_price - (1 * atr)
                if price >= tp_price or price <= sl_price or rsi > 70:
                    signals.at[df.index[i], 'long_exit'] = True
                    position = 0

        elif position == -1:
            atr = calculate_atr(df['High'][:i+1], df['Low'][:i+1], df['Close'][:i+1], 14).iloc[-1]
            if pd.notna(atr):
                tp_price = entry_price - (2 * atr)
                sl_price = entry_price + (1 * atr)
                if price <= tp_price or price >= sl_price or rsi < 30:
                    signals.at[df.index[i], 'short_exit'] = True
                    position = 0

    return signals


def momentum_trend_following(df, fast_ema=8, slow_ema=21, rsi_period=14, volume_threshold=1.2):
    """
    动量趋势跟随策略
    结合趋势、动量和成交量确认
    """
    df = prepare_dataframe(df)

    # 趋势指标
    df['ema_fast'] = calculate_ema(df['Close'], fast_ema)
    df['ema_slow'] = calculate_ema(df['Close'], slow_ema)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])

    # 动量指标
    df['rsi'] = calculate_rsi(df['Close'], rsi_period)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['High'], df['Low'], df['Close'])

    # 成交量确认
    df['volume_sma'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(2, len(df)):
        # 检查数据完整性
        if any(pd.isna(df[col].iloc[i]) for col in ['ema_fast', 'ema_slow', 'rsi', 'macd', 'volume_ratio']):
            continue

        # 趋势确认
        bullish_trend = df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]
        bearish_trend = df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]

        # 动量确认
        momentum_up = df['macd'].iloc[i] > df['macd_signal'].iloc[i]
        momentum_down = df['macd'].iloc[i] < df['macd_signal'].iloc[i]

        # 成交量确认
        volume_confirm = df['volume_ratio'].iloc[i] > volume_threshold

        # RSI过滤（避免极端超买超卖）
        rsi_filter = 20 < df['rsi'].iloc[i] < 80

        # 多重确认的开仓条件
        if bullish_trend and momentum_up and volume_confirm and rsi_filter and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
        elif bearish_trend and momentum_down and volume_confirm and rsi_filter and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1

        # 平仓条件
        if position == 1:
            # 趋势反转或RSI极度超买
            if df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] or df['rsi'].iloc[i] > 80:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            # 趋势反转或RSI极度超卖
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] or df['rsi'].iloc[i] < 20:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def dual_rsi_strategy(df, fast_rsi=7, slow_rsi=21, oversold=30, overbought=70):
    """
    双RSI策略
    使用不同周期的RSI进行确认
    """
    df = prepare_dataframe(df)

    # 计算双RSI
    df['rsi_fast'] = calculate_rsi(df['Close'], fast_rsi)
    df['rsi_slow'] = calculate_rsi(df['Close'], slow_rsi)

    # 计算RSI的移动平均线（平滑）
    df['rsi_fast_ma'] = df['rsi_fast'].rolling(window=3, min_periods=1).mean()
    df['rsi_slow_ma'] = df['rsi_slow'].rolling(window=3, min_periods=1).mean()

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(5, len(df)):
        if any(pd.isna(df[col].iloc[i]) for col in ['rsi_fast', 'rsi_slow']):
            continue

        # 双RSI同时超卖
        if (df['rsi_fast'].iloc[i] < oversold and
            df['rsi_slow'].iloc[i] < oversold and
            df['rsi_fast'].iloc[i] > df['rsi_fast'].iloc[i-1] and
            position == 0):
            signals.at[df.index[i], 'long_entry'] = True
            position = 1

        # 双RSI同时超买
        elif (df['rsi_fast'].iloc[i] > overbought and
              df['rsi_slow'].iloc[i] > overbought and
              df['rsi_fast'].iloc[i] < df['rsi_fast'].iloc[i-1] and
              position == 0):
            signals.at[df.index[i], 'short_entry'] = True
            position = -1

        # 平仓条件
        if position == 1:
            # 快速RSI超买或慢速RSI中性
            if df['rsi_fast'].iloc[i] > overbought or 45 < df['rsi_slow'].iloc[i] < 55:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            # 快速RSI超卖或慢速RSI中性
            if df['rsi_fast'].iloc[i] < oversold or 45 < df['rsi_slow'].iloc[i] < 55:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def price_action_breakout(df, lookback=20, breakout_threshold=0.002, volume_confirm=True):
    """
    价格行为突破策略
    基于支撑阻力位和成交量确认
    """
    df = prepare_dataframe(df)

    # 计算关键价格水平
    df['high_lookback'] = df['High'].rolling(window=lookback, min_periods=1).max()
    df['low_lookback'] = df['Low'].rolling(window=lookback, min_periods=1).min()
    df['close_lookback'] = df['Close'].shift(1).rolling(window=lookback, min_periods=1).max()

    # 成交量指标
    df['volume_ema'] = df['Volume'].ewm(span=20).mean()
    df['volume_spike'] = df['Volume'] > df['volume_ema'] * 1.5

    # 波动率过滤
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    df['breakout_threshold'] = df['atr'] * 0.5

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    entry_price = 0

    for i in range(lookback, len(df)):
        if pd.isna(df['high_lookback'].iloc[i]) or pd.isna(df['atr'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        high_resistance = df['high_lookback'].iloc[i-1]
        low_support = df['low_lookback'].iloc[i-1]

        # 向上突破
        if price > high_resistance + df['breakout_threshold'].iloc[i]:
            if not volume_confirm or df['volume_spike'].iloc[i]:
                signals.at[df.index[i], 'long_entry'] = True
                position = 1
                entry_price = price

        # 向下突破
        elif price < low_support - df['breakout_threshold'].iloc[i]:
            if not volume_confirm or df['volume_spike'].iloc[i]:
                signals.at[df.index[i], 'short_entry'] = True
                position = -1
                entry_price = price

        # 止损策略
        if position == 1:
            # 回测最近低点下方
            recent_low = df['Low'].iloc[i-5:i].min()
            if price < recent_low:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            # 回测最近高点上方
            recent_high = df['High'].iloc[i-5:i].max()
            if price > recent_high:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def volatility_breakout_strategy(df, bb_period=10, bb_std=1.5, keltner_period=20, keltner_mult=1.5):
    """
    波动率突破策略（挤压突破）
    布林带与肯特纳通道挤压后的突破
    """
    df = prepare_dataframe(df)

    # 布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
        df['Close'], bb_period, bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # 肯特纳通道
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], keltner_period)
    df['kc_middle'] = df['Close'].rolling(window=keltner_period, min_periods=1).mean()
    df['kc_upper'] = df['kc_middle'] + (df['atr'] * keltner_mult)
    df['kc_lower'] = df['kc_middle'] - (df['atr'] * keltner_mult)
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']

    # 挤压指标
    df['squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=50, min_periods=1).quantile(0.2)

    # 动量确认
    df['momentum'] = df['Close'] / df['Close'].shift(5) - 1

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(keltner_period + 5, len(df)):
        if any(pd.isna(df[col].iloc[i]) for col in ['bb_upper', 'kc_upper', 'momentum']):
            continue

        price = df['Close'].iloc[i]

        # 挤压后突破
        if df['squeeze'].iloc[i-1]:  # 前一刻在挤压状态
            # 向上突破布林带上轨
            if price > df['bb_upper'].iloc[i] and df['momentum'].iloc[i] > 0.005:
                if position == 0:
                    signals.at[df.index[i], 'long_entry'] = True
                    position = 1

            # 向下突破布林带下轨
            elif price < df['bb_lower'].iloc[i] and df['momentum'].iloc[i] < -0.005:
                if position == 0:
                    signals.at[df.index[i], 'short_entry'] = True
                    position = -1

        # 平仓条件
        if position == 1:
            # 回到布林带中轨或反向突破
            if price < df['bb_middle'].iloc[i] or price < df['bb_lower'].iloc[i]:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            # 回到布林带中轨或反向突破
            if price > df['bb_middle'].iloc[i] or price > df['bb_upper'].iloc[i]:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def smart_money_concepts(df, lookback=20, volume_weight=True):
    """
    智能资金概念策略
    基于机构行为的价格模式识别
    """
    df = prepare_dataframe(df)

    # 计算成交量加权价格
    if volume_weight:
        df['vwap'] = (df['Close'] * df['Volume']).rolling(window=20, min_periods=1).sum() / df['Volume'].rolling(window=20, min_periods=1).sum()
    else:
        df['vwap'] = df['Close'].rolling(window=20, min_periods=1).mean()

    # 计算供需区域
    df['high_volume'] = df['Volume'] > df['Volume'].rolling(window=50, min_periods=1).quantile(0.8)
    df['support'] = df['Low'].rolling(window=lookback, min_periods=1).min()
    df['resistance'] = df['High'].rolling(window=lookback, min_periods=1).max()

    # 计算价格动量
    df['price_change'] = df['Close'].pct_change()
    df['momentum'] = df['price_change'].rolling(window=10, min_periods=1).sum()

    # 识别smart money行为
    df['volume_price_divergence'] = ((df['Volume'] > df['Volume'].shift(1)) &
                                     (abs(df['price_change']) < 0.001))

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(lookback + 1, len(df)):
        if any(pd.isna(df[col].iloc[i]) for col in ['vwap', 'support', 'resistance', 'momentum']):
            continue

        price = df['Close'].iloc[i]
        vwap = df['vwap'].iloc[i]
        support = df['support'].iloc[i-1]
        resistance = df['resistance'].iloc[i-1]

        # 在支撑位附近的高量买入
        if (price <= support * 1.002 and
            price > vwap and
            df['momentum'].iloc[i] > 0 and
            position == 0):
            signals.at[df.index[i], 'long_entry'] = True
            position = 1

        # 在阻力位附近的高量卖出
        elif (price >= resistance * 0.998 and
              price < vwap and
              df['momentum'].iloc[i] < 0 and
              position == 0):
            signals.at[df.index[i], 'short_entry'] = True
            position = -1

        # 平仓条件
        if position == 1:
            # 到达阻力位或跌破VWAP
            if price >= resistance or price < vwap:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            # 到达支撑位或涨破VWAP
            if price <= support or price > vwap:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


# 策略字典
OPTIMIZED_STRATEGIES = {
    '自适应均值回归': adaptive_mean_reversion,
    '动量趋势跟随': momentum_trend_following,
    '双RSI策略': dual_rsi_strategy,
    '价格行为突破': price_action_breakout,
    '波动率挤压突破': volatility_breakout_strategy,
    '智能资金概念': smart_money_concepts,
}


def get_optimized_strategy_list():
    """获取所有优化策略名称列表"""
    return list(OPTIMIZED_STRATEGIES.keys())