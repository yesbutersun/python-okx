# ==============================
# 优化策略模块 - 简化版高性能策略
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


def improved_mean_reversion(df, lookback=15, std_dev=1.5):
    """
    改进的均值回归策略
    - 降低回看周期
    - 降低标准差倍数
    - 添加RSI过滤
    """
    df = prepare_dataframe(df)

    # 计算布林带
    df['sma'] = df['Close'].rolling(window=lookback, min_periods=1).mean()
    df['std'] = df['Close'].rolling(window=lookback, min_periods=1).std()
    df['upper_band'] = df['sma'] + (df['std'] * std_dev)
    df['lower_band'] = df['sma'] - (df['std'] * std_dev)

    # RSI确认
    df['rsi'] = calculate_rsi(df['Close'], 14)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    entry_price = 0
    consecutive_losses = 0

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        rsi = df['rsi'].iloc[i]

        # 确保数据有效
        if pd.isna(upper) or pd.isna(lower) or pd.isna(rsi):
            continue

        # 多重过滤的入场条件
        # 多头：价格低于下轨 AND RSI超卖
        if price < lower and rsi < 35 and position == 0 and consecutive_losses < 3:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price

        # 空头：价格高于上轨 AND RSI超买
        elif price > upper and rsi > 65 and position == 0 and consecutive_losses < 3:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            entry_price = price

        # 出场条件
        if position == 1:
            # 止盈或止损
            atr = calculate_atr(df['High'][:i+1], df['Low'][:i+1], df['Close'][:i+1], 14).iloc[-1]
            if pd.notna(atr):
                tp_price = entry_price + (1.5 * atr)
                sl_price = entry_price - (1 * atr)

                if price >= tp_price or price <= sl_price or rsi > 70:
                    signals.at[df.index[i], 'long_exit'] = True
                    position = 0
                    # 记录盈亏（简化版）
                    if price < entry_price * 0.98:  # 2%以上亏损
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

        elif position == -1:
            # 止盈或止损
            atr = calculate_atr(df['High'][:i+1], df['Low'][:i+1], df['Close'][:i+1], 14).iloc[-1]
            if pd.notna(atr):
                tp_price = entry_price - (1.5 * atr)
                sl_price = entry_price + (1 * atr)

                if price <= tp_price or price >= sl_price or rsi < 30:
                    signals.at[df.index[i], 'short_exit'] = True
                    position = 0
                    # 记录盈亏
                    if price > entry_price * 1.02:  # 2%以上亏损
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

    return signals


def enhanced_rsi_strategy(df, fast_rsi=7, slow_rsi=21):
    """
    增强RSI策略
    结合双RSI和动态阈值
    """
    df = prepare_dataframe(df)

    # 计算双RSI
    df['rsi_fast'] = calculate_rsi(df['Close'], fast_rsi)
    df['rsi_slow'] = calculate_rsi(df['Close'], slow_rsi)

    # 计算RSI的EMA
    df['rsi_fast_ema'] = df['rsi_fast'].ewm(span=3).mean()
    df['rsi_slow_ema'] = df['rsi_slow'].ewm(span=3).mean()

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    position_timer = 0

    for i in range(30, len(df)):
        # 检查数据
        if any(pd.isna(df[col].iloc[i]) for col in ['rsi_fast', 'rsi_slow', 'rsi_fast_ema']):
            continue

        fast_rsi = df['rsi_fast'].iloc[i]
        slow_rsi = df['rsi_slow'].iloc[i]
        fast_ema = df['rsi_fast_ema'].iloc[i]

        # 入场条件
        # 多头：快RSI超卖且上升
        if (fast_rsi < 25 and fast_rsi > fast_ema and
            slow_rsi < 40 and position == 0):
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            position_timer = 0

        # 空头：快RSI超买且下降
        elif (fast_rsi > 75 and fast_rsi < fast_ema and
              slow_rsi > 60 and position == 0):
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            position_timer = 0

        # 出场条件
        position_timer += 1

        if position == 1:
            # RSI过热或持有时间过长
            if fast_rsi > 70 or position_timer > 20:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0

        elif position == -1:
            # RSI过冷或持有时间过长
            if fast_rsi < 30 or position_timer > 20:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def trend_momentum_strategy(df, ema_fast=10, ema_slow=25, momentum_period=10):
    """
    趋势动量策略
    结合EMA交叉和动量确认
    """
    df = prepare_dataframe(df)

    # EMA指标
    df['ema_fast'] = calculate_ema(df['Close'], ema_fast)
    df['ema_slow'] = calculate_ema(df['Close'], ema_slow)

    # 动量指标
    df['momentum'] = df['Close'] / df['Close'].shift(momentum_period) - 1

    # 成交量
    df['volume_sma'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    bars_in_position = 0

    for i in range(ema_slow + momentum_period, len(df)):
        # 检查数据
        if any(pd.isna(df[col].iloc[i]) for col in ['ema_fast', 'ema_slow', 'momentum']):
            continue

        # 趋势条件
        golden_cross = df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]
        death_cross = df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]

        # 动量条件
        momentum_up = df['momentum'].iloc[i] > 0.01  # 1%以上动量
        momentum_down = df['momentum'].iloc[i] < -0.01

        # 成交量确认
        volume_confirm = df['volume_ratio'].iloc[i] > 1.2

        bars_in_position += 1

        # 入场条件
        if golden_cross and momentum_up and volume_confirm and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            bars_in_position = 0

        elif death_cross and momentum_down and volume_confirm and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            bars_in_position = 0

        # 出场条件
        if position == 1:
            # 趋势反转或时间止损
            if df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] or bars_in_position > 50:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0

        elif position == -1:
            # 趋势反转或时间止损
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] or bars_in_position > 50:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def breakout_strategy(df, lookback=20, volume_threshold=1.5):
    """
    改进的突破策略
    - 动态阻力支撑
    - 成交量确认
    - 假突破过滤
    """
    df = prepare_dataframe(df)

    # 计算关键位
    df['highest'] = df['High'].rolling(window=lookback, min_periods=1).max()
    df['lowest'] = df['Low'].rolling(window=lookback, min_periods=1).min()
    df['close_highest'] = df['Close'].rolling(window=lookback, min_periods=1).max()
    df['close_lowest'] = df['Close'].rolling(window=lookback, min_periods=1).min()

    # 成交量
    df['volume_avg'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['volume_spike'] = df['Volume'] > (df['volume_avg'] * volume_threshold)

    # ATR用于动态止损
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    entry_price = 0
    breakout_confirmed = False

    for i in range(lookback + 1, len(df)):
        # 检查数据
        if pd.isna(df['highest'].iloc[i]) or pd.isna(df['atr'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        high_resistance = df['highest'].iloc[i-1]
        low_support = df['lowest'].iloc[i-1]

        # 向上突破
        if price > high_resistance * 1.002 and not breakout_confirmed:
            if df['volume_spike'].iloc[i]:
                breakout_confirmed = True
                if position == 0:
                    signals.at[df.index[i], 'long_entry'] = True
                    position = 1
                    entry_price = price

        # 向下突破
        elif price < low_support * 0.998 and not breakout_confirmed:
            if df['volume_spike'].iloc[i]:
                breakout_confirmed = True
                if position == 0:
                    signals.at[df.index[i], 'short_entry'] = True
                    position = -1
                    entry_price = price

        # 重置突破确认
        if not (price > high_resistance * 1.002 or price < low_support * 0.998):
            breakout_confirmed = False

        # 动态止损
        if position == 1 and entry_price > 0:
            atr = df['atr'].iloc[i]
            stop_loss = entry_price - (2 * atr)
            if price < stop_loss:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0

        elif position == -1 and entry_price > 0:
            atr = df['atr'].iloc[i]
            stop_loss = entry_price + (2 * atr)
            if price > stop_loss:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def volatility_scalping(df, bb_period=10, bb_std=1.2):
    """
    波动率剥头皮策略
    适用于震荡市场的短期交易
    """
    df = prepare_dataframe(df)

    # 紧缩的布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
        df['Close'], bb_period, bb_std)

    # 快速RSI
    df['rsi_fast'] = calculate_rsi(df['Close'], 5)

    # 价格速度
    df['price_change'] = df['Close'].pct_change()
    df['speed'] = abs(df['price_change'].rolling(window=3).sum())

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    max_hold_bars = 5  # 最多持有5个周期

    for i in range(bb_period + 5, len(df)):
        # 检查数据
        if any(pd.isna(df[col].iloc[i]) for col in ['bb_upper', 'bb_lower', 'rsi_fast']):
            continue

        price = df['Close'].iloc[i]
        upper = df['bb_upper'].iloc[i]
        lower = df['bb_lower'].iloc[i]
        rsi = df['rsi_fast'].iloc[i]
        speed = df['speed'].iloc[i]

        # 只在低波动时交易
        if speed < 0.02:  # 2%以下的速度
            # 触及下轨买入
            if price <= lower and rsi < 30 and position == 0:
                signals.at[df.index[i], 'long_entry'] = True
                position = 1
                max_hold_bars = 5

            # 触及上轨卖出
            elif price >= upper and rsi > 70 and position == 0:
                signals.at[df.index[i], 'short_entry'] = True
                position = -1
                max_hold_bars = 5

        # 快速出场
        max_hold_bars -= 1

        if position == 1:
            # 回到中轨或时间止损
            if price >= df['bb_middle'].iloc[i] or max_hold_bars <= 0:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0

        elif position == -1:
            # 回到中轨或时间止损
            if price <= df['bb_middle'].iloc[i] or max_hold_bars <= 0:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


# 策略字典
SIMPLE_OPTIMIZED_STRATEGIES = {
    '改进均值回归': improved_mean_reversion,
    '增强RSI策略': enhanced_rsi_strategy,
    '趋势动量策略': trend_momentum_strategy,
    '改进突破策略': breakout_strategy,
    '波动率剥头皮': volatility_scalping,
}


def get_simple_optimized_strategy_list():
    """获取所有简化优化策略名称列表"""
    return list(SIMPLE_OPTIMIZED_STRATEGIES.keys())