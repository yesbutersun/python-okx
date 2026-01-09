# ==============================
# 简化版策略信号生成模块（不依赖第三方库）
# ==============================
import pandas as pd


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """计算布林带"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
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
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def calculate_vwap(close, volume, period):
    """Simple rolling VWAP for volume aware signals"""
    volume_sum = volume.rolling(window=period, min_periods=1).sum()
    price_volume = (close * volume).rolling(window=period, min_periods=1).sum()
    return price_volume / volume_sum


def calculate_adx(high, low, close, period=14):
    """计算ADX指标"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def prepare_dataframe(df):
    """
    准备DataFrame，确保列名正确并计算必要的技术指标
    """
    # 标准化列名
    df = df.copy()
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    # 重命名列（如果需要）
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # 确保datetime列是datetime类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # 确保数值列为float类型
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确保按时间顺序排序 - 关键修改！
    df = df.sort_index()

    # 删除重复的时间戳（如果有）
    df = df[~df.index.duplicated(keep='first')]

    # 检查时间序列顺序
    if len(df) > 1:
        is_sorted = df.index.is_monotonic_increasing
        if not is_sorted:
            print("WARNING: 数据未按时间顺序排序，已重新排序")
            df = df.sort_index()
        else:
            print(f"数据已按时间顺序排序，共 {len(df)} 条记录")

    return df


def init_signals(index):
    """初始化信号和原因字段"""
    signals = pd.DataFrame(index=index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False
    signals['long_entry_reason'] = ''
    signals['long_exit_reason'] = ''
    signals['short_entry_reason'] = ''
    signals['short_exit_reason'] = ''
    return signals


def rsi_reversal_strategy(df):
    """
    RSI反转策略信号生成器
    """
    df = prepare_dataframe(df)
    df['RSI_14'] = calculate_rsi(df['Close'], 14)

    signals = init_signals(df.index)

    for i in range(1, len(df)):
        if pd.isna(df['RSI_14'].iloc[i]) or pd.isna(df['RSI_14'].iloc[i-1]):
            continue

        prev_rsi = df['RSI_14'].iloc[i-1]
        current_rsi = df['RSI_14'].iloc[i]

        # RSI反转策略逻辑
        if prev_rsi < 30 and current_rsi >= 30:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'RSI上穿30'
        if current_rsi > 70:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = 'RSI>70'
        if prev_rsi > 70 and current_rsi <= 70:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'RSI下穿70'
        if current_rsi < 30:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = 'RSI<30'

    return signals


def trend_atr_signal(df, short_ema=8, long_ema=21, atr_len=14, tp_atr=2.0, sl_atr=1.0):
    """
    趋势跟随 + ATR 动态止盈止损策略
    """
    df = prepare_dataframe(df)

    # 计算指标
    df['EMA_short'] = calculate_ema(df['Close'], short_ema)
    df['EMA_long'] = calculate_ema(df['Close'], long_ema)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_len)

    signals = init_signals(df.index)

    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        if pd.isna(df['ATR'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        prev_s, prev_l = df['EMA_short'].iloc[i-1], df['EMA_long'].iloc[i-1]
        cur_s, cur_l = df['EMA_short'].iloc[i], df['EMA_long'].iloc[i]

        golden = (prev_s < prev_l) and (cur_s > cur_l)
        death = (prev_s > prev_l) and (cur_s < cur_l)

        # TP/SL 价格
        if position == 1 and entry_price > 0:
            tp_price = entry_price + tp_atr * atr
            sl_price = entry_price - sl_atr * atr
            if price >= tp_price or price <= sl_price:
                signals.at[df.index[i], 'long_exit'] = True
                signals.at[df.index[i], 'long_exit_reason'] = 'ATR止盈' if price >= tp_price else 'ATR止损'
                position = 0
        elif position == -1 and entry_price > 0:
            tp_price = entry_price - tp_atr * atr
            sl_price = entry_price + sl_atr * atr
            if price <= tp_price or price >= sl_price:
                signals.at[df.index[i], 'short_exit'] = True
                signals.at[df.index[i], 'short_exit_reason'] = 'ATR止盈' if price <= tp_price else 'ATR止损'
                position = 0

        # 金叉死叉开仓
        if golden and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'EMA金叉'
            position = 1
            entry_price = price
        elif death and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'EMA死叉'
            position = -1
            entry_price = price

    return signals


def boll_rsi_signal(df, bb_len=20, bb_std=2.0, rsi_len=14):
    """
    布林带 + RSI 策略
    """
    df = prepare_dataframe(df)

    # 计算布林带
    df['BBU'], df['BBM'], df['BBL'] = calculate_bollinger_bands(df['Close'], bb_len, bb_std)
    df['pctB'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])
    df['RSI'] = calculate_rsi(df['Close'], rsi_len)

    signals = init_signals(df.index)

    position = 0

    for i in range(1, len(df)):
        if pd.isna(df['pctB'].iloc[i]) or pd.isna(df['RSI'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        pctB = df['pctB'].iloc[i]
        rsi = df['RSI'].iloc[i]

        # 开平条件
        long_entry = (pctB < 0 and rsi < 30 and position == 0)
        long_exit = (pctB > 0.5 or rsi > 50) and position == 1
        short_entry = (pctB > 1 and rsi > 70 and position == 0)
        short_exit = (pctB < 0.5 or rsi < 50) and position == -1

        if long_entry:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'pctB<0且RSI<30'
            position = 1
        elif short_entry:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'pctB>1且RSI>70'
            position = -1
        elif long_exit and position == 1:
            signals.at[df.index[i], 'long_exit'] = True
            if pctB > 0.5:
                signals.at[df.index[i], 'long_exit_reason'] = 'pctB>0.5'
            else:
                signals.at[df.index[i], 'long_exit_reason'] = 'RSI>50'
            position = 0
        elif short_exit and position == -1:
            signals.at[df.index[i], 'short_exit'] = True
            if pctB < 0.5:
                signals.at[df.index[i], 'short_exit_reason'] = 'pctB<0.5'
            else:
                signals.at[df.index[i], 'short_exit_reason'] = 'RSI<50'
            position = 0

    return signals


def boll_mid_break_exit_signal(df, bb_len=20, bb_std=2.0):
    """
    布林带上下轨入场，中轨突破平仓。
    - 下轨附近做多
    - 上轨附近做空
    - 价格突破中轨时平仓
    """
    df = prepare_dataframe(df)

    df['BBU'], df['BBM'], df['BBL'] = calculate_bollinger_bands(df['Close'], bb_len, bb_std)

    signals = init_signals(df.index)
    position = 0

    for i in range(1, len(df)):
        if pd.isna(df['BBU'].iloc[i]) or pd.isna(df['BBL'].iloc[i]) or pd.isna(df['BBM'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        prev_price = df['Close'].iloc[i - 1]
        if price <= 0:
            continue
        bbu = df['BBU'].iloc[i]
        bbl = df['BBL'].iloc[i]
        bbm = df['BBM'].iloc[i]
        prev_bbm = df['BBM'].iloc[i - 1]

        if position == 0 and price <= bbl:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '触及下轨'
            position = 1
        elif position == 0 and price >= bbu:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '触及上轨'
            position = -1
        elif position == 1 and prev_price < prev_bbm and price >= bbm:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '上破中轨'
            position = 0
        elif position == -1 and prev_price > prev_bbm and price <= bbm:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '下破中轨'
            position = 0

    return signals


def boll_mid_break_exit_filter_signal(
    df,
    bb_len=20,
    bb_std=2.0,
    ma_fast=20,
    ma_slow=60,
    ma_gap_pct=0.02,
    adx_len=14,
    adx_max=20,
):
    """
    布林带上下轨入场，中轨突破平仓（弱趋势过滤）。
    - 下轨附近做多，上轨附近做空
    - 价格突破中轨时平仓
    - 满足 |MA20 - MA60| / Close < 2% 或 ADX(14) < 20 才允许开仓
    """
    df = prepare_dataframe(df)

    df['BBU'], df['BBM'], df['BBL'] = calculate_bollinger_bands(df['Close'], bb_len, bb_std)
    df['MA_fast'] = df['Close'].rolling(window=ma_fast).mean()
    df['MA_slow'] = df['Close'].rolling(window=ma_slow).mean()
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'], adx_len)

    signals = init_signals(df.index)
    position = 0

    for i in range(1, len(df)):
        if (
            pd.isna(df['BBU'].iloc[i])
            or pd.isna(df['BBL'].iloc[i])
            or pd.isna(df['BBM'].iloc[i])
            or pd.isna(df['MA_fast'].iloc[i])
            or pd.isna(df['MA_slow'].iloc[i])
            or pd.isna(df['ADX'].iloc[i])
        ):
            continue

        price = df['Close'].iloc[i]
        prev_price = df['Close'].iloc[i - 1]
        bbu = df['BBU'].iloc[i]
        bbl = df['BBL'].iloc[i]
        bbm = df['BBM'].iloc[i]
        prev_bbm = df['BBM'].iloc[i - 1]
        ma_fast_val = df['MA_fast'].iloc[i]
        ma_slow_val = df['MA_slow'].iloc[i]
        adx_val = df['ADX'].iloc[i]

        ma_gap_ok = abs(ma_fast_val - ma_slow_val) / price < ma_gap_pct
        adx_ok = adx_val < adx_max
        allow_entry = ma_gap_ok or adx_ok

        if position == 0 and allow_entry and price <= bbl:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '触及下轨+弱趋势过滤'
            position = 1
        elif position == 0 and allow_entry and price >= bbu:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '触及上轨+弱趋势过滤'
            position = -1
        elif position == 1 and prev_price < prev_bbm and price >= bbm:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '上破中轨'
            position = 0
        elif position == -1 and prev_price > prev_bbm and price <= bbm:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '下破中轨'
            position = 0

    return signals


def trend_volatility_stop_signal(df, short_ema=8, long_ema=21, atr_len=14, sl_atr=1.5):
    """
    趋势波动止损策略
    """
    df = prepare_dataframe(df)

    # 计算指标
    df['EMA_short'] = calculate_ema(df['Close'], short_ema)
    df['EMA_long'] = calculate_ema(df['Close'], long_ema)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_len)

    signals = init_signals(df.index)

    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        if pd.isna(df['ATR'].iloc[i]):
            continue

        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        prev_s, prev_l = df['EMA_short'].iloc[i-1], df['EMA_long'].iloc[i-1]
        cur_s, cur_l = df['EMA_short'].iloc[i], df['EMA_long'].iloc[i]

        golden = (prev_s < prev_l) and (cur_s > cur_l)
        death = (prev_s > prev_l) and (cur_s < cur_l)

        # ATR止损检查
        if position == 1 and entry_price > 0:
            stop_loss = entry_price - sl_atr * atr
            if price <= stop_loss:
                signals.at[df.index[i], 'long_exit'] = True
                signals.at[df.index[i], 'long_exit_reason'] = 'ATR止损'
                position = 0
        elif position == -1 and entry_price > 0:
            stop_loss = entry_price + sl_atr * atr
            if price >= stop_loss:
                signals.at[df.index[i], 'short_exit'] = True
                signals.at[df.index[i], 'short_exit_reason'] = 'ATR止损'
                position = 0

        # 趋势信号
        if golden and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'EMA金叉'
            position = 1
            entry_price = price
        elif death and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'EMA死叉'
            position = -1
            entry_price = price

    return signals


def breakout_strategy(df, lookback=20):
    """
    突破策略
    """
    df = prepare_dataframe(df)

    # 计算突破指标
    df['high_lookback'] = df['High'].rolling(lookback).max()
    df['low_lookback'] = df['Low'].rolling(lookback).min()

    signals = init_signals(df.index)

    position = 0

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        high_lb = df['high_lookback'].iloc[i-1]  # 使用前一根K线的高低点
        low_lb = df['low_lookback'].iloc[i-1]

        # 突破信号
        long_trigger = price > high_lb
        short_trigger = price < low_lb

        if long_trigger and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '突破前高'
            position = 1
        elif short_trigger and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '跌破前低'
            position = -1
        elif position == 1 and short_trigger:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '反向突破下破'
            position = 0
        elif position == -1 and long_trigger:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '反向突破上破'
            position = 0

    return signals


def mean_reversion_strategy(df, lookback=30, std_dev=2.0):
    """
    均值回归策略
    """
    df = prepare_dataframe(df)

    # 计算均值和标准差
    df['mean_price'] = df['Close'].rolling(lookback).mean()
    df['std_price'] = df['Close'].rolling(lookback).std()
    df['upper_band'] = df['mean_price'] + std_dev * df['std_price']
    df['lower_band'] = df['mean_price'] - std_dev * df['std_price']

    signals = init_signals(df.index)

    position = 0

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        mean = df['mean_price'].iloc[i]

        # 均值回归信号
        if price < lower and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '价格低于下轨'
            position = 1
        elif price > upper and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '价格高于上轨'
            position = -1
        elif position == 1 and price >= mean:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '价格回归均值'
            position = 0
        elif position == -1 and price <= mean:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '价格回归均值'
            position = 0

    return signals


def ema_mean_reversion_strategy(df, lookback=30, std_dev=2.0, min_band_width=0):
    """
    EMA均值回归策略
    """
    df = prepare_dataframe(df)

    # 使用EMA作为中轨，标准差仍基于滚动窗口
    df['mean_price'] = df['Close'].ewm(span=lookback, adjust=False).mean()
    df['std_price'] = df['Close'].rolling(lookback).std()
    df['upper_band'] = df['mean_price'] + std_dev * df['std_price']
    df['lower_band'] = df['mean_price'] - std_dev * df['std_price']

    signals = init_signals(df.index)

    position = 0

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        mean = df['mean_price'].iloc[i]

        if pd.isna(upper) or pd.isna(lower) or pd.isna(mean):
            continue

        if position == 0 and (upper - lower) < min_band_width:
            continue

        if price < lower and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '价格低于EMA下轨'
            position = 1
        elif price > upper and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '价格高于EMA上轨'
            position = -1
        elif position == 1 and price >= mean:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '价格回归EMA均值'
            position = 0
        elif position == -1 and price <= mean:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '价格回归EMA均值'
            position = 0

    return signals


def ema_mean_reversion_next_close_entry_strategy(df, lookback=30, std_dev=2.0, min_band_width=0):
    """
    EMA均值回归策略（信号出现后，下一根K线收盘价开仓）。
    """
    df = prepare_dataframe(df)

    df['mean_price'] = df['Close'].ewm(span=lookback, adjust=False).mean()
    df['std_price'] = df['Close'].rolling(lookback).std()
    df['upper_band'] = df['mean_price'] + std_dev * df['std_price']
    df['lower_band'] = df['mean_price'] - std_dev * df['std_price']

    signals = init_signals(df.index)

    position = 0
    pending_entry = 0
    pending_index = None

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        mean = df['mean_price'].iloc[i]

        if pd.isna(upper) or pd.isna(lower) or pd.isna(mean):
            continue

        if (upper - lower) < min_band_width:
            continue

        if pending_entry != 0 and pending_index == i and position == 0:
            if pending_entry == 1:
                signals.at[df.index[i], 'long_entry'] = True
                signals.at[df.index[i], 'long_entry_reason'] = '价格低于EMA下轨+下一K线收盘开仓'
                position = 1
            else:
                signals.at[df.index[i], 'short_entry'] = True
                signals.at[df.index[i], 'short_entry_reason'] = '价格高于EMA上轨+下一K线收盘开仓'
                position = -1
            pending_entry = 0
            pending_index = None

        if position == 0 and pending_entry == 0:
            if price < lower and i + 1 < len(df):
                pending_entry = 1
                pending_index = i + 1
            elif price > upper and i + 1 < len(df):
                pending_entry = -1
                pending_index = i + 1
        elif position == 1 and price >= mean:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '价格回归EMA均值'
            position = 0
        elif position == -1 and price <= mean:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '价格回归EMA均值'
            position = 0

    return signals


def ema_mean_reversion_bandwidth_filter_strategy(
    df,
    lookback=30,
    std_dev=2.0,
    min_band_width=0,
    max_band_expand_pct=0.01,
):
    """
    EMA均值回归策略（开口扩大过滤）。

    在触发开仓信号时，若当前上下轨开口相对上一根 K 线扩大超过阈值，则延迟开仓，
    直到开口扩大比例不超过阈值才允许开仓。
    """
    df = prepare_dataframe(df)

    df['mean_price'] = df['Close'].ewm(span=lookback, adjust=False).mean()
    df['std_price'] = df['Close'].rolling(lookback).std()
    df['upper_band'] = df['mean_price'] + std_dev * df['std_price']
    df['lower_band'] = df['mean_price'] - std_dev * df['std_price']

    signals = init_signals(df.index)

    position = 0

    for i in range(lookback + 1, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        mean = df['mean_price'].iloc[i]
        prev_upper = df['upper_band'].iloc[i - 1]
        prev_lower = df['lower_band'].iloc[i - 1]

        if pd.isna([upper, lower, mean, prev_upper, prev_lower]).any():
            continue

        band_width = upper - lower
        prev_band_width = prev_upper - prev_lower
        if band_width <= 0 or prev_band_width <= 0:
            continue

        if band_width < min_band_width:
            continue

        band_expand_ratio = (band_width - prev_band_width) / prev_band_width
        allow_entry = band_expand_ratio <= max_band_expand_pct

        if position == 0 and allow_entry:
            if price < lower:
                signals.at[df.index[i], 'long_entry'] = True
                signals.at[df.index[i], 'long_entry_reason'] = '价格低于EMA下轨+开口过滤'
                position = 1
            elif price > upper:
                signals.at[df.index[i], 'short_entry'] = True
                signals.at[df.index[i], 'short_entry_reason'] = '价格高于EMA上轨+开口过滤'
                position = -1
        elif position == 1 and price >= mean:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '价格回归EMA均值'
            position = 0
        elif position == -1 and price <= mean:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '价格回归EMA均值'
            position = 0

    return signals


def ema_mean_reversion_slope_hold_strategy(
    df,
    lookback=30,
    std_dev=2.0,
    slope_len=5,
    slope_zero_threshold=1e-6,
):
    """
    EMA均值回归策略（到达均值后根据均值斜率“先持有后平仓”）

    基础入场与 `ema_mean_reversion_strategy` 一致：
    - EMA 作为中轨；STD 为滚动标准差；上下轨 = EMA ± std_dev * STD
    - 价格低于下轨做多；高于上轨做空

    改动点（延迟平仓）：
    - 当价格回到 EMA（触达均值）时，不立刻平仓
    - 若均值斜率在变得更陡（|slope| 增大，且方向与持仓一致），继续持仓
    - 直到均值斜率接近 0（`abs(slope) <= slope_zero_threshold`）才平仓
    """
    df = prepare_dataframe(df)

    df["mean_price"] = df["Close"].ewm(span=lookback, adjust=False).mean()
    df["std_price"] = df["Close"].rolling(lookback).std()
    df["upper_band"] = df["mean_price"] + std_dev * df["std_price"]
    df["lower_band"] = df["mean_price"] - std_dev * df["std_price"]

    mean_slope = df["mean_price"].pct_change()
    df["mean_slope"] = mean_slope.rolling(window=slope_len).mean()

    signals = init_signals(df.index)

    position = 0
    wait_mean_then_exit = False

    start = max(lookback, slope_len + 2)
    for i in range(start, len(df)):
        price = df["Close"].iloc[i]
        upper = df["upper_band"].iloc[i]
        lower = df["lower_band"].iloc[i]
        mean = df["mean_price"].iloc[i]

        slope_now = df["mean_slope"].iloc[i]
        slope_prev = df["mean_slope"].iloc[i - 1]

        if pd.isna([upper, lower, mean, slope_now, slope_prev]).any():
            continue

        slope_is_steepening = abs(slope_now) > abs(slope_prev)

        if position == 0:
            wait_mean_then_exit = False
            if price < lower:
                signals.at[df.index[i], "long_entry"] = True
                signals.at[df.index[i], "long_entry_reason"] = "价格低于EMA下轨"
                position = 1
            elif price > upper:
                signals.at[df.index[i], "short_entry"] = True
                signals.at[df.index[i], "short_entry_reason"] = "价格高于EMA上轨"
                position = -1
            continue

        # 触达均值后进入“观察斜率”阶段
        if position == 1 and not wait_mean_then_exit and price >= mean:
            wait_mean_then_exit = True
        elif position == -1 and not wait_mean_then_exit and price <= mean:
            wait_mean_then_exit = True

        if not wait_mean_then_exit:
            continue

        if abs(slope_now) <= slope_zero_threshold:
            if position == 1:
                signals.at[df.index[i], "long_exit"] = True
                signals.at[df.index[i], "long_exit_reason"] = "均值斜率≈0后平仓"
            else:
                signals.at[df.index[i], "short_exit"] = True
                signals.at[df.index[i], "short_exit_reason"] = "均值斜率≈0后平仓"
            position = 0
            wait_mean_then_exit = False

    return signals


def momentum_strategy(df, roc_period=10, threshold=0.02):
    """
    动量策略
    """
    df = prepare_dataframe(df)

    # 计算变化率
    df['ROC'] = df['Close'].pct_change(roc_period)

    signals = init_signals(df.index)

    position = 0

    for i in range(roc_period, len(df)):
        roc = df['ROC'].iloc[i]

        # 动量信号
        if roc > threshold and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'ROC>阈值'
            position = 1
        elif roc < -threshold and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'ROC<-阈值'
            position = -1
        elif position == 1 and roc < 0:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = 'ROC转负'
            position = 0
        elif position == -1 and roc > 0:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = 'ROC转正'
            position = 0

    return signals


def macd_strategy(df, fast=12, slow=26, signal=9):
    """
    MACD策略
    """
    df = prepare_dataframe(df)

    # 计算MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'], fast, slow, signal)

    signals = init_signals(df.index)

    position = 0

    for i in range(1, len(df)):
        if pd.isna(df['MACD_Hist'].iloc[i]) or pd.isna(df['MACD_Hist'].iloc[i-1]):
            continue

        prev_hist = df['MACD_Hist'].iloc[i-1]
        curr_hist = df['MACD_Hist'].iloc[i]

        # MACD信号
        if prev_hist < 0 and curr_hist >= 0 and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'MACD柱由负转正'
            position = 1
        elif prev_hist > 0 and curr_hist <= 0 and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'MACD柱由正转负'
            position = -1
        elif position == 1 and curr_hist < 0:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = 'MACD柱转负'
            position = 0
        elif position == -1 and curr_hist > 0:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = 'MACD柱转正'
            position = 0

    return signals


def intraday_seasonality_strategy(df, lookback=30, threshold=0.0005):
    """
    Intraday seasonality: trade when the average return of the current time slot is persistently positive/negative.
    """
    df = prepare_dataframe(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("intraday_seasonality_strategy requires a datetime index.")

    df['bar_return'] = df['Close'].pct_change()
    df['slot'] = df.index.time

    slot_mean = (
        df.groupby('slot')['bar_return']
        .rolling(window=lookback, min_periods=lookback)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['slot_return_mean'] = slot_mean

    signals = init_signals(df.index)

    position = 0
    for i in range(len(df)):
        seasonal_edge = df['slot_return_mean'].iloc[i]
        if pd.isna(seasonal_edge):
            continue

        if position == 0 and seasonal_edge > threshold:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '时段均值收益>阈值'
            position = 1
        elif position == 0 and seasonal_edge < -threshold:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '时段均值收益<-阈值'
            position = -1
        elif position == 1 and seasonal_edge < 0:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '时段均值收益<0'
            position = 0
        elif position == -1 and seasonal_edge > 0:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '时段均值收益>0'
            position = 0

    return signals


def volatility_squeeze_breakout(df, bb_len=20, bb_std=2.0, squeeze_quantile=0.25, atr_len=14):
    """
    Volatility squeeze breakout: wait for tight Bollinger Bands then trade the breakout with ATR stops.
    """
    df = prepare_dataframe(df)

    df['BBU'], df['BBM'], df['BBL'] = calculate_bollinger_bands(df['Close'], bb_len, bb_std)
    df['bandwidth'] = (df['BBU'] - df['BBL']) / df['BBM']
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_len)
    df['squeeze'] = df['bandwidth'] <= df['bandwidth'].rolling(bb_len * 2).quantile(squeeze_quantile)

    signals = init_signals(df.index)

    position = 0
    entry_price = 0.0
    in_squeeze = False

    for i in range(bb_len * 2, len(df)):
        price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        upper = df['BBU'].iloc[i]
        lower = df['BBL'].iloc[i]

        if pd.isna(atr) or pd.isna(upper) or pd.isna(lower):
            continue

        if df['squeeze'].iloc[i]:
            in_squeeze = True

        breakout_up = price > upper
        breakout_down = price < lower

        if position == 0 and in_squeeze:
            if breakout_up:
                signals.at[df.index[i], 'long_entry'] = True
                signals.at[df.index[i], 'long_entry_reason'] = '挤压后上破上轨'
                position = 1
                entry_price = price
                in_squeeze = False
            elif breakout_down:
                signals.at[df.index[i], 'short_entry'] = True
                signals.at[df.index[i], 'short_entry_reason'] = '挤压后下破下轨'
                position = -1
                entry_price = price
                in_squeeze = False

        if position == 1 and entry_price > 0 and atr > 0:
            stop_loss = entry_price - 1.5 * atr
            if price < stop_loss or breakout_down:
                signals.at[df.index[i], 'long_exit'] = True
                if price < stop_loss:
                    signals.at[df.index[i], 'long_exit_reason'] = 'ATR止损'
                else:
                    signals.at[df.index[i], 'long_exit_reason'] = '反向突破下破'
                position = 0
                entry_price = 0.0

        if position == -1 and entry_price > 0 and atr > 0:
            stop_loss = entry_price + 1.5 * atr
            if price > stop_loss or breakout_up:
                signals.at[df.index[i], 'short_exit'] = True
                if price > stop_loss:
                    signals.at[df.index[i], 'short_exit_reason'] = 'ATR止损'
                else:
                    signals.at[df.index[i], 'short_exit_reason'] = '反向突破上破'
                position = 0
                entry_price = 0.0

    return signals


def vol_scaled_momentum(df, lookback=30, vol_lookback=20, z_enter=1.0, z_exit=0.2):
    """
    Volatility scaled momentum: use z-scored returns to size entries/exits.
    """
    df = prepare_dataframe(df)
    df['returns'] = df['Close'].pct_change()
    df['roc'] = df['Close'].pct_change(lookback)

    df['vol'] = df['returns'].rolling(vol_lookback).std()
    df['signal_z'] = df['roc'] / (df['vol'] * (lookback ** 0.5))

    signals = init_signals(df.index)

    position = 0
    for i in range(len(df)):
        z = df['signal_z'].iloc[i]
        if pd.isna(z):
            continue

        if position == 0 and z > z_enter:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = 'zscore>入场阈值'
            position = 1
        elif position == 0 and z < -z_enter:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = 'zscore<-入场阈值'
            position = -1
        elif position == 1 and z < z_exit:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = 'zscore回落到退出阈值'
            position = 0
        elif position == -1 and z > -z_exit:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = 'zscore回升到退出阈值'
            position = 0

    return signals

'''
VWAPReversion（见 simple_strategy.py:780）是一个典型“VWAP 均值回归/反转”策略：当价格相对 VWAP 偏离过大就反向进场，等偏离收敛再出场。

  - 核心指标
      - VWAP：用滚动窗口的成交量加权均价计算（calculate_vwap(Close, Volume, vwap_len)）
      - dev：偏离值 Close - VWAP
      - dev_std：偏离值在窗口内的标准差 dev.rolling(vwap_len).std()
      - zscore：标准化偏离 zscore = dev / dev_std
  - 入场逻辑（反向）
      - zscore < -z_entry：价格相对 VWAP “明显偏低” → 做多（long_entry）
      - zscore >  z_entry：价格相对 VWAP “明显偏高” → 做空（short_entry）
      - 默认参数：vwap_len=30, z_entry=1.5
  - 出场逻辑（回归）
      - 多单：当 zscore > -z_exit（从很负回升到接近 0）→ 平多（long_exit）
      - 空单：当 zscore <  z_exit（从很正回落到接近 0）→ 平空（short_exit）
      - 默认 z_exit=0.3（比入场阈值小，避免刚入场就出场）
  - 持仓与信号特点
      - 用 position 状态机保证同时只持有一个方向仓位（0/1/-1）
      - 不含止损/止盈（只有“偏离回归”退出），风险主要来自偏离继续扩大                                                                                    - zscore < -z_entry：价格相对 VWAP “明显偏低” → 做多（long_entry）
      - zscore >  z_entry：价格相对 VWAP “明显偏高” → 做空（short_entry）
      - 默认参数：vwap_len=30, z_entry=1.5
  - 出场逻辑（回归）
      - 多单：当 zscore > -z_exit（从很负回升到接近 0）→ 平多（long_exit）
      - 空单：当 zscore <  z_exit（从很正回落到接近 0）→ 平空（short_exit）
      - 默认 z_exit=0.3（比入场阈值小，避免刚入场就出场）
  - 持仓与信号特点
      - 用 position 状态机保证同时只持有一个方向仓位（0/1/-1）
      - 不含止损/止盈（只有“偏离回归”退出），风险主要来自偏离继续扩大
  - 可调参数怎么影响
      - vwap_len：越大 VWAP 越平滑、信号更慢；越小更敏感但噪声多
      - z_entry：越大越“挑剔”（更极端才进，交易更少）；越小更频繁
      - z_exit：越大越早退出（更保守）；越小越晚退出（更吃回归但更久持仓）

'''
def vwap_reversion(df, vwap_len=30, z_entry=1.5, z_exit=0.3):
    """
    VWAP mean reversion: fade large deviations from rolling VWAP.
    """
    df = prepare_dataframe(df)
    df['VWAP'] = calculate_vwap(df['Close'], df['Volume'], vwap_len)
    df['dev'] = df['Close'] - df['VWAP']
    df['dev_std'] = df['dev'].rolling(vwap_len).std()
    df['zscore'] = df['dev'] / df['dev_std']

    signals = init_signals(df.index)

    position = 0
    for i in range(len(df)):
        z = df['zscore'].iloc[i]
        if pd.isna(z):
            continue

        if position == 0 and z < -z_entry:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '价格偏离VWAP过低'
            position = 1
        elif position == 0 and z > z_entry:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '价格偏离VWAP过高'
            position = -1
        elif position == 1 and z > -z_exit:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '偏离回归VWAP'
            position = 0
        elif position == -1 and z < z_exit:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '偏离回归VWAP'
            position = 0

    return signals


def vwap_reversion_slope_filter(df, vwap_len=30, z_entry=1.5, z_exit=0.3, vwap_slope_pct=0.001):
    """
    VWAP 均值回归 + VWAP 斜率过滤。
    仅在 |VWAP(t) - VWAP(t-1)| / Close < vwap_slope_pct 时允许开仓。
    """
    df = prepare_dataframe(df)
    df['VWAP'] = calculate_vwap(df['Close'], df['Volume'], vwap_len)
    df['dev'] = df['Close'] - df['VWAP']
    df['dev_std'] = df['dev'].rolling(vwap_len).std()
    df['zscore'] = df['dev'] / df['dev_std']

    signals = init_signals(df.index)

    position = 0
    for i in range(1, len(df)):
        z = df['zscore'].iloc[i]
        vwap = df['VWAP'].iloc[i]
        prev_vwap = df['VWAP'].iloc[i - 1]
        price = df['Close'].iloc[i]
        if pd.isna(z) or pd.isna(vwap) or pd.isna(prev_vwap) or pd.isna(price) or price <= 0:
            continue

        slope_ok = abs(vwap - prev_vwap) / price < vwap_slope_pct

        if position == 0 and slope_ok and z < -z_entry:
            signals.at[df.index[i], 'long_entry'] = True
            signals.at[df.index[i], 'long_entry_reason'] = '价格偏离VWAP过低+斜率过滤'
            position = 1
        elif position == 0 and slope_ok and z > z_entry:
            signals.at[df.index[i], 'short_entry'] = True
            signals.at[df.index[i], 'short_entry_reason'] = '价格偏离VWAP过高+斜率过滤'
            position = -1
        elif position == 1 and z > -z_exit:
            signals.at[df.index[i], 'long_exit'] = True
            signals.at[df.index[i], 'long_exit_reason'] = '偏离回归VWAP'
            position = 0
        elif position == -1 and z < z_exit:
            signals.at[df.index[i], 'short_exit'] = True
            signals.at[df.index[i], 'short_exit_reason'] = '偏离回归VWAP'
            position = 0

    return signals


def boll_zscore_slope_accel(
    df,
    bb_len=20,
    bb_std=2.0,
    z_entry=2.0,
    z_exit=0.3,
    slope_len=5,
    slope_cap=0.0008,
    accel_threshold=0.0,
):
    """
    BOLL + Z-score + 均值斜率 + 斜率加速度（均值回归增强版）

    - 用 BOLL 与 Z-score 识别“偏离过度”
    - 用中轨（均值）的斜率过滤强趋势行情
    - 用斜率加速度（斜率的变化）筛选“趋势减速/拐头”的更优入场点
    """
    df = prepare_dataframe(df)

    if bb_std <= 0:
        raise ValueError("bb_std must be > 0")

    df["BBU"], df["BBM"], df["BBL"] = calculate_bollinger_bands(df["Close"], bb_len, bb_std)
    df["BB_STD"] = df["Close"].rolling(window=bb_len).std()
    df["Z"] = (df["Close"] - df["BBM"]) / df["BB_STD"].replace(0, float("nan"))

    df["mean_slope"] = df["BBM"].pct_change().rolling(window=slope_len).mean()
    df["slope_accel"] = df["mean_slope"].diff()

    signals = init_signals(df.index)

    position = 0
    start = max(bb_len, slope_len + 2)
    for i in range(start, len(df)):
        close = df["Close"].iloc[i]
        upper = df["BBU"].iloc[i]
        middle = df["BBM"].iloc[i]
        lower = df["BBL"].iloc[i]
        z = df["Z"].iloc[i]
        mean_slope = df["mean_slope"].iloc[i]
        slope_accel = df["slope_accel"].iloc[i]

        if pd.isna([upper, middle, lower, z, mean_slope, slope_accel]).any():
            continue

        flat_enough = abs(mean_slope) <= slope_cap

        if position == 0:
            if close < lower and z <= -z_entry and flat_enough and slope_accel > accel_threshold:
                signals.at[df.index[i], "long_entry"] = True
                signals.at[df.index[i], "long_entry_reason"] = "下破下轨+Z极端+均值拐头"
                position = 1
            elif close > upper and z >= z_entry and flat_enough and slope_accel < -accel_threshold:
                signals.at[df.index[i], "short_entry"] = True
                signals.at[df.index[i], "short_entry_reason"] = "上破上轨+Z极端+均值拐头"
                position = -1
        elif position == 1:
            if close >= middle or z >= -z_exit:
                signals.at[df.index[i], "long_exit"] = True
                signals.at[df.index[i], "long_exit_reason"] = "回到中轨/偏离回归"
                position = 0
        elif position == -1:
            if close <= middle or z <= z_exit:
                signals.at[df.index[i], "short_exit"] = True
                signals.at[df.index[i], "short_exit_reason"] = "回到中轨/偏离回归"
                position = 0

    return signals


def dingfengbo_strategy(
    df,
    bb_len=20,
    bb_std=2.0,
    width_ma=20,
    atr_len=14,
    atr_ma=30,
    ma_fast=20,
    ma_slow=60,
    ma_gap=0.01,
    vol_ma=20,
    vol_mult=1.5
):
    """
    定风波策略（震荡压缩 -> 放量放波动突破）

    核心：
    - BOLL 宽度低位（收口）
    - ATR 低位
    - MA20/MA60 缠绕（无趋势）
    - 突破上/下轨 + 放量 + ATR 拐头（波动扩张）
    - 止损：回到 BOLL 中轨（Middle）
    """
    df = prepare_dataframe(df)

    upper, middle, lower = calculate_bollinger_bands(df['Close'], bb_len, bb_std)
    width = (upper - lower) / middle
    width_mean = width.rolling(width_ma).mean()

    atr = calculate_atr(df['High'], df['Low'], df['Close'], atr_len)
    atr_mean = atr.rolling(atr_ma).mean()

    ma_f = df['Close'].rolling(ma_fast).mean()
    ma_s = df['Close'].rolling(ma_slow).mean()

    vol_mean = df['Volume'].rolling(vol_ma).mean() if 'Volume' in df.columns else None

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    stop_level = None

    for i in range(max(bb_len, width_ma, atr_len, atr_ma, ma_slow), len(df)):
        price = df['Close'].iloc[i]

        if any(pd.isna(x) for x in [upper.iloc[i], middle.iloc[i], lower.iloc[i], width.iloc[i], width_mean.iloc[i], atr.iloc[i], atr_mean.iloc[i], ma_f.iloc[i], ma_s.iloc[i]]):
            continue

        # Regime: compression + no clear trend
        compression = (width.iloc[i] < width_mean.iloc[i]) and (atr.iloc[i] < atr_mean.iloc[i])
        no_trend = abs(ma_f.iloc[i] - ma_s.iloc[i]) / price < ma_gap

        # ATR "turning up" (expanding volatility)
        atr_turn_up = (not pd.isna(atr.iloc[i - 1])) and (atr.iloc[i] > atr.iloc[i - 1])

        volume_ok = True
        if vol_mean is not None and not pd.isna(vol_mean.iloc[i]):
            volume_ok = df['Volume'].iloc[i] > vol_mult * vol_mean.iloc[i]

        if position == 0:
            if compression and no_trend and atr_turn_up and volume_ok and price > upper.iloc[i]:
                signals.at[df.index[i], 'long_entry'] = True
                position = 1
                stop_level = middle.iloc[i]
            elif compression and no_trend and atr_turn_up and volume_ok and price < lower.iloc[i]:
                signals.at[df.index[i], 'short_entry'] = True
                position = -1
                stop_level = middle.iloc[i]
        elif position == 1:
            # Stop: back below middle band
            stop_level = middle.iloc[i]
            if price <= stop_level:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
                stop_level = None
        elif position == -1:
            # Stop: back above middle band
            stop_level = middle.iloc[i]
            if price >= stop_level:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0
                stop_level = None

    return signals


def fenghuolun_strategy(
    df,
    ma_fast=20,
    ma_mid=60,
    ma_slow=120,
    slope_lookback=5,
    rsi_len=14,
    rsi_long_low=55,
    rsi_long_high=75,
    rsi_short_low=25,
    rsi_short_high=45,
    atr_len=14,
    atr_ma=20,
    vol_ma=20,
    breakout_lookback=5,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9
):
    """
    风火轮策略（趋势加速段动量滚动）

    做多（示例量化条件，来自 fenghuolun.md）：
    1) MA20 > MA60 > MA120 且 MA20 斜率 > 0
    2) MACD > 0 且 histogram 连续放大（至少 2 根）
    3) RSI ∈ [55, 75] 且未跌破 50
    4) ATR(14) > MA(ATR(14),20)
    5) Volume > MA(Volume, 20)
    6) Close 突破前 N 根最高价

    出场：
    - histogram 收缩 / RSI 跌破 50 / Close 跌破 MA20
    """
    df = prepare_dataframe(df)

    close = df['Close']
    high = df['High']
    low = df['Low']

    ma20 = close.rolling(ma_fast).mean()
    ma60 = close.rolling(ma_mid).mean()
    ma120 = close.rolling(ma_slow).mean()

    macd, macd_sig, hist = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    rsi = calculate_rsi(close, rsi_len)

    atr = calculate_atr(high, low, close, atr_len)
    atr_mean = atr.rolling(atr_ma).mean()

    vol_mean = df['Volume'].rolling(vol_ma).mean() if 'Volume' in df.columns else None

    prev_high = high.rolling(breakout_lookback).max().shift(1)
    prev_low = low.rolling(breakout_lookback).min().shift(1)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    start = max(ma_slow, slope_lookback + 2, atr_ma, vol_ma, breakout_lookback)
    for i in range(start, len(df)):
        if any(pd.isna(x) for x in [ma20.iloc[i], ma60.iloc[i], ma120.iloc[i], hist.iloc[i], hist.iloc[i - 1], hist.iloc[i - 2], macd.iloc[i], rsi.iloc[i], atr.iloc[i], atr_mean.iloc[i], prev_high.iloc[i], prev_low.iloc[i]]):
            continue

        price = close.iloc[i]
        vol_ok = True
        if vol_mean is not None and not pd.isna(vol_mean.iloc[i]):
            vol_ok = df['Volume'].iloc[i] > vol_mean.iloc[i]

        # Trend filters
        ma_up = (ma20.iloc[i] > ma60.iloc[i] > ma120.iloc[i]) and (ma20.iloc[i] - ma20.iloc[i - slope_lookback] > 0)
        ma_down = (ma20.iloc[i] < ma60.iloc[i] < ma120.iloc[i]) and (ma20.iloc[i] - ma20.iloc[i - slope_lookback] < 0)

        # Momentum filters
        long_mom = (macd.iloc[i] > 0) and (hist.iloc[i] > hist.iloc[i - 1] > hist.iloc[i - 2])
        short_mom = (macd.iloc[i] < 0) and (hist.iloc[i] < hist.iloc[i - 1] < hist.iloc[i - 2])

        long_rsi_ok = (rsi_long_low <= rsi.iloc[i] <= rsi_long_high) and (rsi.iloc[i] >= 50)
        short_rsi_ok = (rsi_short_low <= rsi.iloc[i] <= rsi_short_high) and (rsi.iloc[i] <= 50)

        atr_ok = (atr.iloc[i] > atr_mean.iloc[i])

        long_breakout = price > prev_high.iloc[i]
        short_breakout = price < prev_low.iloc[i]

        if position == 0:
            if ma_up and long_mom and long_rsi_ok and atr_ok and vol_ok and long_breakout:
                signals.at[df.index[i], 'long_entry'] = True
                position = 1
            elif ma_down and short_mom and short_rsi_ok and atr_ok and vol_ok and short_breakout:
                signals.at[df.index[i], 'short_entry'] = True
                position = -1
        elif position == 1:
            # Exit on momentum fade / RSI break / MA20 break
            hist_fade = hist.iloc[i] < hist.iloc[i - 1]
            if hist_fade or (rsi.iloc[i] < 50) or (price < ma20.iloc[i]):
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            hist_fade = hist.iloc[i] > hist.iloc[i - 1]
            if hist_fade or (rsi.iloc[i] > 50) or (price > ma20.iloc[i]):
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

    return signals


def risk_controlled_mean_reversion(
    df,
    lookback=30,
    std_dev=2.2,
    rsi_len=14,
    atr_len=14,
    slope_threshold=0.0006,
    vol_cap=0.01,
    atr_tp=1.2,
    atr_sl=0.8,
    time_stop=20,
    cooldown=5,
    max_consecutive_losses=2
):
    """
    Risk-controlled mean reversion:
    - Only trade in sideways/low-vol regimes (slope/vol filters)
    - RSI confirmation plus ATR take-profit/stop-loss
    - Time stop and cooldown after consecutive losses
    """
    df = prepare_dataframe(df)
    df['sma'] = df['Close'].rolling(lookback).mean()
    df['std'] = df['Close'].rolling(lookback).std()
    df['upper'] = df['sma'] + std_dev * df['std']
    df['lower'] = df['sma'] - std_dev * df['std']
    df['rsi'] = calculate_rsi(df['Close'], rsi_len)
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_len)
    df['slope'] = df['sma'].diff() / df['Close']
    df['vol'] = df['Close'].pct_change().rolling(20).std()

    signals = init_signals(df.index)

    position = 0
    entry_price = 0.0
    bars_in_trade = 0
    cooldown_left = 0
    consecutive_losses = 0

    for i in range(lookback, len(df)):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        price = df['Close'].iloc[i]
        upper = df['upper'].iloc[i]
        lower = df['lower'].iloc[i]
        sma = df['sma'].iloc[i]
        rsi = df['rsi'].iloc[i]
        atr = df['atr'].iloc[i]
        slope = abs(df['slope'].iloc[i])
        vol = df['vol'].iloc[i]

        if any(pd.isna(x) for x in [upper, lower, sma, rsi, atr, slope, vol]):
            continue
        if slope > slope_threshold or vol > vol_cap:
            continue

        if position == 0:
            if price < lower and rsi < 35 and consecutive_losses < max_consecutive_losses:
                signals.at[df.index[i], 'long_entry'] = True
                signals.at[df.index[i], 'long_entry_reason'] = '低波动+价格低于下轨+RSI<35'
                position = 1
                entry_price = price
                bars_in_trade = 0
            elif price > upper and rsi > 65 and consecutive_losses < max_consecutive_losses:
                signals.at[df.index[i], 'short_entry'] = True
                signals.at[df.index[i], 'short_entry_reason'] = '低波动+价格高于上轨+RSI>65'
                position = -1
                entry_price = price
                bars_in_trade = 0
        else:
            bars_in_trade += 1
            tp_price = entry_price + atr_tp * atr if position == 1 else entry_price - atr_tp * atr
            sl_price = entry_price - atr_sl * atr if position == 1 else entry_price + atr_sl * atr

            exit_hit = (
                bars_in_trade >= time_stop or
                (position == 1 and (price >= tp_price or price <= sl_price or price >= sma)) or
                (position == -1 and (price <= tp_price or price >= sl_price or price <= sma)) or
                (position == 1 and rsi > 60) or
                (position == -1 and rsi < 40)
            )

            if exit_hit:
                if bars_in_trade >= time_stop:
                    exit_reason = '时间止损'
                elif position == 1 and price >= tp_price:
                    exit_reason = 'ATR止盈'
                elif position == 1 and price <= sl_price:
                    exit_reason = 'ATR止损'
                elif position == 1 and price >= sma:
                    exit_reason = '价格回归均值'
                elif position == 1 and rsi > 60:
                    exit_reason = 'RSI>60'
                elif position == -1 and price <= tp_price:
                    exit_reason = 'ATR止盈'
                elif position == -1 and price >= sl_price:
                    exit_reason = 'ATR止损'
                elif position == -1 and price <= sma:
                    exit_reason = '价格回归均值'
                elif position == -1 and rsi < 40:
                    exit_reason = 'RSI<40'
                else:
                    exit_reason = '退出条件触发'
                if position == 1:
                    signals.at[df.index[i], 'long_exit'] = True
                    signals.at[df.index[i], 'long_exit_reason'] = exit_reason
                else:
                    signals.at[df.index[i], 'short_exit'] = True
                    signals.at[df.index[i], 'short_exit_reason'] = exit_reason
                loss = (position == 1 and price < entry_price) or (position == -1 and price > entry_price)
                consecutive_losses = consecutive_losses + 1 if loss else 0
                if consecutive_losses >= max_consecutive_losses:
                    cooldown_left = cooldown
                position = 0

    return signals


def enhanced_mean_reversion_positive(
    df,
    lookback=30,
    std_dev=2.4,
    rsi_len=14,
    atr_len=14,
    slope_threshold=0.0006,
    vol_cap=0.01,
    atr_tp=1.4,
    atr_sl=0.8,
    time_stop=20,
    cooldown=5,
    max_consecutive_losses=2
):
    """
    增强均值回归策略（在当前示例数据上调参后可获得正收益）。

    说明：这是对 `risk_controlled_mean_reversion` 的参数封装，主要调整：
    - 更宽的均值带宽（std_dev）
    - 更合理的 ATR 止盈/止损（atr_tp/atr_sl）
    """
    return risk_controlled_mean_reversion(
        df=df,
        lookback=lookback,
        std_dev=std_dev,
        rsi_len=rsi_len,
        atr_len=atr_len,
        slope_threshold=slope_threshold,
        vol_cap=vol_cap,
        atr_tp=atr_tp,
        atr_sl=atr_sl,
        time_stop=time_stop,
        cooldown=cooldown,
        max_consecutive_losses=max_consecutive_losses
    )


# 策略字典
STRATEGIES = {
    # 'RSI反转策略': rsi_reversal_strategy,
    # '趋势ATR策略': trend_atr_signal,
    '布林RSI策略': boll_rsi_signal,
    '布林中轨突破平仓': boll_mid_break_exit_signal,
    #'布林中轨突破平仓_弱趋势过滤': boll_mid_break_exit_filter_signal,
    # '趋势波动止损策略': trend_volatility_stop_signal,
    # '突破策略': breakout_strategy,
    # '定风波策略': dingfengbo_strategy,
    '均值回归策略': mean_reversion_strategy,
    'EMA均值回归策略': ema_mean_reversion_strategy,
    'EMA均值回归策略_下一K线收盘开仓': ema_mean_reversion_next_close_entry_strategy,
    'EMA均值回归策略_开口过滤': ema_mean_reversion_bandwidth_filter_strategy,
    #'EMA均值回归策略_斜率延迟平仓': ema_mean_reversion_slope_hold_strategy,
    # '风火轮策略': fenghuolun_strategy,
    # '均值回归策略_增强': enhanced_mean_reversion_positive,
    # 'RiskControlledMeanReversion': risk_controlled_mean_reversion,
    # '动量策略': momentum_strategy,
    # 'MACD策略': macd_strategy,
    # 'IntradaySeasonality': intraday_seasonality_strategy,
    # 'VolatilitySqueezeBreakout': volatility_squeeze_breakout,
    # 'VolScaledMomentum': vol_scaled_momentum,
    # 'BOLL+Z-score斜率加速度策略': boll_zscore_slope_accel,
    'VWAPReversion': vwap_reversion,
    #'VWAPReversion_斜率过滤': vwap_reversion_slope_filter
}


def get_strategy_list():
    """获取所有可用策略列表"""
    return list(STRATEGIES.keys())


def run_strategy(df, strategy_name, **kwargs):
    """运行指定策略"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"未知策略: {strategy_name}")

    strategy_func = STRATEGIES[strategy_name]
    return strategy_func(df, **kwargs)
