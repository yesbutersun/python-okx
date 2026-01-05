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

    signals = pd.DataFrame(index=df.index)
    signals['long_entry'] = False
    signals['long_exit'] = False
    signals['short_entry'] = False
    signals['short_exit'] = False

    for i in range(1, len(df)):
        if pd.isna(df['RSI_14'].iloc[i]) or pd.isna(df['RSI_14'].iloc[i-1]):
            continue

        prev_rsi = df['RSI_14'].iloc[i-1]
        current_rsi = df['RSI_14'].iloc[i]

        # RSI反转策略逻辑
        signals.at[df.index[i], 'long_entry'] = (prev_rsi < 30 and current_rsi >= 30)
        signals.at[df.index[i], 'long_exit'] = (current_rsi > 70)
        signals.at[df.index[i], 'short_entry'] = (prev_rsi > 70 and current_rsi <= 70)
        signals.at[df.index[i], 'short_exit'] = (current_rsi < 30)

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

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

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
                position = 0
        elif position == -1 and entry_price > 0:
            tp_price = entry_price - tp_atr * atr
            sl_price = entry_price + sl_atr * atr
            if price <= tp_price or price >= sl_price:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

        # 金叉死叉开仓
        if golden and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price
        elif death and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
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

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

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
            position = 1
        elif short_entry:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
        elif long_exit and position == 1:
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif short_exit and position == -1:
            signals.at[df.index[i], 'short_exit'] = True
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

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

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
                position = 0
        elif position == -1 and entry_price > 0:
            stop_loss = entry_price + sl_atr * atr
            if price >= stop_loss:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

        # 趋势信号
        if golden and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price
        elif death and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
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

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

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
            position = 1
        elif short_trigger and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
        elif position == 1 and short_trigger:
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif position == -1 and long_trigger:
            signals.at[df.index[i], 'short_exit'] = True
            position = 0

    return signals


def mean_reversion_strategy(df, lookback=20, std_dev=2.0):
    """
    均值回归策略
    """
    df = prepare_dataframe(df)

    # 计算均值和标准差
    df['mean_price'] = df['Close'].rolling(lookback).mean()
    df['std_price'] = df['Close'].rolling(lookback).std()
    df['upper_band'] = df['mean_price'] + std_dev * df['std_price']
    df['lower_band'] = df['mean_price'] - std_dev * df['std_price']

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(lookback, len(df)):
        price = df['Close'].iloc[i]
        upper = df['upper_band'].iloc[i]
        lower = df['lower_band'].iloc[i]
        mean = df['mean_price'].iloc[i]

        # 均值回归信号
        if price < lower and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
        elif price > upper and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
        elif position == 1 and price >= mean:
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif position == -1 and price <= mean:
            signals.at[df.index[i], 'short_exit'] = True
            position = 0

    return signals


def boll_zscore_slope_accel_strategy(
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
    BOLL + Z-score + 均值斜率 + 斜率加速度 策略（均值回归增强版）

    指标构造：
    - BOLL：使用 `bb_len` 的滚动均值（中轨）与标准差形成上下轨
    - Z-score：z = (Close - 中轨) / 标准差，用于衡量偏离强度
    - 均值斜率：中轨的 pct_change 的滚动均值（`slope_len`），用于过滤强趋势
    - 斜率加速度：均值斜率的一阶差分，用于捕捉“趋势减速/拐头”的时刻

    交易逻辑（对称）：
    - 做多：价格下破下轨 + Z-score 低于阈值 + 均值斜率不极端 + 斜率加速度转正
    - 做空：价格上破上轨 + Z-score 高于阈值 + 均值斜率不极端 + 斜率加速度转负
    - 平仓：回到中轨附近（或 z-score 回归到退出阈值）
    """
    df = prepare_dataframe(df)

    if bb_std <= 0:
        raise ValueError("bb_std must be > 0")

    df["BBU"], df["BBM"], df["BBL"] = calculate_bollinger_bands(df["Close"], bb_len, bb_std)
    df["BB_STD"] = df["Close"].rolling(window=bb_len).std()
    df["Z"] = (df["Close"] - df["BBM"]) / df["BB_STD"].replace(0, float("nan"))

    df["mean_slope"] = df["BBM"].pct_change().rolling(window=slope_len).mean()
    df["slope_accel"] = df["mean_slope"].diff()

    signals = pd.DataFrame(index=df.index)
    signals[["long_entry", "long_exit", "short_entry", "short_exit"]] = False

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
                position = 1
            elif close > upper and z >= z_entry and flat_enough and slope_accel < -accel_threshold:
                signals.at[df.index[i], "short_entry"] = True
                position = -1
        elif position == 1:
            if close >= middle or z >= -z_exit:
                signals.at[df.index[i], "long_exit"] = True
                position = 0
        elif position == -1:
            if close <= middle or z <= z_exit:
                signals.at[df.index[i], "short_exit"] = True
                position = 0

    return signals


def momentum_strategy(df, roc_period=10, threshold=0.02):
    """
    动量策略
    """
    df = prepare_dataframe(df)

    # 计算变化率
    df['ROC'] = df['Close'].pct_change(roc_period)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(roc_period, len(df)):
        roc = df['ROC'].iloc[i]

        # 动量信号
        if roc > threshold and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
        elif roc < -threshold and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
        elif position == 1 and roc < 0:
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif position == -1 and roc > 0:
            signals.at[df.index[i], 'short_exit'] = True
            position = 0

    return signals


def macd_strategy(df, fast=12, slow=26, signal=9):
    """
    MACD策略
    """
    df = prepare_dataframe(df)

    # 计算MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'], fast, slow, signal)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(1, len(df)):
        if pd.isna(df['MACD_Hist'].iloc[i]) or pd.isna(df['MACD_Hist'].iloc[i-1]):
            continue

        prev_hist = df['MACD_Hist'].iloc[i-1]
        curr_hist = df['MACD_Hist'].iloc[i]

        # MACD信号
        if prev_hist < 0 and curr_hist >= 0 and position == 0:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
        elif prev_hist > 0 and curr_hist <= 0 and position == 0:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
        elif position == 1 and curr_hist < 0:
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif position == -1 and curr_hist > 0:
            signals.at[df.index[i], 'short_exit'] = True
            position = 0

    return signals


# 策略字典
STRATEGIES = {
    'RSI反转策略': rsi_reversal_strategy,
    '趋势ATR策略': trend_atr_signal,
    '布林RSI策略': boll_rsi_signal,
    '布林中轨突破平仓': boll_mid_break_exit_signal,
    '布林中轨突破平仓_弱趋势过滤': boll_mid_break_exit_filter_signal,
    '趋势波动止损策略': trend_volatility_stop_signal,
    '突破策略': breakout_strategy,
    '均值回归策略': mean_reversion_strategy,
    'BOLL+Z-score斜率加速度策略': boll_zscore_slope_accel_strategy,
    '动量策略': momentum_strategy,
    'MACD策略': macd_strategy
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
