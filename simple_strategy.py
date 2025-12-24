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


def ema_mean_reversion_strategy(df, lookback=30, std_dev=2.0):
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


# 策略字典
STRATEGIES = {
    'RSI反转策略': rsi_reversal_strategy,
    '趋势ATR策略': trend_atr_signal,
    '布林RSI策略': boll_rsi_signal,
    '趋势波动止损策略': trend_volatility_stop_signal,
    '突破策略': breakout_strategy,
    '均值回归策略': mean_reversion_strategy,
    'EMA均值回归策略': ema_mean_reversion_strategy,
    'RiskControlledMeanReversion': risk_controlled_mean_reversion,
    '动量策略': momentum_strategy,
    'MACD策略': macd_strategy,
    'IntradaySeasonality': intraday_seasonality_strategy,
    'VolatilitySqueezeBreakout': volatility_squeeze_breakout,
    'VolScaledMomentum': vol_scaled_momentum,
    'VWAPReversion': vwap_reversion
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
