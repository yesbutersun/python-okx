# ==============================
# 3. 策略信号生成 (可替换模块)
# ==============================
import pandas as pd


def rsi_reversal_strategy(df):
    """
    RSI反转策略信号生成器
    返回包含信号的DataFrame (long_entry, long_exit, short_entry, short_exit)
    """
    signals = pd.DataFrame(index=df.index)
    signals['long_entry'] = False
    signals['long_exit'] = False
    signals['short_entry'] = False
    signals['short_exit'] = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # RSI反转策略逻辑
        signals.at[row.name, 'long_entry'] = (prev['RSI_14'] < 30 and row['RSI_14'] >= 30)
        signals.at[row.name, 'long_exit'] = (row['RSI_14'] > 70)
        signals.at[row.name, 'short_entry'] = (prev['RSI_14'] > 70 and row['RSI_14'] <= 70)
        signals.at[row.name, 'short_exit'] = (row['RSI_14'] < 30)

    return signals

# 信号生成：趋势跟随 + ATR 动态止盈止损
def trend_atr_signal(df, short_ema=8, long_ema=21, atr_len=14,
                     tp_atr=2.0, sl_atr=1.0):
    """
    输入：原始df，包含Open/High/Low/Close/Volume
    输出：包含 long_entry/exit 和 short_entry/exit 的DataFrame
    """
    df = df.copy()
    # 1) 计算指标
    df['EMA_short'] = df['Close'].ewm(span=short_ema, adjust=False).mean()
    df['EMA_long']  = df['Close'].ewm(span=long_ema, adjust=False).mean()
    tr = ta.true_range(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = tr.rolling(atr_len).mean()

    signals = pd.DataFrame(index=df.index)
    signals['long_entry'] = False
    signals['long_exit'] = False
    signals['short_entry'] = False
    signals['short_exit'] = False

    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        atr   = df['ATR'].iat[i]
        prev_s, prev_l = df['EMA_short'].iat[i-1], df['EMA_long'].iat[i-1]
        cur_s,  cur_l  = df['EMA_short'].iat[i],   df['EMA_long'].iat[i]

        golden = (prev_s < prev_l) and (cur_s > cur_l)
        death  = (prev_s > prev_l) and (cur_s < cur_l)

        # TP/SL 价格
        tp_price = entry_price + tp_atr * atr if position == 1 else entry_price - tp_atr * atr
        sl_price = entry_price - sl_atr * atr if position == 1 else entry_price + sl_atr * atr

        # 止盈止损
        if position == 1 and not pd.isna(atr) and (price >= tp_price or price <= sl_price):
            signals.at[df.index[i], 'long_exit'] = True
            position = 0
        elif position == -1 and not pd.isna(atr) and (price <= tp_price or price >= sl_price):
            signals.at[df.index[i], 'short_exit'] = True
            position = 0

        # 金叉死叉开仓
        if golden:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price
        elif death:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            entry_price = price

    return signals


def boll_rsi_signal(df, bb_len=20, bb_std=2.0, rsi_len=14):
    """
    输入：原始 df，包含 Close
    输出：包含 long_entry/exit 和 short_entry/exit 的DataFrame
    """
    df = df.copy()

    # 1) 计算布林带与 %B
    df['BBU'] = ta.bbands(close=df['Close'], length=bb_len, std=bb_std).iloc[:, 0]
    df['BBM'] = ta.bbands(close=df['Close'], length=bb_len, std=bb_std).iloc[:, 1]
    df['BBL'] = ta.bbands(close=df['Close'], length=bb_len, std=bb_std).iloc[:, 2]
    df['pctB'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])

    # 2) 计算 RSI
    df['RSI'] = ta.rsi(close=df['Close'], length=rsi_len)

    #df.dropna(inplace=True)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        pctB = df['pctB'].iat[i]
        rsi = df['RSI'].iat[i]

        # 开平条件
        long_entry = (pctB < 0 and rsi < 30 and position == 0)
        long_exit = (pctB > 0.5 or rsi > 50) and position == 1
        short_entry = (pctB > 1 and rsi > 70 and position == 0)
        short_exit = (pctB < 0.5 or rsi < 50) and position == -1

        signals.iat[i, 0] = long_entry
        signals.iat[i, 1] = long_exit
        signals.iat[i, 2] = short_entry
        signals.iat[i, 3] = short_exit

        if long_entry:
            position = 1
        elif short_entry:
            position = -1
        elif long_exit or short_exit:
            position = 0

    return signals


def trend_volatility_stop_signal(df, short_ema=8, long_ema=21, atr_len=14, sl_atr=1.5):
    """
    输入：原始df，包含Open/High/Low/Close/Volume
    输出：包含 long_entry/exit 和 short_entry/exit 的DataFrame
    """
    df = df.copy()

    # 1) 计算指标
    df['EMA_short'] = df['Close'].ewm(span=short_ema, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_ema, adjust=False).mean()
    tr = ta.true_range(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = tr.rolling(atr_len).mean()

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        atr = df['ATR'].iat[i]
        prev_s, prev_l = df['EMA_short'].iat[i - 1], df['EMA_long'].iat[i - 1]
        cur_s, cur_l = df['EMA_short'].iat[i], df['EMA_long'].iat[i]

        golden = (prev_s < prev_l) and (cur_s > cur_l)
        death = (prev_s > prev_l) and (cur_s < cur_l)

        # 止损价格（基于 ATR）
        if position == 1:
            stop_loss = entry_price - sl_atr * atr
            if price <= stop_loss:
                signals.at[df.index[i], 'long_exit'] = True
                position = 0
        elif position == -1:
            stop_loss = entry_price + sl_atr * atr
            if price >= stop_loss:
                signals.at[df.index[i], 'short_exit'] = True
                position = 0

        # 趋势信号（金叉/死叉）
        if golden:
            signals.at[df.index[i], 'long_entry'] = True
            position = 1
            entry_price = price
        elif death:
            signals.at[df.index[i], 'short_entry'] = True
            position = -1
            entry_price = price

    return signals

def boll_rsi_reversal_signal(df, bb_len=20, bb_std=2.0, rsi_len=14):
    """
    布林带 + RSI 反转策略信号生成器
    输入：原始 df，包含 Close
    输出：包含 long_entry/exit 和 short_entry/exit 的DataFrame
    """
    df = df.copy()

    # 1) 计算布林带与 %B
    bbands = ta.bbands(close=df['Close'], length=bb_len, std=bb_std)
    df['BBU'] = bbands.iloc[:, 0]
    df['BBM'] = bbands.iloc[:, 1]
    df['BBL'] = bbands.iloc[:, 2]
    df['pctB'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])

    # 2) 计算 RSI
    df['RSI'] = ta.rsi(close=df['Close'], length=rsi_len)

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        pctB = df['pctB'].iat[i]
        rsi = df['RSI'].iat[i]

        # 开平条件
        long_entry = (pctB < 0 and rsi < 30 and position == 0)
        long_exit = (pctB > 0.5 or rsi > 50) and position == 1
        short_entry = (pctB > 1 and rsi > 70 and position == 0)
        short_exit = (pctB < 0.5 or rsi < 50) and position == -1

        signals.iat[i, 0] = long_entry
        signals.iat[i, 1] = long_exit
        signals.iat[i, 2] = short_entry
        signals.iat[i, 3] = short_exit

        if long_entry:
            position = 1
        elif short_entry:
            position = -1
        elif long_exit or short_exit:
            position = 0

    return signals



def breakout_multi_tf_filter_signal(df, lookback=20, tf='D'):
    """
    基于价格突破 + 多周期趋势过滤的交易信号生成器

    参数:
        df (pd.DataFrame): 包含 OHLC 数据
        lookback (int): 突破周期长度
        tf (str): 过滤所用的时间框架（如 'D' 表示日线）

    返回:
        pd.DataFrame: 包含 long_entry/long_exit/short_entry/short_exit 的信号表
    """
    df = df.copy()

    # 1) 计算突破信号所需指标
    df['high_lookback'] = df['Close'].rolling(lookback).max()
    df['low_lookback'] = df['Close'].rolling(lookback).min()

    signals = pd.DataFrame(index=df.index)
    signals[['long_entry', 'long_exit', 'short_entry', 'short_exit']] = False

    position = 0

    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        high_lb = df['high_lookback'].iat[i]
        low_lb = df['low_lookback'].iat[i]

        # 判断当前是否触发突破
        long_trigger = price > high_lb
        short_trigger = price < low_lb

        # 多周期过滤（示例：使用上一日 Close 判断趋势方向）
        prev_close = df['Close'].iat[i - 1]
        trend_direction = 1 if df['Close'].iat[i] > df['Close'].shift(1).iat[i] else -1

        # 只有在趋势方向一致时才允许交易
        if long_trigger and trend_direction == 1 and position == 0:
            signals.iat[i, 0] = True
            position = 1
        elif short_trigger and trend_direction == -1 and position == 0:
            signals.iat[i, 2] = True
            position = -1
        # 平仓条件（简单设定为反向突破）
        elif position == 1 and short_trigger:
            signals.iat[i, 1] = True
            position = 0
        elif position == -1 and long_trigger:
            signals.iat[i, 3] = True
            position = 0

    return signals


def multi_factor_scoring_signal(df, roc_period=12, stoch_period=14, sl_percent=0.02, tp_levels=[0.01, 0.02, 0.04]):
    """
    多因子打分策略信号生成器（pandas 版本）

    参数:
        df (pd.DataFrame): 包含 OHLC 数据
        roc_period (int): 动量周期长度
        stoch_period (int): 随机指标周期长度
        sl_percent (float): 固定百分比止损
        tp_levels (list): 止盈层级列表（如 [0.01, 0.02, 0.04]）

    返回:
        pd.DataFrame: 包含 long_entry/exit 和 short_entry/exit 的信号表
    """
    df = df.copy()

    # 1) 计算所需因子
    df['ROC'] = df['Close'].pct_change(roc_period)
    df['MACD_Hist'] = ta.macd(close=df['Close']).iloc[:, 2]
    stoch = ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], length=stoch_period)
    df['Stoch_K'] = stoch.iloc[:, 0]
    df['Stoch_D'] = stoch.iloc[:, 1]

    signals = pd.DataFrame(index=df.index, columns=['long_entry', 'long_exit', 'short_entry', 'short_exit'])
    signals[:] = False

    position = 0
    entry_price = None
    position_level = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # 初始化 score
        score = 0
        if row['ROC'] > 0:
            score += 1
        elif row['ROC'] < 0:
            score -= 1

        if row['MACD_Hist'] > 0:
            score += 1
        elif row['MACD_Hist'] < 0:
            score -= 1

        if row['Stoch_K'] > row['Stoch_D'] and prev_row['Stoch_K'] <= prev_row['Stoch_D']:
            score += 1
        elif row['Stoch_K'] < row['Stoch_D'] and prev_row['Stoch_K'] >= prev_row['Stoch_D']:
            score -= 1

        # 开仓逻辑
        if position == 0:
            if score >= 2:
                signals.iat[i, 0] = True
                position = 1
                entry_price = row['Close']
                position_level = score
            elif score <= -2:
                signals.iat[i, 2] = True
                position = -1
                entry_price = row['Close']
                position_level = abs(score)

        # 平仓逻辑
        elif position == 1:
            price = row['Close']
            if score == 0 or price <= entry_price * (1 - sl_percent):
                signals.iat[i, 1] = True
                position = 0
                position_level = 0
            elif price >= entry_price * (1 + tp_levels[0]) and position_level > 1:
                signals.iat[i, 1] = True  # 部分止盈卖出30%
                position_level -= 1
            elif price >= entry_price * (1 + tp_levels[1]) and position_level > 1:
                signals.iat[i, 1] = True  # 部分止盈卖出50%
                position_level -= 1
            elif price >= entry_price * (1 + tp_levels[2]):
                signals.iat[i, 1] = True
                position = 0
                position_level = 0

        elif position == -1:
            price = row['Close']
            if score == 0 or price >= entry_price * (1 + sl_percent):
                signals.iat[i, 3] = True
                position = 0
                position_level = 0
            elif price <= entry_price * (1 - tp_levels[0]) and position_level > 1:
                signals.iat[i, 3] = True  # 部分止盈买入平仓30%
                position_level -= 1
            elif price <= entry_price * (1 - tp_levels[1]) and position_level > 1:
                signals.iat[i, 3] = True  # 部分止盈买入平仓50%
                position_level -= 1
            elif price <= entry_price * (1 - tp_levels[2]):
                signals.iat[i, 3] = True
                position = 0
                position_level = 0

    return signals
