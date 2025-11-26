import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={
        'ts':       'Timestamp',
        'timestamp':'Timestamp',
        'open':     'Open',
        'high':     'High',
        'low':      'Low',
        'close':    'Close',
        'volume':   'Volume'
    }, inplace=True)

    # 如果有datetime列，直接使用；否则使用timestamp列
    if 'datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['datetime'])
    elif 'Timestamp' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Timestamp'])
    else:
        raise ValueError("数据文件中没有找到时间列 (datetime或timestamp)")

    df.sort_values('Datetime', inplace=True)
    df.drop_duplicates('Datetime', keep='first', inplace=True)
    df.set_index('Datetime', inplace=True)
    return df


def save_data_to_csv(df, filename):
    """
    保存数据到CSV文件
    """
    try:
        df.to_csv(filename, index=False)
        print(f"数据已保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存数据失败: {e}")
        return False


def load_data_from_csv(filename):
    """
    从CSV文件加载数据
    """
    try:
        df = pd.read_csv(filename)
        df['ts'] = pd.to_datetime(df['ts'])
        print(f"从 {filename} 加载了 {len(df)} 条数据")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

if __name__ == '__main__':
    file_path = '../stock_data/BTCUSDT_15m.csv'
    df = load_and_prepare_data(file_path)
    save_data_to_csv(df, "../stock_data/BTCUSDT_15m.csv")
