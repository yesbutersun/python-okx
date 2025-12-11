# ==============================
# 交易数据可视化组件
# ==============================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import platform
import warnings

# 抑制字体警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 根据操作系统设置中文字体
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'DejaVu Sans']
elif system == 'Windows':  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

# 设置样式
try:
    sns.set_style('darkgrid')
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    # 如果seaborn版本较旧，使用默认样式
    sns.set_style('darkgrid')
    plt.style.use('seaborn-darkgrid')


class TradingVisualizer:
    """交易数据可视化组件"""

    def __init__(self, figsize=(15, 10), dpi=100):
        """
        初始化可视化组件

        Args:
            figsize: 图表大小，默认 (15, 10)
            dpi: 图表分辨率，默认 100
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'buy': '#2E7D32',      # 绿色 - 买入
            'sell': '#C62828',     # 红色 - 卖出
            'long': '#1E88E5',     # 蓝色 - 多头
            'short': '#FB8C00',    # 橙色 - 空头
            'background': '#FAFAFA'
        }

    def plot_kline_with_trades(self, price_data, trades_data, title="K-line Chart with Trading Points", save_path=None):
        """
        绘制K线图和交易点

        Args:
            price_data: 价格数据 DataFrame，需包含 datetime, open, high, low, close 列
            trades_data: 交易数据 DataFrame，需包含 datetime, action, price, type 列
            title: 图表标题
            save_path: 保存路径，如 None 则不保存

        Returns:
            matplotlib Figure 对象
        """
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=self.colors['background'])

        # 处理时间数据
        price_data = price_data.copy()

        # 标准化列名（处理大小写）
        column_mapping = {
            'datetime': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        }

        # 重命名列
        price_data = price_data.rename(columns=column_mapping)

        # 检查必要列
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in price_data.columns]
        if missing_cols:
            raise ValueError(f"价格数据缺少必要列: {missing_cols}")

        price_data['datetime'] = pd.to_datetime(price_data['datetime'])
        price_data.set_index('datetime', inplace=True)

        # 删除空值行
        price_data = price_data.dropna(subset=['open', 'high', 'low', 'close'])

        # Draw K-line chart
        for i in range(len(price_data)):
            date = price_data.index[i]
            open_price = price_data['open'].iloc[i]
            high_price = price_data['high'].iloc[i]
            low_price = price_data['low'].iloc[i]
            close_price = price_data['close'].iloc[i]

            # Determine color (red for up, green for down)
            color = 'red' if close_price >= open_price else 'green'

            # Draw candle body
            ax1.bar(date, abs(close_price - open_price), bottom=min(open_price, close_price),
                   width=0.0008, color=color, alpha=0.8, edgecolor='none')

            # Draw wicks
            ax1.vlines(date, low_price, high_price, color=color, linewidth=1, alpha=0.8)

        # Draw trading points
        if trades_data is not None and len(trades_data) > 0:
            trades_data = trades_data.copy()
            trades_data['datetime'] = pd.to_datetime(trades_data['datetime'])

            # Separate buy and sell signals
            buy_trades = trades_data[trades_data['action'].isin(['BUY', 'BUY_TO_COVER'])]
            sell_trades = trades_data[trades_data['action'].isin(['SELL', 'SELL_SHORT'])]

            # Draw buy points
            if len(buy_trades) > 0:
                ax1.scatter(buy_trades['datetime'], buy_trades['price'],
                          color=self.colors['buy'], marker='^', s=100,
                          label='Buy', zorder=5, alpha=0.9)

            # Draw sell points
            if len(sell_trades) > 0:
                ax1.scatter(sell_trades['datetime'], sell_trades['price'],
                          color=self.colors['sell'], marker='v', s=100,
                          label='Sell', zorder=5, alpha=0.9)

        # Set main chart style
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(price_data)//20)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Draw volume chart (if available)
        if 'volume' in price_data.columns:
            ax2.bar(price_data.index, price_data['volume'], width=0.0008,
                   color='gray', alpha=0.6)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(price_data)//20)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.axis('off')

        plt.tight_layout()

        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"K-line chart saved to: {save_path}")

        return fig

    def plot_equity_curve(self, equity_data, benchmark_data=None, title="Equity Curve", save_path=None):
        """
        绘制权益曲线图

        Args:
            equity_data: 权益数据 DataFrame，需包含 datetime, equity 列
            benchmark_data: 基准数据 DataFrame（可选），需包含 datetime, equity 列
            title: 图表标题
            save_path: 保存路径，如 None 则不保存

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi,
                               facecolor=self.colors['background'])

        # Process data
        equity_data = equity_data.copy()
        equity_data['datetime'] = pd.to_datetime(equity_data['datetime'])

        # Calculate returns
        initial_equity = equity_data['equity'].iloc[0]
        equity_data['return_pct'] = (equity_data['equity'] - initial_equity) / initial_equity * 100

        # Draw strategy equity curve
        ax.plot(equity_data['datetime'], equity_data['return_pct'],
               color=self.colors['long'], linewidth=2, label='Strategy Returns', alpha=0.9)

        # Draw benchmark curve (if available)
        if benchmark_data is not None:
            benchmark_data = benchmark_data.copy()
            benchmark_data['datetime'] = pd.to_datetime(benchmark_data['datetime'])
            benchmark_initial = benchmark_data['equity'].iloc[0]
            benchmark_data['return_pct'] = (benchmark_data['equity'] - benchmark_initial) / benchmark_initial * 100
            ax.plot(benchmark_data['datetime'], benchmark_data['return_pct'],
                   color='gray', linewidth=1, label='Benchmark', alpha=0.7, linestyle='--')

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        # Mark highest and lowest points
        max_return = equity_data['return_pct'].max()
        min_return = equity_data['return_pct'].min()
        max_date = equity_data.loc[equity_data['return_pct'].idxmax(), 'datetime']
        min_date = equity_data.loc[equity_data['return_pct'].idxmin(), 'datetime']

        ax.scatter(max_date, max_return, color='green', s=100, zorder=5,
                  marker='o', label=f'High: {max_return:.2f}%')
        ax.scatter(min_date, min_return, color='red', s=100, zorder=5,
                  marker='o', label=f'Low: {min_return:.2f}%')

        # Set chart style
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(equity_data)//20)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add statistics box
        stats_text = self._calculate_equity_stats(equity_data)
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Equity curve chart saved to: {save_path}")

        return fig

    def _calculate_equity_stats(self, equity_data):
        """
        计算权益曲线统计数据

        Args:
            equity_data: 权益数据 DataFrame

        Returns:
            格式化的统计信息字符串
        """
        if len(equity_data) == 0:
            return "No data"

        # Ensure required columns exist
        if 'equity' not in equity_data.columns:
            return "Data error: missing equity column"

        # Calculate total return
        initial_equity = equity_data['equity'].iloc[0]
        final_equity = equity_data['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # Calculate maximum drawdown
        equity_data_copy = equity_data.copy()
        equity_data_copy['cummax'] = equity_data_copy['equity'].cummax()
        equity_data_copy['drawdown'] = (equity_data_copy['equity'] - equity_data_copy['cummax']) / equity_data_copy['cummax'] * 100
        max_drawdown = equity_data_copy['drawdown'].min()

        # Calculate daily returns
        equity_data_copy['daily_return'] = equity_data_copy['equity'].pct_change() * 100
        daily_returns = equity_data_copy['daily_return'].dropna()

        if len(daily_returns) > 0:
            volatility = daily_returns.std()
            sharpe_ratio = daily_returns.mean() / volatility * np.sqrt(365) if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0

        stats_text = f"""Total Return: {total_return:.2f}%
Max Drawdown: {max_drawdown:.2f}%
Annual Volatility: {volatility:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}"""

        return stats_text

    def plot_backtest_summary(self, backtest_results, save_path=None):
        """
        绘制回测结果汇总图

        Args:
            backtest_results: 回测结果字典
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        # Prepare data
        strategies = []
        returns = []
        sharpes = []
        drawdowns = []
        win_rates = []

        for strategy_name, result in backtest_results.items():
            if result and result['stats']:
                strategies.append(strategy_name)
                stats = result['stats']
                returns.append(stats['total_return_pct'])
                sharpes.append(stats['sharpe_ratio'])
                drawdowns.append(abs(stats['max_drawdown_pct']))
                win_rates.append(stats['win_rate'] * 100)

        if not strategies:
            print("No backtest results available")
            return None

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15),
                                                      dpi=self.dpi,
                                                      facecolor=self.colors['background'])

        # Returns comparison
        bars1 = ax1.bar(strategies, returns, color=[self.colors['long'] if r > 0 else self.colors['sell'] for r in returns])
        ax1.set_title('Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        # Sharpe ratio comparison
        bars2 = ax2.bar(strategies, sharpes, color=self.colors['long'])
        ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe=1')
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')

        # Max drawdown comparison
        bars3 = ax3.bar(strategies, drawdowns, color=self.colors['sell'])
        ax3.set_title('Max Drawdown Comparison', fontsize=14, fontweight='bold)
        ax3.set_ylabel('Max Drawdown (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')

        # Win rate comparison
        bars4 = ax4.bar(strategies, win_rates, color=self.colors['buy'])
        ax4.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Win Rate=50%')
        ax4.legend()
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Backtest summary chart saved to: {save_path}")

        return fig


# 示例使用方法
if __name__ == "__main__":
    # 创建可视化器
    visualizer = TradingVisualizer()

    # 示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    price_data = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    trades_data = pd.DataFrame({
        'datetime': dates[::20],
        'action': ['BUY', 'SELL'] * 5,
        'price': np.random.randn(10).cumsum() + 100,
        'type': ['Long', 'Short'] * 5
    })

    equity_data = pd.DataFrame({
        'datetime': dates,
        'equity': np.random.randn(100).cumsum() + 10000
    })

    # 绘制K线图
    visualizer.plot_kline_with_trades(price_data, trades_data,
                                      title="示例K线图",
                                      save_path="sample_kline.png")

    # 绘制收益曲线
    visualizer.plot_equity_curve(equity_data,
                                 title="示例收益曲线",
                                 save_path="sample_equity.png")

    plt.show()