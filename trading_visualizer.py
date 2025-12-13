# ==============================
# Trading Data Visualization Component (English Version)
# ==============================
import platform
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 静音中文字体缺失告警，避免回测批量生成图表时刷屏
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# Set font based on operating system
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
elif system == 'Windows':  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

# Set style
try:
    sns.set_style('darkgrid')
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    # If seaborn version is old, use default style
    sns.set_style('darkgrid')
    plt.style.use('seaborn-darkgrid')


class TradingVisualizer:
    """Trading Data Visualization Component"""

    def __init__(self, figsize=(15, 10), dpi=100):
        """
        Initialize visualization component

        Args:
            figsize: Chart size, default (15, 10)
            dpi: Chart resolution, default 100
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'buy': '#2E7D32',      # Green - Buy
            'sell': '#C62828',     # Red - Sell
            'long': '#1E88E5',     # Blue - Long
            'short': '#FB8C00',    # Orange - Short
            'background': '#FAFAFA'
        }

    def plot_kline_with_trades(self, price_data, trades_data, title="K-line Chart with Trading Points", save_path=None):
        """
        Plot K-line chart with trading points

        Args:
            price_data: Price data DataFrame, needs datetime, open, high, low, close columns
            trades_data: Trading data DataFrame, needs datetime, action, price, type columns
            title: Chart title
            save_path: Save path, if None then don't save

        Returns:
            matplotlib Figure object
        """
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=self.colors['background'])

        # Process time data
        price_data = price_data.copy()

        # Standardize column names (handle case)
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

        # Rename columns并去重
        price_data = price_data.rename(columns=column_mapping)
        price_data = price_data.loc[:, ~price_data.columns.duplicated()]

        # Check required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in price_data.columns]
        if missing_cols:
            raise ValueError(f"Price data missing required columns: {missing_cols}")

        price_data['datetime'] = pd.to_datetime(price_data['datetime'])
        price_data.set_index('datetime', inplace=True)

        # Remove null rows
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
        Plot equity curve

        Args:
            equity_data: Equity data DataFrame, needs datetime, equity columns
            benchmark_data: Benchmark data DataFrame (optional), needs datetime, equity columns
            title: Chart title
            save_path: Save path, if None then don't save

        Returns:
            matplotlib Figure object
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
        Calculate equity curve statistics

        Args:
            equity_data: Equity data DataFrame

        Returns:
            Formatted statistics string
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
        Plot backtest results summary

        Args:
            backtest_results: Backtest results dictionary
            save_path: Save path

        Returns:
            matplotlib Figure object
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

        # Create subplots
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
        ax3.set_title('Max Drawdown Comparison', fontsize=14, fontweight='bold')
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


# Example usage
if __name__ == "__main__":
    # Create visualizer
    visualizer = TradingVisualizer()

    # Example data
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    base_price = 100

    # Generate price data
    price_changes = np.random.randn(100) * 2
    price_data = pd.DataFrame({
        'datetime': dates,
        'open': base_price + np.cumsum(price_changes),
        'high': base_price + np.cumsum(price_changes) + np.random.uniform(0, 5, 100),
        'low': base_price + np.cumsum(price_changes) - np.random.uniform(0, 5, 100),
        'close': base_price + np.cumsum(price_changes) + np.random.uniform(-2, 2, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Ensure high >= max(open, close) and low <= min(open, close)
    price_data['high'] = np.maximum(price_data['high'],
                                    np.maximum(price_data['open'], price_data['close']))
    price_data['low'] = np.minimum(price_data['low'],
                                   np.minimum(price_data['open'], price_data['close']))

    # Generate trades data
    trade_dates = dates[::20]  # Every 20 hours
    num_trades = len(trade_dates)
    actions = ['BUY', 'SELL'] * (num_trades // 2)
    if num_trades % 2 == 1:
        actions.append('BUY')
    actions = actions[:num_trades]

    trades_data = pd.DataFrame({
        'datetime': trade_dates,
        'action': actions,
        'price': price_data['close'].iloc[::20][:num_trades],
        'type': ['Long' if a == 'BUY' else 'Short' for a in actions]
    })

    equity_data = pd.DataFrame({
        'datetime': dates,
        'equity': 10000 + np.cumsum(np.random.randn(100) * 50)
    })

    # Draw K-line chart
    visualizer.plot_kline_with_trades(
        price_data=price_data,
        trades_data=trades_data,
        title="Sample K-line Chart",
        save_path="sample_kline_en.png"
    )

    # Draw equity curve
    visualizer.plot_equity_curve(
        equity_data=equity_data,
        title="Sample Equity Curve",
        save_path="sample_equity_en.png"
    )

    plt.show()
