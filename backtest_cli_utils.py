from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from enhanced_backtest import BacktestEngine
from stop_loss import LossPriceDiffStopLoss, NoStopLoss, StopLossPolicy


_DEFAULT_STOP_LOSS_POLICY = object()

DEFAULT_INITIAL_CAPITAL: float = 10000
DEFAULT_COMMISSION: float = 0.001
DEFAULT_SLIPPAGE: float = 0.0005
DEFAULT_STOP_LOSS_THRESHOLD: float = 500


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        time_range = f"{df['datetime'].min()} 至 {df['datetime'].max()}"
    else:
        time_range = "N/A"

    if "close" in df.columns:
        price_range = f"${df['close'].min():.2f} - ${df['close'].max():.2f}"
    else:
        price_range = "N/A"

    print("数据加载完成:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 时间范围: {time_range}")
    print(f"  - 价格范围: {price_range}")
    return df


def create_engine(
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    enable_visualization: bool = False,
    chart_dir: str = "charts",
    stop_loss_threshold: float = DEFAULT_STOP_LOSS_THRESHOLD,
    stop_loss_policy: StopLossPolicy | None | object = _DEFAULT_STOP_LOSS_POLICY,
) -> BacktestEngine:
    if stop_loss_policy is None:
        policy: StopLossPolicy = NoStopLoss()
    elif stop_loss_policy is _DEFAULT_STOP_LOSS_POLICY:
        # default: enable threshold-based stop loss
        policy = LossPriceDiffStopLoss(stop_loss_threshold)
    else:
        policy = stop_loss_policy  # type: ignore[assignment]
    return BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        enable_visualization=enable_visualization,
        chart_dir=chart_dir,
        stop_loss_policy=policy,
    )


def save_detailed_results(
    results: dict[str, Any],
    *,
    output_dir: str = "backtest_results",
    summary_filename: str = "summary.json",
    csv_encoding: str = "utf-8-sig",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for strategy_name, result in results.items():
        if not result:
            continue

        trades_df = result.get("trades")
        if trades_df is not None and len(trades_df) > 0:
            trades_df.to_csv(
                os.path.join(output_dir, f"{strategy_name}_trades.csv"),
                index=False,
                encoding=csv_encoding,
            )

        equity_df = result.get("equity_curve")
        if equity_df is not None and len(equity_df) > 0:
            equity_df.to_csv(
                os.path.join(output_dir, f"{strategy_name}_equity.csv"),
                index=False,
                encoding=csv_encoding,
            )

    summary: dict[str, Any] = {}
    for strategy_name, result in results.items():
        if result and result.get("stats"):
            summary[strategy_name] = result["stats"]

    with open(os.path.join(output_dir, summary_filename), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
