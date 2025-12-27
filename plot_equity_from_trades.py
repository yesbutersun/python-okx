#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def configure_matplotlib_fonts() -> str | None:
    """
    Best-effort configure a CJK-capable font to avoid "Glyph missing from font" warnings on Windows.
    Returns the chosen font name (or None if not found).
    """
    import matplotlib
    from matplotlib import font_manager

    preferred_fonts = [
        "Microsoft YaHei",  # 微软雅黑
        "SimHei",  # 黑体
        "SimSun",  # 宋体
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font_name

    return None


def _read_trades_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def equity_curve_from_trades(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if "datetime" not in trades.columns:
        raise ValueError("trades csv missing required column: datetime")

    trades = trades.copy()
    trades["datetime"] = pd.to_datetime(trades["datetime"])
    trades = trades.sort_values("datetime")

    if "pnl" not in trades.columns:
        raise ValueError("trades csv missing required column: pnl")

    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["equity"] = initial_capital + trades["pnl"].cumsum()
    return trades[["datetime", "equity"]]


def plot_equity_from_trades_csvs(
    trades_csv_paths: list[Path],
    *,
    initial_capital: float = 10000.0,
    out: Path = Path("equity_from_trades.png"),
    title: str = "Equity Curve (from trades pnl)",
) -> None:
    chosen_font = configure_matplotlib_fonts()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for path in trades_csv_paths:
        df = _read_trades_csv(path)
        curve = equity_curve_from_trades(df, initial_capital)
        label = path.stem
        if label.endswith("_trades"):
            label = label[: -len("_trades")]
        plt.plot(curve["datetime"], curve["equity"], label=label, linewidth=1.5)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    if chosen_font:
        print(f"Font: {chosen_font}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    INITIAL_CAPITAL = 10000.0
    OUT_PATH = Path("charts/equity.png")
    TITLE = "Equity Curve (from trades pnl)"

    TRADE_FILES = [
        Path("backtest_results/EMA均值回归策略_trades.csv"),
        Path("backtest_results/MACD策略_trades.csv"),
    ]

    plot_equity_from_trades_csvs(
        TRADE_FILES,
        initial_capital=INITIAL_CAPITAL,
        out=OUT_PATH,
        title=TITLE,
    )
