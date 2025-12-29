from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    exit_price: float | None = None
    reason: str = ""


class StopLossPolicy(Protocol):
    def decide_exit(self, position: int, entry_price: float, current_price: float) -> ExitDecision: ...


@dataclass(frozen=True)
class NoStopLoss:
    def decide_exit(self, position: int, entry_price: float, current_price: float) -> ExitDecision:
        return ExitDecision(False)


@dataclass(frozen=True)
class LossPriceDiffStopLoss:
    """
    Stop loss rule (threshold-based, losing only):
    - Long: trigger when current_price <= entry_price - threshold, exit at entry_price - threshold
    - Short: trigger when current_price >= entry_price + threshold, exit at entry_price + threshold

    This makes the exported `abs_entry_exit_price_diff` stay at ~`threshold` (instead of often exceeding it
    due to bar-close checking / price gaps).
    """

    threshold: float = 500.0

    def decide_exit(self, position: int, entry_price: float, current_price: float) -> ExitDecision:
        if position == 1:
            stop_price = entry_price - self.threshold
            if current_price <= stop_price:
                return ExitDecision(True, exit_price=stop_price, reason=f"stop_loss_loss_price_diff_{self.threshold:g}")
            return ExitDecision(False)

        if position == -1:
            stop_price = entry_price + self.threshold
            if current_price >= stop_price:
                return ExitDecision(True, exit_price=stop_price, reason=f"stop_loss_loss_price_diff_{self.threshold:g}")
            return ExitDecision(False)

        return ExitDecision(False)
