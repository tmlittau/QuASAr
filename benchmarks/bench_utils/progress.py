"""Light-weight textual progress reporting utilities for benchmark CLIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO
import sys


@dataclass
class ProgressReporter:
    """Render an inline progress bar for long-running benchmark scripts."""

    total: int
    prefix: str = ""
    bar_width: int = 30
    stream: IO[str] = field(default_factory=lambda: sys.stdout)

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 1)
        self._count = 0
        self._last_len = 0
        self._render("", done=False)

    # ------------------------------------------------------------------
    def advance(self, message: str = "") -> None:
        """Advance the bar by one step and display ``message``."""

        self._count += 1
        if self._count > self.total:
            self.total = self._count
        done = self._count >= self.total
        self._render(message, done=done)

    # ------------------------------------------------------------------
    def announce(self, message: str = "") -> None:
        """Display ``message`` without advancing the progress counter."""

        self._render(message, done=False)

    # ------------------------------------------------------------------
    def _render(self, message: str, *, done: bool) -> None:
        fraction = min(self._count / self.total, 1.0)
        filled = int(self.bar_width * fraction)
        bar = f"[{'#' * filled}{'-' * (self.bar_width - filled)}]"
        parts: list[str] = []
        if self.prefix:
            parts.append(self.prefix)
        parts.append(f"{bar} {self._count}/{self.total}")
        if message:
            parts.append(message)
        line = " ".join(parts)
        self._write(line, done=done)

    # ------------------------------------------------------------------
    def _write(self, text: str, *, done: bool = False) -> None:
        blank = "\r" + (" " * self._last_len) + "\r"
        self.stream.write(blank)
        self.stream.write(text)
        if done:
            self.stream.write("\n")
            self._last_len = 0
        else:
            self._last_len = len(text)
        self.stream.flush()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Ensure the current line is terminated with a newline."""

        if self._last_len:
            self.stream.write("\n")
            self.stream.flush()
            self._last_len = 0


__all__ = ["ProgressReporter"]

