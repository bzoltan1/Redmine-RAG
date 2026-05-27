"""
core/timing.py — Lightweight timing and progress utilities.

Provides:
  - StageTimer   context manager that measures wall time for a named stage
  - ProgressBar  single-line overwriting progress bar with ETA
  - format_duration  human-readable duration string

All output goes to stdout so it stays in sync with pipeline log lines.
No external dependencies beyond the standard library.
"""

import sys
import time
from contextlib import contextmanager
from typing import Generator


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """
    Return a compact human-readable duration string.

    Examples
    --------
    >>> format_duration(0.4)
    '0.4s'
    >>> format_duration(75)
    '1m 15s'
    >>> format_duration(3725)
    '1h 2m 5s'
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------

class StageTimer:
    """
    Measures and reports wall-clock time for a named pipeline stage.

    Usage
    -----
    ::

        timer = StageTimer("Anonymization")
        timer.start()
        ... do work ...
        timer.stop()          # prints "  Anonymization: 1.2s"

    Or as a context manager::

        with StageTimer("Anonymization"):
            ... do work ...

    The timing summary is stored in ``timer.elapsed`` (seconds) after
    ``stop()`` / context exit so callers can include it in a final report.
    """

    def __init__(self, name: str, print_start: bool = False) -> None:
        self.name = name
        self.print_start = print_start
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def start(self) -> "StageTimer":
        if self.print_start:
            print(f"  [{self.name}] starting ...", flush=True)
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        print(f"  {self.name}: {format_duration(self.elapsed)}", flush=True)
        return self.elapsed

    def __enter__(self) -> "StageTimer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# ProgressBar
# ---------------------------------------------------------------------------

class ProgressBar:
    """
    Single-line overwriting progress bar with ETA.

    Writes to stdout using ``\\r`` so it updates in place without
    scrolling. Call ``close()`` (or use as a context manager) to
    print the final summary on a new line.

    Parameters
    ----------
    total:       Total number of items to process.
    label:       Short label shown before the bar (max ~20 chars).
    width:       Width of the ``[===   ]`` bar section in characters.
    unit:        Unit name appended to throughput figure (e.g. "chunks").
    print_every: Redraw frequency — update every N items.

    Usage
    -----
    ::

        bar = ProgressBar(total=1121, label="Embedding", unit="chunks")
        for i, chunk in enumerate(chunks, 1):
            process(chunk)
            bar.update(i)
        bar.close()

    Or as a context manager::

        with ProgressBar(total=100, label="Stage") as bar:
            for i in range(1, 101):
                do_work()
                bar.update(i)
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        width: int = 30,
        unit: str = "items",
        print_every: int = 1,
    ) -> None:
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.unit = unit
        self.print_every = max(1, print_every)
        self._start = time.perf_counter()
        self._last_drawn = -1
        self._closed = False

    def update(self, done: int) -> None:
        """Redraw the progress bar for *done* items completed."""
        if self._closed:
            return
        if done != self.total and done - self._last_drawn < self.print_every:
            return
        self._last_drawn = done
        self._draw(done)

    def _draw(self, done: int) -> None:
        elapsed = time.perf_counter() - self._start
        pct = done / self.total
        filled = int(self.width * pct)
        bar = "=" * filled + "-" * (self.width - filled)

        rate = done / elapsed if elapsed > 0 else 0.0
        if rate > 0 and done < self.total:
            eta = format_duration((self.total - done) / rate)
            eta_str = f" ETA {eta}"
        elif done >= self.total:
            eta_str = ""
        else:
            eta_str = " ETA --"

        throughput = f"{rate:.1f} {self.unit}/s" if rate > 0 else ""

        line = (
            f"\r  {self.label}: [{bar}] "
            f"{done}/{self.total} ({pct:.0%}) "
            f"{throughput}{eta_str}"
        )
        # Pad to overwrite any longer previous line
        sys.stdout.write(line.ljust(78) + "\r")
        sys.stdout.flush()

    def close(self, message: str | None = None) -> float:
        """
        Finish the bar and print a summary line.

        Returns total elapsed seconds.
        """
        if self._closed:
            return 0.0
        self._closed = True
        elapsed = time.perf_counter() - self._start
        rate = self.total / elapsed if elapsed > 0 else 0.0
        summary = message or (
            f"  {self.label}: {self.total} {self.unit} "
            f"in {format_duration(elapsed)} "
            f"({rate:.1f} {self.unit}/s)"
        )
        sys.stdout.write("\r" + " " * 79 + "\r")  # clear the bar line
        print(summary, flush=True)
        return elapsed

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *_) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------------------
# PipelineReport
# ---------------------------------------------------------------------------

class PipelineReport:
    """
    Accumulates per-stage timings and prints a summary table at the end.

    Usage
    -----
    ::

        report = PipelineReport("Ingest")
        report.record("Load JSON", 0.3)
        report.record("Chunking", 0.1)
        report.record("Embedding + ingest", 281.4)
        report.print()
    """

    def __init__(self, title: str = "Pipeline") -> None:
        self.title = title
        self._stages: list[tuple[str, float]] = []
        self._wall_start = time.perf_counter()

    def record(self, name: str, elapsed: float) -> None:
        """Add a completed stage with its elapsed time."""
        self._stages.append((name, elapsed))

    def print(self) -> None:
        """Print the timing summary table."""
        total_wall = time.perf_counter() - self._wall_start
        sep = "=" * 55

        print(f"\n{sep}")
        print(f"  {self.title} — timing summary")
        print(sep)

        total_measured = sum(e for _, e in self._stages)
        for name, elapsed in self._stages:
            pct = elapsed / total_measured * 100 if total_measured > 0 else 0
            bar = "#" * int(pct / 5)  # one # per 5%
            print(f"  {name:<28} {format_duration(elapsed):>8}  {bar}")

        print(sep)
        print(f"  {'Total wall time':<28} {format_duration(total_wall):>8}")
        print(sep)
