"""
Real-time line-by-line profiler for monitoring execution.

Usage:
    from utils.line_profiler import profile_run

    # In your main script:
    if __name__ == "__main__":
        profile_run(main_function, *args, **kwargs)

    # Or use as decorator:
    @line_profile
    def my_function():
        ...
"""

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Callable, Any
import linecache
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
import threading


@dataclass
class LineStats:
    """Statistics for a single line of code"""
    hits: int = 0
    total_time: float = 0.0
    last_hit_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.hits if self.hits > 0 else 0.0


class RealtimeLineProfiler:
    """
    Line-by-line profiler that displays results in real-time.

    Shows:
    - Current line being executed
    - Number of hits per line
    - Cumulative time per line
    - Average time per line
    """

    def __init__(self, target_files: Optional[list[str]] = None,
                 update_interval: float = 0.5,
                 top_n: int = 30):
        """
        Args:
            target_files: List of file patterns to profile (e.g., ['final_pipeline.py'])
                         If None, profiles all files
            update_interval: How often to update the display (seconds)
            top_n: Show top N slowest lines
        """
        self.target_files = target_files or []
        self.update_interval = update_interval
        self.top_n = top_n

        # Stats: {(filename, lineno): LineStats}
        self.stats = defaultdict(LineStats)
        self.lock = Lock()

        # Current execution
        self.current_file: Optional[str] = None
        self.current_line: Optional[int] = None
        self.last_update = time.time()

        # Rich display
        self.console = Console()
        self.live: Optional[Live] = None

        # Timing
        self.start_time = time.time()
        self.last_line_time = time.time()

    def _should_profile(self, filename: str) -> bool:
        """Check if this file should be profiled"""
        if not self.target_files:
            # Skip standard library and site-packages
            return not ('site-packages' in filename or
                       'lib/python' in filename or
                       '<frozen' in filename)

        # Check if filename matches any target pattern
        filename_lower = filename.lower()
        return any(target.lower() in filename_lower
                  for target in self.target_files)

    def _trace_callback(self, frame, event, arg):
        """Callback for sys.settrace"""
        if event != 'line':
            return self._trace_callback

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        if not self._should_profile(filename):
            return self._trace_callback

        current_time = time.time()
        elapsed = current_time - self.last_line_time

        with self.lock:
            # Update stats for previous line
            if self.current_file and self.current_line:
                key = (self.current_file, self.current_line)
                stats = self.stats[key]
                stats.hits += 1
                stats.total_time += elapsed
                stats.last_hit_time = current_time

            # Update current line
            self.current_file = filename
            self.current_line = lineno
            self.last_line_time = current_time

        return self._trace_callback

    def _get_line_source(self, filename: str, lineno: int) -> str:
        """Get source code for a line"""
        try:
            line = linecache.getline(filename, lineno).strip()
            return line[:80] + '...' if len(line) > 80 else line
        except:
            return '<unavailable>'

    def _create_display(self) -> Layout:
        """Create the rich display layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="current", size=5),
            Layout(name="top_lines")
        )

        # Header
        elapsed = time.time() - self.start_time
        header_text = Text(f"Line Profiler - Running for {elapsed:.1f}s",
                          style="bold magenta")
        layout["header"].update(Panel(header_text))

        # Current line
        with self.lock:
            current_file = self.current_file
            current_line = self.current_line

        if current_file and current_line:
            filename = Path(current_file).name
            source = self._get_line_source(current_file, current_line)
            current_text = Text.assemble(
                ("Currently executing:\n", "bold cyan"),
                (f"{filename}:{current_line}\n", "yellow"),
                (source, "white")
            )
            layout["current"].update(Panel(current_text, title="Current Line"))
        else:
            layout["current"].update(Panel("No line executing", title="Current Line"))

        # Top slowest lines
        table = Table(title=f"Top {self.top_n} Slowest Lines (by total time)")
        table.add_column("File:Line", style="cyan", no_wrap=True)
        table.add_column("Hits", justify="right", style="green")
        table.add_column("Total (s)", justify="right", style="magenta")
        table.add_column("Avg (ms)", justify="right", style="yellow")
        table.add_column("Source", style="white")

        # Get top lines by total time
        with self.lock:
            sorted_stats = sorted(
                self.stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )[:self.top_n]

        for (filename, lineno), stats in sorted_stats:
            short_filename = Path(filename).name
            source = self._get_line_source(filename, lineno)

            table.add_row(
                f"{short_filename}:{lineno}",
                str(stats.hits),
                f"{stats.total_time:.3f}",
                f"{stats.avg_time * 1000:.2f}",
                source
            )

        layout["top_lines"].update(table)
        return layout

    def start(self):
        """Start profiling"""
        sys.settrace(self._trace_callback)
        self.start_time = time.time()
        self.last_line_time = time.time()

        # Start live display in a separate thread
        self.live = Live(self._create_display(), console=self.console,
                        refresh_per_second=1 / self.update_interval)
        self.live.start()

        # Update display periodically
        def update_loop():
            while self.live:
                time.sleep(self.update_interval)
                if self.live:
                    self.live.update(self._create_display())

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def stop(self):
        """Stop profiling"""
        sys.settrace(None)
        if self.live:
            self.live.update(self._create_display())
            self.live.stop()
            self.live = None

    def print_summary(self):
        """Print final summary"""
        self.console.print("\n[bold cyan]═" * 40)
        self.console.print("[bold cyan]Profiling Summary")
        self.console.print("[bold cyan]═" * 40)

        total_time = time.time() - self.start_time
        self.console.print(f"\n[yellow]Total execution time: {total_time:.2f}s")

        # Top lines by total time
        table = Table(title=f"Top {self.top_n} Slowest Lines")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("File:Line", style="cyan", no_wrap=True)
        table.add_column("Hits", justify="right", style="green")
        table.add_column("Total (s)", justify="right", style="magenta")
        table.add_column("% Total", justify="right", style="red")
        table.add_column("Avg (ms)", justify="right", style="yellow")
        table.add_column("Source", style="white")

        with self.lock:
            sorted_stats = sorted(
                self.stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )[:self.top_n]

        for rank, ((filename, lineno), stats) in enumerate(sorted_stats, 1):
            short_filename = Path(filename).name
            source = self._get_line_source(filename, lineno)
            pct = (stats.total_time / total_time) * 100

            table.add_row(
                str(rank),
                f"{short_filename}:{lineno}",
                str(stats.hits),
                f"{stats.total_time:.3f}",
                f"{pct:.1f}%",
                f"{stats.avg_time * 1000:.2f}",
                source
            )

        self.console.print(table)


def profile_run(func: Callable, *args,
                target_files: Optional[list[str]] = None,
                update_interval: float = 0.5,
                top_n: int = 30,
                **kwargs) -> Any:
    """
    Run a function with line profiling enabled.

    Args:
        func: Function to profile
        *args: Positional arguments for func
        target_files: List of file patterns to profile (e.g., ['final_pipeline.py'])
        update_interval: Display update interval in seconds
        top_n: Number of top lines to show
        **kwargs: Keyword arguments for func

    Returns:
        Result of func(*args, **kwargs)

    Example:
        >>> from utils.line_profiler import profile_run
        >>> def main():
        ...     pipeline = FinalPipeline(config)
        ...     pipeline.fit()
        >>>
        >>> if __name__ == "__main__":
        ...     profile_run(main, target_files=['final_pipeline.py'])
    """
    profiler = RealtimeLineProfiler(
        target_files=target_files,
        update_interval=update_interval,
        top_n=top_n
    )

    try:
        profiler.start()
        result = func(*args, **kwargs)
        return result
    finally:
        profiler.stop()
        profiler.print_summary()


def line_profile(target_files: Optional[list[str]] = None,
                 update_interval: float = 0.5,
                 top_n: int = 30):
    """
    Decorator for line profiling.

    Example:
        >>> from utils.line_profiler import line_profile
        >>>
        >>> @line_profile(target_files=['final_pipeline.py'])
        >>> def main():
        ...     pipeline = FinalPipeline(config)
        ...     pipeline.fit()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return profile_run(func, *args,
                             target_files=target_files,
                             update_interval=update_interval,
                             top_n=top_n,
                             **kwargs)
        return wrapper
    return decorator
