#!/usr/bin/env python3
"""
OOM Debugging Hook: Reports RAM usage and execution time when each line is first executed.

Usage:
    python oom_debug_hook.py <script.py>          # For file execution
    python oom_debug_hook.py <module.path>        # For module execution (with -m)

Requirements:
    pip install psutil
"""

import sys
import time
import psutil
import os
import runpy


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <script.py|module.path>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]
    is_module = not target.endswith('.py') and ('/' not in target or target.startswith('.'))

    if is_module:
        # Module path: find the actual file
        try:
            module_spec = __import__(target, fromlist=['__file__']).__spec__
            target_file = module_spec.origin
            if target_file is None or not target_file.endswith('.py'):
                print(f"Error: Cannot find .py file for module {target}", file=sys.stderr)
                sys.exit(1)
        except ImportError as e:
            print(f"Error: Cannot import module {target}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # File path
        target_file = os.path.abspath(target)
        if not os.path.exists(target_file):
            print(f"Error: File not found: {target_file}", file=sys.stderr)
            sys.exit(1)

    # Setup state
    seen_lines = set()
    process = psutil.Process()
    start_time = time.perf_counter()
    target_file_abs = os.path.abspath(target_file)

    def trace_func(frame, event, arg):
        if event != 'line':
            return trace_func

        filename = os.path.abspath(frame.f_code.co_filename)
        lineno = frame.f_lineno

        # Track only the target file
        if filename != target_file_abs:
            return trace_func

        key = (filename, lineno)
        if key in seen_lines:
            return trace_func

        seen_lines.add(key)

        # Report
        mem_mb = process.memory_info().rss / (1024 * 1024)
        elapsed = time.perf_counter() - start_time

        print(f"[OOM_DBG] {os.path.basename(filename)}:{lineno:4d} | "
              f"RAM: {mem_mb:8.1f} MB | Time: {elapsed:8.3f}s")

        return trace_func

    # Execute with tracing
    sys.settrace(trace_func)
    try:
        if is_module:
            # Run as module (preserves relative imports)
            runpy.run_module(target, run_name='__main__', alter_sys=True)
        else:
            # Run as file
            runpy.run_path(target_file, run_name='__main__')
    finally:
        sys.settrace(None)

if __name__ == '__main__':
    main()