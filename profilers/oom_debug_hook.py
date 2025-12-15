#!/usr/bin/env python3
"""
OOM Debugging Hook: Reports RAM usage and execution time when each line is first executed.

Usage:
    python oom_debug_hook.py <script.py>          # For file execution
    python oom_debug_hook.py <module.path>        # For module execution (with -m)

Requirements:
    pip install psutil

Notes:
    - Tracks all code in the repo (excludes .venv and site-packages)
    - DDP-aware: only rank 0 prints by default (set OOM_DBG_ALL_RANKS=1 to print from all)
"""

import sys
import time
import psutil
import os
import runpy


def _find_repo_root(start_path):
    """Find repo root by looking for .git directory."""
    path = os.path.abspath(start_path)
    while path != '/':
        if os.path.isdir(os.path.join(path, '.git')):
            return path
        path = os.path.dirname(path)
    return None


def _is_repo_code(filename, repo_root):
    """Check if file is in repo and not in .venv or site-packages."""
    if repo_root is None:
        return False
    abs_path = os.path.abspath(filename)
    if not abs_path.startswith(repo_root):
        return False
    # Exclude virtual environments and installed packages
    rel_path = abs_path[len(repo_root):]
    excluded = ('/.venv/', '/site-packages/', '/__pycache__/')
    return not any(exc in rel_path for exc in excluded)


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

    # Find repo root from target file location
    repo_root = _find_repo_root(target_file)
    if repo_root is None:
        print(f"Warning: Could not find repo root (.git), tracking only target file",
              file=sys.stderr)

    # Setup state
    seen_lines = {}  # (filename, lineno) -> mem_mb at first execution
    process = psutil.Process()
    start_time = time.perf_counter()

    def trace_func(frame, event, arg):
        if event != 'line':
            return trace_func

        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Track all repo code (not in .venv)
        if not _is_repo_code(filename, repo_root):
            return trace_func

        key = (filename, lineno)
        if key in seen_lines:
            return trace_func

        # Get memory before marking as seen
        mem_mb = process.memory_info().rss / (1024 * 1024)
        elapsed = time.perf_counter() - start_time
        seen_lines[key] = mem_mb

        # DDP: only print from rank 0 unless OOM_DBG_ALL_RANKS is set
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        print_all = os.environ.get('OOM_DBG_ALL_RANKS', '0') == '1'

        if rank == 0 or print_all:
            # Show relative path from repo root for cleaner output
            if repo_root:
                rel_path = os.path.relpath(filename, repo_root)
            else:
                rel_path = os.path.basename(filename)

            rank_prefix = f"[R{rank}]" if print_all else ""
            print(f"[OOM_DBG]{rank_prefix} {rel_path}:{lineno:4d} | "
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