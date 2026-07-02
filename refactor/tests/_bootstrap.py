"""Test bootstrap: put ``refactor/`` on sys.path and provide a tiny runner.

These tests run WITHOUT pytest (it is not a dependency here). Each test file
defines ``test_*`` functions and calls ``run_tests(globals())`` under
``__main__``; pytest can still discover them if it is installed.
"""
from __future__ import annotations

import os
import sys
import traceback


def add_path() -> None:
    refactor_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if refactor_dir not in sys.path:
        sys.path.insert(0, refactor_dir)


def run_tests(ns: dict) -> int:
    """Run every ``test_*`` callable in ``ns``; return process exit code."""
    tests = sorted((n, f) for n, f in ns.items()
                   if n.startswith("test_") and callable(f))
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1
