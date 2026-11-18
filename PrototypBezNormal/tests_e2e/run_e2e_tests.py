#!/usr/bin/env python3
"""
run_e2e_tests.py
Standalone CLI test runner for PrototypBezNormal E2E Test Suite.

Features:
- Selectable execution tiers via --tier (e.g., --tier 1,2,3 or --tier 1)
- Full suite execution via --all
- Formatted console output with ANSI indicators
- Structured JSON test execution report via --output (default: PrototypBezNormal/output_smpc/e2e_test_report.json)
- Standard POSIX exit codes: 0 on success, 1 on test failure/error
- Also discoverable via: python -m unittest discover -s PrototypBezNormal/tests_e2e -p "test_*.py" -v
"""

import os
import sys

# Prevent OpenBLAS / OpenMP thread explosion when executing full E2E suite
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import datetime
import importlib
import json
import time
import unittest
from pathlib import Path
import torch

torch.set_num_threads(2)

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOTYPE_DIR = REPO_ROOT / "PrototypBezNormal"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

TESTS_E2E_DIR = PROTOTYPE_DIR / "tests_e2e"
if str(TESTS_E2E_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_E2E_DIR))

# Mapping from tier number to module and display name
TIER_MODULE_MAP = {
    1: ("PrototypBezNormal.tests_e2e.test_tier1_invariants", "Tier 1: Feature & Invariant Coverage (ADR-001)"),
    2: ("PrototypBezNormal.tests_e2e.test_tier2_components", "Tier 2: Component Contracts & Boundary Cases (SMPC)"),
    3: ("PrototypBezNormal.tests_e2e.test_tier3_workflows", "Tier 3: CLI Workflows & Hardware Device Guards"),
    4: ("PrototypBezNormal.tests_e2e.test_tier4_benchmarks", "Tier 4: Real-World Scenarios & Statistical Quality Gates"),
    5: ("PrototypBezNormal.tests_e2e.test_tier5_adversarial", "Tier 5: Adversarial Hardening & Stress Testing"),
}


class DetailedJSONTestResult(unittest.TextTestResult):
    """Custom TextTestResult collecting detailed execution records for JSON reporting."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_records = []
        self._test_start_time = None

    def startTest(self, test):
        super().startTest(test)
        self._test_start_time = time.perf_counter()

    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.perf_counter() - self._test_start_time if self._test_start_time else 0.0
        self.test_records.append({
            "test_id": test.id(),
            "description": test.shortDescription() or str(test),
            "status": "PASSED",
            "duration_s": round(duration, 4),
            "error": None,
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.perf_counter() - self._test_start_time if self._test_start_time else 0.0
        self.test_records.append({
            "test_id": test.id(),
            "description": test.shortDescription() or str(test),
            "status": "FAILED",
            "duration_s": round(duration, 4),
            "error": self._exc_info_to_string(err, test),
        })

    def addError(self, test, err):
        super().addError(test, err)
        duration = time.perf_counter() - self._test_start_time if self._test_start_time else 0.0
        self.test_records.append({
            "test_id": test.id(),
            "description": test.shortDescription() or str(test),
            "status": "ERROR",
            "duration_s": round(duration, 4),
            "error": self._exc_info_to_string(err, test),
        })

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.perf_counter() - self._test_start_time if self._test_start_time else 0.0
        self.test_records.append({
            "test_id": test.id(),
            "description": test.shortDescription() or str(test),
            "status": "SKIPPED",
            "duration_s": round(duration, 4),
            "reason": reason,
        })


def parse_tier_selection(tier_str: str) -> list[int]:
    """Parse comma-separated tier numbers (e.g., '1,2,3' -> [1, 2, 3])."""
    tiers = []
    for item in tier_str.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            val = int(item)
            if val in TIER_MODULE_MAP:
                tiers.append(val)
            else:
                print(f"Warning: Unknown tier {val}. Available tiers: {list(TIER_MODULE_MAP.keys())}")
        except ValueError:
            print(f"Warning: Invalid tier specification '{item}'")
    return sorted(list(set(tiers)))


def build_suite_for_tiers(selected_tiers: list[int]) -> unittest.TestSuite:
    """Build unified TestSuite containing all test cases from selected tiers."""
    master_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader

    for tier_num in selected_tiers:
        module_path, _ = TIER_MODULE_MAP[tier_num]
        try:
            mod = importlib.import_module(module_path)
            suite = loader.loadTestsFromModule(mod)
            master_suite.addTest(suite)
        except Exception as e:
            print(f"Error loading module {module_path}: {e}")
            raise

    return master_suite


def run_e2e_tests():
    parser = argparse.ArgumentParser(
        description="E2E Test Runner for PrototypBezNormal (Acoustic HDD Sentinel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Comma-separated tier list to execute (e.g. --tier 1,2 or --tier 1,2,3,4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute all test tiers (Tiers 1 through 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PrototypBezNormal/output_smpc/e2e_test_report.json",
        help="Path to save detailed JSON execution report (default: PrototypBezNormal/output_smpc/e2e_test_report.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Display detailed progress per test",
    )

    args = parser.parse_args()

    # Determine tiers to execute
    if args.tier:
        selected_tiers = parse_tier_selection(args.tier)
    elif args.all or not args.tier:
        selected_tiers = sorted(list(TIER_MODULE_MAP.keys()))
    else:
        selected_tiers = sorted(list(TIER_MODULE_MAP.keys()))

    print("=" * 80)
    print(" 🛡️  ACOUSTIC HDD SENTINEL - E2E MULTI-TIER TEST HARNESS")
    print("=" * 80)
    print(f" Active Workspace: {REPO_ROOT}")
    print(f" Execution Date:   {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(" Selected Tiers:")
    for t in selected_tiers:
        _, desc = TIER_MODULE_MAP[t]
        print(f"   - Tier {t}: {desc}")
    print("=" * 80 + "\n")

    suite = build_suite_for_tiers(selected_tiers)
    verbosity = 2 if args.verbose else 1

    # Run tests with custom TestResult collecting JSON records
    runner = unittest.TextTestRunner(verbosity=verbosity, resultclass=DetailedJSONTestResult)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed_time = time.perf_counter() - start_time

    total_run = result.testsRun
    failures_cnt = len(result.failures)
    errors_cnt = len(result.errors)
    skipped_cnt = len(result.skipped)
    passed_cnt = total_run - failures_cnt - errors_cnt - skipped_cnt
    is_success = (failures_cnt == 0 and errors_cnt == 0)

    print("\n" + "=" * 80)
    print(" E2E EXECUTION SUMMARY")
    print("=" * 80)
    print(f" Total Tests Run:  {total_run}")
    print(f" Passed:           {passed_cnt}")
    print(f" Failures:         {failures_cnt}")
    print(f" Errors:           {errors_cnt}")
    print(f" Skipped:          {skipped_cnt}")
    print(f" Execution Time:   {elapsed_time:.3f} s")
    print(f" Overall Status:   {'PASSED ✅' if is_success else 'FAILED ❌'}")
    print("=" * 80)

    # Save JSON report
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    test_records = getattr(result, "test_records", [])
    report_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASSED" if is_success else "FAILED",
        "selected_tiers": selected_tiers,
        "summary": {
            "total_tests": total_run,
            "passed": passed_cnt,
            "failures": failures_cnt,
            "errors": errors_cnt,
            "skipped": skipped_cnt,
            "elapsed_seconds": round(elapsed_time, 3),
        },
        "tests": test_records,
    }

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 Detailed JSON report written to: {report_path.resolve()}\n")
    except Exception as e:
        print(f"Warning: Could not write JSON report to {report_path}: {e}")

    # Return standard POSIX exit code
    sys.exit(0 if is_success else 1)


if __name__ == "__main__":
    run_e2e_tests()
