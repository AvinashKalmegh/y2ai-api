#!/usr/bin/env python3
"""
ARGUS+Y2AI Test Runner

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --quick      # Run fast tests only (skip slow)
    python run_tests.py --verbose    # Extra verbose output
"""

import subprocess
import sys
import argparse


def run_tests(args):
    """Run pytest with specified options"""
    
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    
    # Skip slow tests
    if args.quick:
        cmd.extend(["-m", "not slow"])
    
    # Coverage
    if args.coverage:
        cmd.extend([
            "--cov=shared",
            "--cov=argus1", 
            "--cov=y2ai",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html"
        ])
    
    # Verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Specific test file
    if args.file:
        cmd.append(args.file)
    
    # Output
    if not args.quiet:
        cmd.append("--tb=short")
    else:
        cmd.append("--tb=no")
    
    # Run
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="ARGUS+Y2AI Test Runner")
    
    parser.add_argument("--unit", action="store_true", 
                        help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests only")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true",
                        help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true",
                        help="Extra verbose output")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--file", type=str,
                        help="Run specific test file")
    
    args = parser.parse_args()
    
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
