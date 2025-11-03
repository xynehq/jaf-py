"""
A2A Memory Comprehensive Test Runner

Executes the complete A2A memory test suite across all providers and generates
detailed validation reports. This is the main entry point for running the
"crucible" test suite.

Usage:
    python run_comprehensive_tests.py [options]

Options:
    --providers: Comma-separated list of providers to test (memory,redis,postgres)
    --coverage: Generate code coverage report
    --performance: Include performance benchmarks
    --stress: Include stress tests (may take longer)
    --report: Generate detailed validation report
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.types import A2AInMemoryTaskConfig


@dataclass
class TestResult:
    """Test result information"""

    test_name: str
    status: str  # "PASSED", "FAILED", "SKIPPED", "ERROR"
    duration_ms: float
    error_message: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Results for an entire test suite"""

    suite_name: str
    provider: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    results: List[TestResult] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""

    timestamp: str
    total_duration_ms: float
    providers_tested: List[str]
    suite_results: List[TestSuiteResult] = field(default_factory=list)
    coverage_report: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    final_verdict: str = ""


class ComprehensiveTestRunner:
    """Main test runner for A2A memory system validation"""

    def __init__(self, args):
        self.args = args
        self.providers = self._parse_providers(args.providers)
        self.report = ValidationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            total_duration_ms=0.0,
            providers_tested=self.providers,
        )

    def _parse_providers(self, providers_str: str) -> List[str]:
        """Parse provider list from command line"""
        if not providers_str:
            return ["memory"]  # Default to in-memory only

        available_providers = ["memory", "redis", "postgres"]
        requested = [p.strip() for p in providers_str.split(",")]

        valid_providers = []
        for provider in requested:
            if provider in available_providers:
                valid_providers.append(provider)
            else:
                print(f"Warning: Unknown provider '{provider}', skipping")

        return valid_providers or ["memory"]

    async def run_all_tests(self) -> ValidationReport:
        """Run the complete test suite"""
        print("🧪 A2A Memory System Comprehensive Validation")
        print("=" * 60)
        print(f"Testing providers: {', '.join(self.providers)}")
        print(
            f"Options: coverage={self.args.coverage}, performance={self.args.performance}, stress={self.args.stress}"
        )
        print()

        start_time = time.perf_counter()

        # Phase 1: Basic functionality tests
        await self._run_phase_1_tests()

        # Phase 2: Lifecycle integration tests
        await self._run_phase_2_tests()

        # Phase 3: Stress and concurrency tests
        if self.args.stress:
            await self._run_phase_3_tests()

        # Phase 4: Coverage and performance analysis
        if self.args.coverage:
            await self._run_coverage_analysis()

        if self.args.performance:
            await self._run_performance_benchmarks()

        end_time = time.perf_counter()
        self.report.total_duration_ms = (end_time - start_time) * 1000

        # Generate final analysis
        self._analyze_results()

        # Generate report if requested
        if self.args.report:
            await self._generate_validation_report()

        return self.report

    async def _run_phase_1_tests(self):
        """Run Phase 1: Foundational serialization tests"""
        print("📋 Phase 1: Foundational Serialization Tests")
        print("-" * 50)

        for provider in self.providers:
            suite_result = TestSuiteResult(suite_name="Serialization Tests", provider=provider)

            try:
                # Run serialization tests
                results = await self._execute_pytest_suite(
                    "test_serialization.py",
                    provider,
                    timeout=300,  # 5 minutes
                )

                suite_result.results.extend(results)
                self._update_suite_stats(suite_result)

            except Exception as e:
                suite_result.errors += 1
                print(f"❌ Phase 1 failed for {provider}: {e}")

            self.report.suite_results.append(suite_result)
            self._print_suite_summary(suite_result)

    async def _run_phase_2_tests(self):
        """Run Phase 2: Lifecycle integration tests"""
        print("\n🔄 Phase 2: Lifecycle Integration Tests")
        print("-" * 50)

        for provider in self.providers:
            suite_result = TestSuiteResult(suite_name="Lifecycle Tests", provider=provider)

            try:
                # Run lifecycle tests
                results = await self._execute_pytest_suite(
                    "test_task_lifecycle.py",
                    provider,
                    timeout=600,  # 10 minutes
                )

                suite_result.results.extend(results)
                self._update_suite_stats(suite_result)

            except Exception as e:
                suite_result.errors += 1
                print(f"❌ Phase 2 failed for {provider}: {e}")

            self.report.suite_results.append(suite_result)
            self._print_suite_summary(suite_result)

    async def _run_phase_3_tests(self):
        """Run Phase 3: Stress and concurrency tests"""
        print("\n⚡ Phase 3: Stress & Concurrency Tests")
        print("-" * 50)

        for provider in self.providers:
            # Stress tests
            stress_suite = TestSuiteResult(suite_name="Stress Tests", provider=provider)

            try:
                stress_results = await self._execute_pytest_suite(
                    "test_stress_concurrency.py",
                    provider,
                    timeout=1800,  # 30 minutes
                )

                stress_suite.results.extend(stress_results)
                self._update_suite_stats(stress_suite)

            except Exception as e:
                stress_suite.errors += 1
                print(f"❌ Stress tests failed for {provider}: {e}")

            self.report.suite_results.append(stress_suite)
            self._print_suite_summary(stress_suite)

            # Cleanup tests
            cleanup_suite = TestSuiteResult(suite_name="Cleanup Tests", provider=provider)

            try:
                cleanup_results = await self._execute_pytest_suite(
                    "test_cleanup.py",
                    provider,
                    timeout=600,  # 10 minutes
                )

                cleanup_suite.results.extend(cleanup_results)
                self._update_suite_stats(cleanup_suite)

            except Exception as e:
                cleanup_suite.errors += 1
                print(f"❌ Cleanup tests failed for {provider}: {e}")

            self.report.suite_results.append(cleanup_suite)
            self._print_suite_summary(cleanup_suite)

    async def _execute_pytest_suite(
        self, test_file: str, provider: str, timeout: int = 300
    ) -> List[TestResult]:
        """Execute a pytest suite and parse results"""
        test_path = Path(__file__).parent / test_file

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json",
        ]

        if provider != "memory":
            # Add provider-specific markers if needed
            cmd.extend(["-m", f"not skip_{provider}"])

        print(f"  Running {test_file} for {provider}...")

        try:
            # Execute pytest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent.parent,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

                # Parse JSON report
                try:
                    with open("/tmp/pytest_report.json") as f:
                        report_data = json.load(f)

                    return self._parse_pytest_results(report_data, provider)

                except FileNotFoundError:
                    # Fallback to parsing stdout
                    return self._parse_pytest_stdout(stdout.decode(), provider)

            except asyncio.TimeoutError:
                process.kill()
                return [
                    TestResult(
                        test_name=f"{test_file}_timeout",
                        status="ERROR",
                        duration_ms=timeout * 1000,
                        error_message=f"Test suite timed out after {timeout}s",
                        provider=provider,
                    )
                ]

        except Exception as e:
            return [
                TestResult(
                    test_name=f"{test_file}_execution_error",
                    status="ERROR",
                    duration_ms=0.0,
                    error_message=str(e),
                    provider=provider,
                )
            ]

    def _parse_pytest_results(self, report_data: Dict, provider: str) -> List[TestResult]:
        """Parse pytest JSON report"""
        results = []

        for test in report_data.get("tests", []):
            result = TestResult(
                test_name=test.get("nodeid", "unknown"),
                status=test.get("outcome", "UNKNOWN").upper(),
                duration_ms=test.get("duration", 0.0) * 1000,
                provider=provider,
            )

            if test.get("call", {}).get("longrepr"):
                result.error_message = str(test["call"]["longrepr"])

            results.append(result)

        return results

    def _parse_pytest_stdout(self, stdout: str, provider: str) -> List[TestResult]:
        """Fallback parser for pytest stdout"""
        results = []
        lines = stdout.split("\n")

        for line in lines:
            if "::" in line and any(
                status in line for status in ["PASSED", "FAILED", "SKIPPED", "ERROR"]
            ):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    status = (
                        parts[1]
                        if parts[1] in ["PASSED", "FAILED", "SKIPPED", "ERROR"]
                        else "UNKNOWN"
                    )

                    results.append(
                        TestResult(
                            test_name=test_name,
                            status=status,
                            duration_ms=0.0,  # Duration not available from stdout
                            provider=provider,
                        )
                    )

        return results

    def _update_suite_stats(self, suite_result: TestSuiteResult):
        """Update suite statistics from individual test results"""
        suite_result.total_tests = len(suite_result.results)
        suite_result.duration_ms = sum(r.duration_ms for r in suite_result.results)

        for result in suite_result.results:
            if result.status == "PASSED":
                suite_result.passed += 1
            elif result.status == "FAILED":
                suite_result.failed += 1
            elif result.status == "SKIPPED":
                suite_result.skipped += 1
            else:
                suite_result.errors += 1

    def _print_suite_summary(self, suite_result: TestSuiteResult):
        """Print summary for a test suite"""
        total = suite_result.total_tests
        passed = suite_result.passed
        failed = suite_result.failed
        skipped = suite_result.skipped
        errors = suite_result.errors
        duration = suite_result.duration_ms / 1000

        status_icon = "✅" if failed == 0 and errors == 0 else "❌"

        print(f"  {status_icon} {suite_result.provider} - {suite_result.suite_name}")
        print(
            f"    Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}, Errors: {errors}"
        )
        print(f"    Duration: {duration:.2f}s")

        if failed > 0 or errors > 0:
            for result in suite_result.results:
                if result.status in ["FAILED", "ERROR"]:
                    print(f"    ❌ {result.test_name}: {result.error_message or 'No details'}")

    async def _run_coverage_analysis(self):
        """Run code coverage analysis"""
        print("\n📊 Code Coverage Analysis")
        print("-" * 50)

        try:
            # Run pytest with coverage
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(Path(__file__).parent),
                "--cov=jaf.a2a.memory",
                "--cov-report=json:/tmp/coverage.json",
                "--cov-report=term",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent.parent,
            )

            stdout, stderr = await process.communicate()

            # Parse coverage report
            try:
                with open("/tmp/coverage.json") as f:
                    coverage_data = json.load(f)

                self.report.coverage_report = {
                    "coverage_percent": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                    "lines_missing": coverage_data.get("totals", {}).get("missing_lines", 0),
                    "files": list(coverage_data.get("files", {}).keys()),
                }

                coverage_percent = self.report.coverage_report["coverage_percent"]

                if coverage_percent >= 95:
                    print(f"✅ Excellent coverage: {coverage_percent:.1f}%")
                elif coverage_percent >= 85:
                    print(f"⚠️ Good coverage: {coverage_percent:.1f}%")
                else:
                    print(f"❌ Low coverage: {coverage_percent:.1f}%")
                    self.report.critical_issues.append(
                        f"Code coverage below 85%: {coverage_percent:.1f}%"
                    )

            except FileNotFoundError:
                print("❌ Coverage report not generated")

        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")

    async def _run_performance_benchmarks(self):
        """Run performance benchmarks"""
        print("\n⚡ Performance Benchmarks")
        print("-" * 50)

        try:
            # Create a simple benchmark provider
            config = A2AInMemoryTaskConfig(max_tasks=10000)
            provider = create_a2a_in_memory_task_provider(config)

            # Basic operation benchmarks
            from jaf.a2a.types import A2ATask, A2ATaskStatus, TaskState

            test_task = A2ATask(
                id="benchmark_task",
                contextId="benchmark_ctx",
                kind="task",
                status=A2ATaskStatus(state=TaskState.SUBMITTED),
            )

            # Benchmark store operation
            start_time = time.perf_counter()
            for i in range(100):
                task = test_task.model_copy(update={"id": f"benchmark_task_{i}"})
                await provider.store_task(task)
            store_time = (time.perf_counter() - start_time) * 1000

            # Benchmark get operation
            start_time = time.perf_counter()
            for i in range(100):
                await provider.get_task(f"benchmark_task_{i}")
            get_time = (time.perf_counter() - start_time) * 1000

            await provider.close()

            self.report.performance_metrics = {
                "store_ops_per_sec": 100 / (store_time / 1000),
                "get_ops_per_sec": 100 / (get_time / 1000),
                "avg_store_time_ms": store_time / 100,
                "avg_get_time_ms": get_time / 100,
            }

            metrics = self.report.performance_metrics
            print(
                f"✅ Store: {metrics['store_ops_per_sec']:.2f} ops/sec ({metrics['avg_store_time_ms']:.2f}ms avg)"
            )
            print(
                f"✅ Get: {metrics['get_ops_per_sec']:.2f} ops/sec ({metrics['avg_get_time_ms']:.2f}ms avg)"
            )

            # Performance thresholds
            if metrics["store_ops_per_sec"] < 100:
                self.report.critical_issues.append(
                    f"Store performance too slow: {metrics['store_ops_per_sec']:.2f} ops/sec"
                )

            if metrics["get_ops_per_sec"] < 200:
                self.report.critical_issues.append(
                    f"Get performance too slow: {metrics['get_ops_per_sec']:.2f} ops/sec"
                )

        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")

    def _analyze_results(self):
        """Analyze all test results and generate recommendations"""
        total_tests = sum(suite.total_tests for suite in self.report.suite_results)
        total_passed = sum(suite.passed for suite in self.report.suite_results)
        total_failed = sum(suite.failed for suite in self.report.suite_results)
        total_errors = sum(suite.errors for suite in self.report.suite_results)

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Critical issues analysis
        if total_failed > 0:
            self.report.critical_issues.append(f"{total_failed} test failures detected")

        if total_errors > 0:
            self.report.critical_issues.append(f"{total_errors} test errors detected")

        if success_rate < 95:
            self.report.critical_issues.append(f"Test success rate below 95%: {success_rate:.1f}%")

        # Provider compliance analysis
        provider_compliance = {}
        for provider in self.providers:
            provider_suites = [s for s in self.report.suite_results if s.provider == provider]
            provider_passed = sum(s.passed for s in provider_suites)
            provider_total = sum(s.total_tests for s in provider_suites)
            provider_compliance[provider] = (
                (provider_passed / provider_total * 100) if provider_total > 0 else 0
            )

        # Recommendations
        if self.report.coverage_report and self.report.coverage_report["coverage_percent"] < 90:
            self.report.recommendations.append("Increase test coverage to at least 90%")

        if any(compliance < 100 for compliance in provider_compliance.values()):
            self.report.recommendations.append("Fix failing tests for full provider compliance")

        if self.report.performance_metrics:
            metrics = self.report.performance_metrics
            if metrics.get("store_ops_per_sec", 0) < 500:
                self.report.recommendations.append(
                    "Optimize storage performance for production workloads"
                )

        # Final verdict
        if len(self.report.critical_issues) == 0 and success_rate >= 98:
            self.report.final_verdict = (
                "✅ PRODUCTION READY - All tests pass with excellent coverage"
            )
        elif len(self.report.critical_issues) <= 2 and success_rate >= 95:
            self.report.final_verdict = (
                "⚠️ READY WITH MINOR ISSUES - Address recommendations before production"
            )
        else:
            self.report.final_verdict = "❌ NOT READY - Critical issues must be resolved"

    async def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📋 Generating Validation Report")
        print("-" * 50)

        report_path = Path("a2a_memory_validation_report.json")

        # Convert dataclasses to dicts for JSON serialization
        report_dict = {
            "timestamp": self.report.timestamp,
            "total_duration_ms": self.report.total_duration_ms,
            "providers_tested": self.report.providers_tested,
            "suite_results": [
                {
                    "suite_name": suite.suite_name,
                    "provider": suite.provider,
                    "total_tests": suite.total_tests,
                    "passed": suite.passed,
                    "failed": suite.failed,
                    "skipped": suite.skipped,
                    "errors": suite.errors,
                    "duration_ms": suite.duration_ms,
                    "results": [
                        {
                            "test_name": result.test_name,
                            "status": result.status,
                            "duration_ms": result.duration_ms,
                            "error_message": result.error_message,
                            "provider": result.provider,
                        }
                        for result in suite.results
                    ],
                }
                for suite in self.report.suite_results
            ],
            "coverage_report": self.report.coverage_report,
            "performance_metrics": self.report.performance_metrics,
            "critical_issues": self.report.critical_issues,
            "recommendations": self.report.recommendations,
            "final_verdict": self.report.final_verdict,
        }

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"✅ Validation report saved to: {report_path}")

        # Generate markdown summary
        md_path = Path("a2a_memory_validation_summary.md")
        await self._generate_markdown_summary(md_path)
        print(f"✅ Summary report saved to: {md_path}")

    async def _generate_markdown_summary(self, md_path: Path):
        """Generate markdown summary report"""
        with open(md_path, "w") as f:
            f.write("# A2A Memory System Validation Report\n\n")
            f.write(f"**Generated:** {self.report.timestamp}\n")
            f.write(f"**Duration:** {self.report.total_duration_ms / 1000:.2f} seconds\n")
            f.write(f"**Providers Tested:** {', '.join(self.report.providers_tested)}\n\n")

            # Summary
            total_tests = sum(suite.total_tests for suite in self.report.suite_results)
            total_passed = sum(suite.passed for suite in self.report.suite_results)
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

            f.write("## Summary\n\n")
            f.write(f"- **Total Tests:** {total_tests}\n")
            f.write(f"- **Passed:** {total_passed}\n")
            f.write(f"- **Success Rate:** {success_rate:.1f}%\n")

            if self.report.coverage_report:
                f.write(
                    f"- **Code Coverage:** {self.report.coverage_report['coverage_percent']:.1f}%\n"
                )

            f.write(f"\n**Final Verdict:** {self.report.final_verdict}\n\n")

            # Critical Issues
            if self.report.critical_issues:
                f.write("## Critical Issues\n\n")
                for issue in self.report.critical_issues:
                    f.write(f"- ❌ {issue}\n")
                f.write("\n")

            # Recommendations
            if self.report.recommendations:
                f.write("## Recommendations\n\n")
                for rec in self.report.recommendations:
                    f.write(f"- 💡 {rec}\n")
                f.write("\n")

            # Test Suite Results
            f.write("## Test Suite Results\n\n")
            for suite in self.report.suite_results:
                status_icon = "✅" if suite.failed == 0 and suite.errors == 0 else "❌"
                f.write(f"### {status_icon} {suite.suite_name} ({suite.provider})\n\n")
                f.write(f"- **Tests:** {suite.total_tests}\n")
                f.write(f"- **Passed:** {suite.passed}\n")
                f.write(f"- **Failed:** {suite.failed}\n")
                f.write(f"- **Errors:** {suite.errors}\n")
                f.write(f"- **Duration:** {suite.duration_ms / 1000:.2f}s\n\n")

    def print_final_summary(self):
        """Print final validation summary"""
        print("\n" + "=" * 60)
        print("🏁 A2A MEMORY SYSTEM VALIDATION COMPLETE")
        print("=" * 60)

        total_tests = sum(suite.total_tests for suite in self.report.suite_results)
        total_passed = sum(suite.passed for suite in self.report.suite_results)
        total_failed = sum(suite.failed for suite in self.report.suite_results)
        total_errors = sum(suite.errors for suite in self.report.suite_results)

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print("📊 RESULTS SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Errors: {total_errors}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Duration: {self.report.total_duration_ms / 1000:.2f} seconds")

        if self.report.coverage_report:
            coverage = self.report.coverage_report["coverage_percent"]
            print(f"   Code Coverage: {coverage:.1f}%")

        if self.report.performance_metrics:
            metrics = self.report.performance_metrics
            print(
                f"   Performance: Store {metrics['store_ops_per_sec']:.0f} ops/s, Get {metrics['get_ops_per_sec']:.0f} ops/s"
            )

        print(f"\n🎯 FINAL VERDICT: {self.report.final_verdict}")

        if self.report.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.report.critical_issues)}):")
            for issue in self.report.critical_issues:
                print(f"   ❌ {issue}")

        if self.report.recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(self.report.recommendations)}):")
            for rec in self.report.recommendations:
                print(f"   • {rec}")

        print("\n" + "=" * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="A2A Memory Comprehensive Test Runner")
    parser.add_argument(
        "--providers", default="memory", help="Providers to test (memory,redis,postgres)"
    )
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--performance", action="store_true", help="Include performance benchmarks")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--report", action="store_true", help="Generate validation report")

    args = parser.parse_args()

    runner = ComprehensiveTestRunner(args)

    try:
        report = await runner.run_all_tests()
        runner.print_final_summary()

        # Return appropriate exit code
        total_failed = sum(suite.failed for suite in report.suite_results)
        total_errors = sum(suite.errors for suite in report.suite_results)

        if total_failed > 0 or total_errors > 0:
            sys.exit(1)  # Test failures
        else:
            sys.exit(0)  # Success

    except KeyboardInterrupt:
        print("\n❌ Test run interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ Test run failed with exception: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
