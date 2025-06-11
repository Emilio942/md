#!/usr/bin/env python3
"""
Cross-Platform Integration Test Runner

Task 10.2: Integration Tests 📊 - Cross-Platform Execution Script

This script provides unified test execution across Linux, Windows, and macOS
platforms with platform-specific optimizations and reporting.
"""

import os
import sys
import platform
import subprocess
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrossPlatformTestRunner:
    """Cross-platform integration test runner for ProteinMD."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.platform_version = platform.release()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.start_time = time.time()
        
        # Platform-specific configurations
        self.platform_configs = {
            'linux': {
                'pytest_args': ['-v', '--tb=short', '--durations=10'],
                'parallel': True,
                'max_workers': os.cpu_count() or 4,
                'temp_dir': '/tmp/proteinmd_tests'
            },
            'windows': {
                'pytest_args': ['-v', '--tb=short', '--durations=10'],
                'parallel': False,  # Windows threading issues
                'max_workers': 1,
                'temp_dir': os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'proteinmd_tests')
            },
            'darwin': {  # macOS
                'pytest_args': ['-v', '--tb=short', '--durations=10'],
                'parallel': True,
                'max_workers': os.cpu_count() or 4,
                'temp_dir': '/tmp/proteinmd_tests'
            }
        }
        
        self.config = self.platform_configs.get(self.platform, self.platform_configs['linux'])
        
    def setup_environment(self) -> bool:
        """Setup test environment with platform-specific configurations."""
        try:
            logger.info(f"Setting up test environment for {self.platform}")
            
            # Create temporary directory
            temp_dir = Path(self.config['temp_dir'])
            temp_dir.mkdir(exist_ok=True, parents=True)
            os.environ['PROTEINMD_TEST_TEMP'] = str(temp_dir)
            
            # Platform-specific setup
            if self.platform == 'windows':
                # Windows-specific PATH adjustments
                current_path = os.environ.get('PATH', '')
                python_path = Path(sys.executable).parent
                if str(python_path) not in current_path:
                    os.environ['PATH'] = f"{python_path};{current_path}"
                    
                # Set Windows-specific environment variables
                os.environ['PROTEINMD_WINDOWS_MODE'] = '1'
                
            elif self.platform == 'darwin':
                # macOS-specific setup
                os.environ['PROTEINMD_MACOS_MODE'] = '1'
                # Disable macOS-specific warnings that can interfere with tests
                os.environ['PYTHONHTTPSVERIFY'] = '0'
                
            elif self.platform == 'linux':
                # Linux-specific setup
                os.environ['PROTEINMD_LINUX_MODE'] = '1'
                
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            return False
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required dependencies are available."""
        required_packages = [
            'pytest', 'numpy', 'scipy', 'matplotlib', 'mock'
        ]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.debug(f"✓ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"✗ {package} missing")
        
        return len(missing_packages) == 0, missing_packages
    
    def run_integration_tests(self, test_pattern: Optional[str] = None) -> Dict:
        """Run integration tests with platform-specific configurations."""
        logger.info(f"Running integration tests on {self.platform} {self.platform_version}")
        
        # Base pytest command
        cmd = [sys.executable, '-m', 'pytest']
        cmd.extend(self.config['pytest_args'])
        
        # Add test pattern if specified
        if test_pattern:
            cmd.append(f"tests/{test_pattern}")
        else:
            cmd.append("tests/test_integration_workflows.py")
        
        # Add parallel execution if supported
        if self.config['parallel']:
            try:
                import pytest_xdist
                cmd.extend(['-n', str(self.config['max_workers'])])
            except ImportError:
                logger.warning("pytest-xdist not available, running tests sequentially")
        
        # Platform-specific pytest options
        if self.platform == 'windows':
            cmd.extend(['--tb=line'])  # Simpler output for Windows
        
        # Add output formats
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_output = f"test_results_{self.platform}_{timestamp}.json"
        html_output = f"test_report_{self.platform}_{timestamp}.html"
        
        # Try to add JSON report if available
        try:
            import pytest_json_report
            cmd.extend([f"--json-report", f"--json-report-file={json_output}"])
        except ImportError:
            logger.warning("pytest-json-report not available, skipping JSON output")
        
        # Try to add HTML report if available  
        try:
            import pytest_html
            cmd.extend([f"--html={html_output}", "--self-contained-html"])
        except ImportError:
            logger.warning("pytest-html not available, skipping HTML output")
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=Path(__file__).parent.parent
            )
            
            # Parse results
            test_results = {
                'platform': self.platform,
                'platform_version': self.platform_version,
                'python_version': self.python_version,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': time.time() - self.start_time,
                'json_output': json_output,
                'html_output': html_output
            }
            
            # Load JSON results if available
            if Path(json_output).exists():
                with open(json_output) as f:
                    pytest_json = json.load(f)
                    test_results['pytest_summary'] = pytest_json.get('summary', {})
                    test_results['test_count'] = len(pytest_json.get('tests', []))
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out after 1 hour")
            return {
                'platform': self.platform,
                'error': 'timeout',
                'execution_time': 3600
            }
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return {
                'platform': self.platform,
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }
    
    def run_benchmark_tests(self) -> Dict:
        """Run performance benchmark tests."""
        logger.info("Running benchmark tests")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/test_integration_workflows.py::TestPerformanceBenchmarks',
            '-v', '--benchmark-only', '--benchmark-json=benchmark_results.json'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes
                cwd=Path(__file__).parent.parent
            )
            
            benchmark_results = {
                'platform': self.platform,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Load benchmark JSON if available
            benchmark_file = Path('benchmark_results.json')
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    benchmark_results['benchmarks'] = benchmark_data
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark tests failed: {e}")
            return {'platform': self.platform, 'error': str(e)}
    
    def cleanup(self):
        """Cleanup temporary files and directories."""
        try:
            temp_dir = Path(self.config['temp_dir'])
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def main():
    """Main entry point for cross-platform test runner."""
    parser = argparse.ArgumentParser(description='ProteinMD Cross-Platform Integration Test Runner')
    parser.add_argument('--pattern', '-p', help='Test pattern to run (e.g., "*workflow*")')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmark tests')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--output', '-o', default='test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    runner = CrossPlatformTestRunner()
    
    logger.info(f"ProteinMD Integration Test Runner")
    logger.info(f"Platform: {runner.platform} {runner.platform_version}")
    logger.info(f"Python: {runner.python_version}")
    
    # Setup environment
    if not runner.setup_environment():
        logger.error("Failed to setup test environment")
        sys.exit(1)
    
    # Check dependencies
    if not args.skip_deps:
        deps_ok, missing = runner.check_dependencies()
        if not deps_ok:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            logger.error("Install missing packages or use --skip-deps to continue")
            sys.exit(1)
    
    try:
        # Run integration tests
        results = runner.run_integration_tests(args.pattern)
        
        # Run benchmarks if requested
        if args.benchmark:
            benchmark_results = runner.run_benchmark_tests()
            results['benchmarks'] = benchmark_results
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        if 'pytest_summary' in results:
            summary = results['pytest_summary']
            logger.info(f"Test Summary:")
            logger.info(f"  Total: {summary.get('total', 0)}")
            logger.info(f"  Passed: {summary.get('passed', 0)}")
            logger.info(f"  Failed: {summary.get('failed', 0)}")
            logger.info(f"  Skipped: {summary.get('skipped', 0)}")
            logger.info(f"  Execution time: {results['execution_time']:.2f}s")
        
        # Exit with proper code
        sys.exit(results.get('return_code', 0))
        
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
