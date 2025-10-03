"""
Command-line interface for RAMwich simulator.

This module provides the main CLI entry point for the RAMwich simulator,
offering a comprehensive interface for running simulations, generating
visualizations, and exporting results.
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .ramwich import RAMwich
from .performance import PerformanceMonitor


def setup_logging(level: str = "WARNING") -> None:
    """
    Configure logging for the CLI application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="ramwich",
        description="RAMwich: SRAM-based Compute-in-Memory Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  ramwich --config config.yaml --ops ops.json
  
  # Full simulation with weights and activation
  ramwich --config config.yaml --ops ops.json --weights weights.npz --activation input.npy
  
  # With visualization output
  ramwich --config config.yaml --ops ops.json --visualize --output-dir results/
  
  # Quiet mode with JSON output
  ramwich --config config.yaml --ops ops.json --quiet --json-output results.json
  
  # Performance benchmarking
  ramwich --config config.yaml --ops ops.json --benchmark --performance-report perf.json

For more information, see: https://github.com/your-org/ramwich/docs
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--ops",
        required=True,
        help="Operations file (JSON format)"
    )
    
    # Optional input files
    parser.add_argument(
        "--weights",
        help="Pre-trained weights file (NPZ format)"
    )
    
    parser.add_argument(
        "--activation",
        help="Input activation data (NPY format)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--json-output",
        help="Save results to JSON file"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization dashboard"
    )
    
    # Performance options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarking"
    )
    
    parser.add_argument(
        "--performance-report",
        help="Save performance report to JSON file"
    )
    
    # Control options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set logging level (default: WARNING)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input files, don't run simulation"
    )
    
    # Advanced options
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1)"
    )
    
    parser.add_argument(
        "--memory-limit",
        help="Memory limit (e.g., '4GB', '512MB')"
    )
    
    return parser


def validate_files(args: argparse.Namespace) -> bool:
    """
    Validate that required files exist and are accessible.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if all files are valid, False otherwise
    """
    missing_files = []
    
    # Check required files
    if not Path(args.config).exists():
        missing_files.append(args.config)
    
    if not Path(args.ops).exists():
        missing_files.append(args.ops)
    
    # Check optional files
    if args.weights and not Path(args.weights).exists():
        missing_files.append(args.weights)
    
    if args.activation and not Path(args.activation).exists():
        missing_files.append(args.activation)
    
    if missing_files:
        print("Missing required files:")
        for missing in missing_files:
            print(f"   {missing}")
        return False
    
    return True


def run_simulation(args: argparse.Namespace) -> Optional[dict]:
    """
    Run the main simulation.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Simulation results dictionary or None if failed
    """
    try:
        # Initialize performance monitoring if requested
        monitor = None
        if args.benchmark:
            monitor = PerformanceMonitor(enable_detailed_monitoring=True)
            monitor.start_monitoring()
        
        # Initialize simulator
        if not args.quiet:
            print("RAMwich SRAM-CIM Simulator")
            print("=" * 40)
            print("Initializing simulator...")
        
        simulator = RAMwich(
            config_file=args.config,
            ops_file=args.ops,
            weights_file=args.weights,
            quiet=args.quiet
        )
        
        # Run simulation
        if not args.quiet:
            print("Running simulation...")
        
        results = simulator.run(activation=args.activation)
        
        # Get statistics
        stats = simulator.get_stats()
        
        # Stop performance monitoring
        if monitor:
            perf_metrics = monitor.stop_monitoring()
            
            # Calculate performance metrics
            if hasattr(simulator, 'total_operations'):
                monitor.calculate_throughput(simulator.total_operations)
            
            if hasattr(stats, 'total_energy'):
                monitor.calculate_energy_efficiency(
                    stats.total_energy, 
                    getattr(simulator, 'total_operations', 1)
                )
        
        # Save performance report if requested
        if args.performance_report and monitor:
            import json
            with open(args.performance_report, 'w') as f:
                json.dump(monitor.get_summary(), f, indent=2)
            
            if not args.quiet:
                print(f"Performance report saved to: {args.performance_report}")
        
        return {
            "stats": stats,
            "results": results,
            "performance": monitor.get_summary() if monitor else None
        }
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        if not args.quiet:
            print(f"Error: {e}")
        return None


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input files
    if not validate_files(args):
        return 1
    
    # Validation-only mode
    if args.validate_only:
        if not args.quiet:
            print("All input files validated successfully")
        return 0
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    results = run_simulation(args)
    if results is None:
        return 1
    
    # Save JSON output if requested
    if args.json_output:
        import json
        try:
            with open(args.json_output, 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    import numpy as np
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                def deep_convert(data):
                    if isinstance(data, dict):
                        return {k: deep_convert(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [deep_convert(item) for item in data]
                    else:
                        return convert_numpy(data)
                
                converted_results = deep_convert(results)
                json.dump(converted_results, f, indent=2)
            
            if not args.quiet:
                print(f"Results saved to: {args.json_output}")
                
        except Exception as e:
            logging.error(f"Failed to save JSON output: {e}")
            if not args.quiet:
                print(f"Warning: Could not save JSON output: {e}")
    
    # Generate visualization if requested
    if args.visualize:
        try:
            if not args.quiet:
                print("Generating visualization...")
            
            # Import visualization modules
            from .visualization import VisualizationDashboard
            
            dashboard = VisualizationDashboard(results["stats"])
            html_path = dashboard.generate_comprehensive_dashboard(
                "RAMwich Simulation Results",
                output_dir=args.output_dir
            )
            
            if not args.quiet:
                print(f"Visualization dashboard created: {html_path}")
                print(f"Open in browser: file://{Path(html_path).absolute()}")
                
        except ImportError:
            if not args.quiet:
                print("Warning: Visualization modules not available. Install with: pip install ramwich[viz]")
        except Exception as e:
            logging.error(f"Visualization generation failed: {e}")
            if not args.quiet:
                print(f"Warning: Could not generate visualization: {e}")
    
    if not args.quiet:
        print("Simulation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())