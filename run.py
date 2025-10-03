#!/usr/bin/env python3
"""
RAMwich Simulator - Main Entry Point

A comprehensive simulator for SRAM-based Compute-in-Memory (CIM) architectures
with advanced visualization and analysis capabilities.

Usage:
    python run.py --config config.yaml --ops ops.json [OPTIONS]

Examples:
    # Basic simulation
    python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json
    
    # With weights and activation
    python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json \
                  --weights examples/mlp_l4_mnist/weights.npz --activation examples/mlp_l4_mnist/activation.npy
    
    # With visualization output
    python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json \
                  --weights examples/mlp_l4_mnist/weights.npz --activation examples/mlp_l4_mnist/activation.npy \
                  --visualize --output-dir results/
    
    # Quiet mode with JSON output
    python run.py --config config.yaml --ops ops.json --quiet --json-output results.json

Author: RAMwich Development Team
License: MIT
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from ramwich import RAMwich


def setup_logging(level: str = "WARNING", quiet: bool = False) -> None:
    """Setup logging configuration"""
    if quiet:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
    else:
        log_level = getattr(logging, level.upper(), logging.WARNING)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def validate_files(config_file: str, ops_file: str, weights_file: Optional[str] = None, 
                  activation_file: Optional[str] = None) -> bool:
    """Validate that required files exist"""
    files_to_check = [
        ("Configuration", config_file),
        ("Operations", ops_file)
    ]
    
    if weights_file:
        files_to_check.append(("Weights", weights_file))
    if activation_file:
        files_to_check.append(("Activation", activation_file))
    
    missing_files = []
    for file_type, file_path in files_to_check:
        if not Path(file_path).exists():
            missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        print("Missing required files:")
        for missing in missing_files:
            print(f"   {missing}")
        return False
    
    return True


def format_stats_summary(stats: Dict[str, Any], simulation_time: float) -> str:
    """Format statistics summary for display"""
    summary = []
    summary.append("=" * 60)
    summary.append("SIMULATION RESULTS SUMMARY")
    summary.append("=" * 60)
    
    # Calculate totals from component stats
    total_dynamic = 0
    total_leakage = 0
    total_area = 0
    
    for key, value in stats.items():
        if key not in ['simulation_time', 'total_time'] and hasattr(value, 'dynamic_energy'):
            # Access Stats object attributes
            dynamic = getattr(value, 'dynamic_energy', 0)
            leakage = getattr(value, 'leakage_energy', 0)
            area = getattr(value, 'area', 0)
            total_dynamic += dynamic
            total_leakage += leakage
            total_area += area
    
    # Convert energy from pJ to J (component values appear to be in pJ)
    total_dynamic_j = total_dynamic * 1e-12
    total_leakage_j = total_leakage * 1e-12
    total_energy = total_dynamic_j + total_leakage_j
    
    summary.append(f"Energy Analysis:")
    summary.append(f"  Total Energy:    {total_energy:.3e} J")
    if total_energy > 0:
        summary.append(f"  Dynamic Energy:  {total_dynamic_j:.3e} J ({total_dynamic_j/total_energy*100:.1f}%)")
        summary.append(f"  Leakage Energy:  {total_leakage_j:.3e} J ({total_leakage_j/total_energy*100:.1f}%)")
    
    # Area summary
    summary.append(f"Area Analysis:")
    summary.append(f"  Total Area:      {total_area:.3f} mm²")
    
    # Performance summary
    summary.append(f"Performance:")
    summary.append(f"  Simulation Time: {simulation_time:.3f} seconds")
    
    summary.append("=" * 60)
    return "\n".join(summary)


def save_results(stats: Dict[str, Any], output_file: str, format_type: str = "json", quiet: bool = False) -> None:
    """Save simulation results to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type.lower() == "json":
        # Convert numpy types and Pydantic models to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle Pydantic BaseModel objects
            elif hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            return obj
        
        # Recursively convert numpy types and Pydantic models
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_stats = deep_convert(stats)
        
        with open(output_path, 'w') as f:
            json.dump(converted_stats, f, indent=2)
        if not quiet:
            print(f"Results saved to: {output_path}")
    else:
        if not quiet:
            print(f"Unsupported output format: {format_type}")


def create_visualization(simulator: RAMwich, output_dir: str, quiet: bool = False) -> None:
    """Create visualization dashboard from simulation results"""
    try:
        from ramwich.visualization import VisualizationDashboard
        
        dashboard = VisualizationDashboard(output_dir)
        stats = simulator.get_stats()
        cycles = getattr(simulator, 'total_cycles', 0)
        
        dashboard.add_simulation_result("Simulation", stats, cycles)
        html_path = dashboard.generate_comprehensive_dashboard("RAMwich Simulation Results")
        
        if not quiet:
            print(f"Visualization dashboard created: {html_path}")
            print(f"Open in browser: file://{Path(html_path).absolute()}")
        
    except ImportError:
        if not quiet:
            print("Warning: Visualization modules not available. Install with: pip install matplotlib seaborn")
    except Exception as e:
        if not quiet:
            print(f"Error creating visualization: {e}")


def main():
    """Main entry point for RAMwich simulator"""
    parser = argparse.ArgumentParser(
        description="RAMwich: SRAM-based Compute-in-Memory Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python run.py --config config.yaml --ops ops.json
  
  # Full simulation with weights and activation
  python run.py --config config.yaml --ops ops.json --weights weights.npz --activation input.npy
  
  # With visualization output
  python run.py --config config.yaml --ops ops.json --visualize --output-dir results/
  
  # Quiet mode with JSON output
  python run.py --config config.yaml --ops ops.json --quiet --json-output results.json

For more information, see: docs/usage.md
        """
    )
    
    # Required arguments
    parser.add_argument("--config", required=True, 
                       help="Configuration file (YAML format)")
    parser.add_argument("--ops", required=True, 
                       help="Operations file (JSON format)")
    
    # Optional input files
    parser.add_argument("--weights", 
                       help="Pre-trained weights file (NPZ format)")
    parser.add_argument("--activation", 
                       help="Input activation data (NPY format)")
    
    # Output options
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for results (default: output)")
    parser.add_argument("--json-output",
                       help="Save results to JSON file")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization dashboard")
    
    # Execution options
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--log-level", default="WARNING",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level (default: WARNING)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate input files, don't run simulation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.quiet)
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        print("RAMwich SRAM-CIM Simulator")
        print("=" * 40)
    
    # Validate input files
    if not validate_files(args.config, args.ops, args.weights, args.activation):
        sys.exit(1)
    
    if args.validate_only:
        if not args.quiet:
            print("All input files validated successfully")
        return
    
    try:
        # Initialize simulator
        if not args.quiet:
            print("Initializing simulator...")
        
        start_time = time.time()
        simulator = RAMwich(
            config_file=args.config,
            ops_file=args.ops,
            weights_file=args.weights,
            quiet=args.quiet
        )
        
        # Run simulation
        if not args.quiet:
            print("Running simulation...")
        
        simulation_start = time.time()
        simulator.run(activation=args.activation)
        simulation_time = time.time() - simulation_start
        
        # Collect results
        stats = simulator.get_stats()
        stats['simulation_time'] = simulation_time
        stats['total_time'] = time.time() - start_time
        
        if hasattr(simulator, 'total_cycles'):
            stats['total_cycles'] = simulator.total_cycles
        
        # Display results
        if not args.quiet:
            print(format_stats_summary(stats, simulation_time))
        
        # Save JSON output if requested
        if args.json_output:
            save_results(stats, args.json_output, "json", args.quiet)
        
        # Create visualization if requested
        if args.visualize:
            if not args.quiet:
                print("Generating visualization...")
            create_visualization(simulator, args.output_dir, args.quiet)
        
        if not args.quiet:
            print("Simulation completed successfully!")
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()