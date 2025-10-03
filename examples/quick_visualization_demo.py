#!/usr/bin/env python3

"""
Quick RAMwich Visualization Demo

A simple demonstration of the RAMwich visualization capabilities.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.mvmu import MVMU
from ramwich.visualization import VisualizationDashboard


def main():
    print("RAMwich Visualization Quick Demo")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("quick_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dashboard
    dashboard = VisualizationDashboard(str(output_dir))
    
    print("1. Creating SRAM CIM configuration...")
    
    # SRAM CIM Configuration
    sram_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],
        weight_format="Q2.0",
        activation_format="Q8.0"
    )
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=sram_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    print("2. Running simulation...")
    
    # Simple simulation
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    weights = np.eye(xbar_size, dtype=np.float64)  # Identity matrix
    input_data = np.arange(1, xbar_size + 1, dtype=np.int32)
    
    mvmu.load_weights(weights)
    mvmu.write_to_inreg(0, input_data)
    mvmu.execute_mvm()
    
    stats = mvmu.get_stats()
    
    print("3. Adding results to dashboard...")
    
    # Add to dashboard
    dashboard.add_simulation_result("SRAM_CIM_Demo", stats, 1000)
    
    print("4. Generating visualization dashboard...")
    
    # Generate dashboard
    html_path = dashboard.generate_comprehensive_dashboard("Quick Demo Dashboard")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Dashboard generated: {html_path}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Show what was generated
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    
    if plots_dir.exists():
        plot_count = len(list(plots_dir.glob("*.png")))
        print(f"📈 Generated {plot_count} visualization plots")
    
    if reports_dir.exists():
        report_count = len(list(reports_dir.glob("*.txt")))
        print(f"📋 Generated {report_count} analysis reports")
    
    print(f"\n🌐 Open {html_path} in your browser to view the dashboard!")


if __name__ == "__main__":
    main()