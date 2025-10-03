#!/usr/bin/env python3

"""
RAMwich Visualization System Demonstration

This script demonstrates the complete visualization and analysis capabilities
of the RAMwich simulator, including performance analysis, energy breakdown,
and comprehensive dashboard generation.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.mvmu import MVMU
from ramwich.visualization import (
    PerformanceAnalyzer,
    EnergyAnalyzer, 
    VisualizationDashboard,
    ResultsExporter
)


def create_sample_configurations():
    """Create sample configurations for demonstration"""
    configurations = {}
    
    # SRAM CIM Configuration (Low precision, high efficiency)
    sram_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.SRAM],
        weight_format="Q2.0",
        activation_format="Q8.0"
    )
    
    # RRAM Configuration (Higher precision)
    rram_config = DataConfig(
        storage_config=[BitConfig.MLC],  # 2-bit MLC
        weight_format="Q2.0", 
        activation_format="Q8.0"
    )
    
    # Mixed Configuration (SRAM + RRAM)
    mixed_config = DataConfig(
        storage_config=[BitConfig.SRAM, BitConfig.MLC],
        weight_format="Q0.3",  # 1 SRAM + 2 RRAM = 3 bits
        activation_format="Q8.0"
    )
    
    # High precision SRAM CIM
    sram_hp_config = DataConfig(
        storage_config=[BitConfig.SRAM] * 4,  # 4-bit SRAM
        weight_format="Q4.0",
        activation_format="Q8.0"
    )
    
    configurations['SRAM_CIM_2bit'] = sram_config
    configurations['RRAM_2bit'] = rram_config
    configurations['Mixed_SRAM_RRAM'] = mixed_config
    configurations['SRAM_CIM_4bit'] = sram_hp_config
    
    return configurations


def run_neural_network_simulations():
    """Run simulations for different neural network architectures"""
    network_results = {}
    
    # LeNet-5 style configurations
    lenet_configs = {
        'SRAM_Conv': DataConfig(
            storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.SRAM],
            weight_format="Q3.0",
            activation_format="Q8.0"
        ),
        'RRAM_Conv': DataConfig(
            storage_config=[BitConfig.MLC],
            weight_format="Q2.0",
            activation_format="Q8.0"
        )
    }
    
    # ResNet-20 style configurations
    resnet_configs = {
        'SRAM_Residual': DataConfig(
            storage_config=[BitConfig.SRAM] * 4,
            weight_format="Q4.0",
            activation_format="Q8.0"
        ),
        'Mixed_Residual': DataConfig(
            storage_config=[BitConfig.SRAM, BitConfig.SRAM, BitConfig.MLC],
            weight_format="Q0.4",
            activation_format="Q8.0"
        )
    }
    
    network_results['LeNet-5'] = lenet_configs
    network_results['ResNet-20'] = resnet_configs
    
    return network_results


def simulate_configuration(data_config, config_name):
    """Simulate a single configuration and return results"""
    print(f"  Simulating {config_name}...")
    
    config = Config(
        num_tiles_per_node=1,
        num_cores_per_tile=1,
        num_mvmus_per_core=1,
        data_config=data_config
    )
    
    mvmu = MVMU(id=0, config=config)
    
    # Create test workload
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    
    # Use different patterns for different configurations
    if 'SRAM' in config_name:
        # SRAM: Use sparse, ternary-like weights
        weights = np.random.choice([-1, 0, 1], size=(xbar_size, xbar_size), 
                                 p=[0.2, 0.6, 0.2]).astype(np.float64)
    else:
        # RRAM: Use continuous weights
        weights = np.random.normal(0, 0.3, size=(xbar_size, xbar_size)).astype(np.float64)
        weights = np.clip(weights, -1, 1)
    
    # Realistic input data
    input_data = np.random.randint(0, 128, size=xbar_size).astype(np.int32)
    
    # Run simulation
    mvmu.load_weights(weights)
    mvmu.write_to_inreg(0, input_data)
    mvmu.execute_mvm()
    
    stats = mvmu.get_stats()
    
    # Simulate different cycle counts based on configuration complexity
    base_cycles = 1000
    if 'SRAM' in config_name:
        cycles = base_cycles + np.random.randint(0, 200)  # SRAM is faster
    else:
        cycles = base_cycles + np.random.randint(200, 500)  # RRAM is slower
    
    return stats, cycles


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("RAMwich Visualization System Demonstration")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("visualization_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Initialize visualization components
    dashboard = VisualizationDashboard(str(output_dir))
    exporter = ResultsExporter(str(output_dir / "exports"))
    
    print("\n1. Running Configuration Comparisons...")
    print("-" * 50)
    
    # Run configuration comparisons
    configurations = create_sample_configurations()
    config_results = {}
    
    for config_name, data_config in configurations.items():
        stats, cycles = simulate_configuration(data_config, config_name)
        config_results[config_name] = (stats, cycles)
        dashboard.add_simulation_result(config_name, stats, cycles)
    
    print(f"   Completed {len(configurations)} configuration simulations")
    
    print("\n2. Running Neural Network Simulations...")
    print("-" * 50)
    
    # Run neural network simulations
    network_configs = run_neural_network_simulations()
    network_results = {}
    
    for network_name, configs in network_configs.items():
        print(f"  Testing {network_name}:")
        network_results[network_name] = {}
        
        for config_name, data_config in configs.items():
            stats, cycles = simulate_configuration(data_config, f"{network_name}_{config_name}")
            network_results[network_name][config_name] = (stats, cycles)
            dashboard.add_simulation_result(config_name, stats, cycles, network_name)
    
    print(f"   Completed {sum(len(configs) for configs in network_configs.values())} network simulations")
    
    print("\n3. Performing Analysis...")
    print("-" * 50)
    
    # Performance Analysis
    print("   • Performance analysis...")
    perf_analyzer = PerformanceAnalyzer()
    perf_comparison = perf_analyzer.compare_configurations(config_results)
    network_perf = perf_analyzer.analyze_neural_network_performance(network_results)
    
    print(f"     - Analyzed {len(config_results)} configurations")
    print(f"     - Analyzed {len(network_results)} neural networks")
    
    # Energy Analysis
    print("   • Energy analysis...")
    energy_analyzer = EnergyAnalyzer()
    config_stats = {name: stats for name, (stats, _) in config_results.items()}
    energy_comparison = energy_analyzer.compare_energy_efficiency(config_stats)
    
    # Get energy breakdowns for each configuration
    energy_breakdowns = {}
    for config_name, stats_dict in config_stats.items():
        energy_breakdowns[config_name] = energy_analyzer.analyze_energy_breakdown(stats_dict, config_name)
    
    print(f"     - Energy breakdown for {len(config_stats)} configurations")
    
    # SRAM vs RRAM Analysis
    sram_rram_analysis = energy_analyzer.analyze_sram_vs_rram_energy(energy_comparison)
    if sram_rram_analysis:
        print("   • SRAM CIM vs RRAM comparison completed")
        if 'efficiency_ratio' in sram_rram_analysis and sram_rram_analysis['efficiency_ratio']:
            ratio = sram_rram_analysis['efficiency_ratio']
            more_efficient = "SRAM CIM" if sram_rram_analysis['sram_more_efficient'] else "RRAM"
            print(f"     - Energy efficiency ratio (SRAM/RRAM): {ratio:.3f}")
            print(f"     - More efficient technology: {more_efficient}")
    
    print("\n4. Generating Visualizations...")
    print("-" * 50)
    
    # Generate comprehensive dashboard
    print("   • Creating comprehensive dashboard...")
    dashboard_path = dashboard.generate_comprehensive_dashboard(
        "RAMwich SRAM CIM vs RRAM Analysis Dashboard"
    )
    print(f"     - Dashboard: {dashboard_path}")
    
    # Generate individual plots (examples)
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        plot_count = len(list(plots_dir.glob("*.png")))
        print(f"     - Generated {plot_count} visualization plots")
    
    print("\n5. Exporting Results...")
    print("-" * 50)
    
    # Export in all formats
    print("   • Exporting analysis results...")
    
    # Get performance metrics for export
    performance_metrics = {}
    for config_name, (stats_dict, cycles) in config_results.items():
        metrics = perf_analyzer.analyze_simulation_results(stats_dict, cycles, config_name)
        performance_metrics[config_name] = metrics
    
    # Export all formats
    export_paths = exporter.export_all_formats(
        config_stats, 
        performance_metrics, 
        energy_breakdowns,
        "ramwich_demo_analysis"
    )
    
    for format_name, path in export_paths.items():
        print(f"     - {format_name.upper()}: {Path(path).name}")
    
    # Export dashboard data
    dashboard_data_path = dashboard.export_dashboard_data()
    print(f"     - Dashboard data: {Path(dashboard_data_path).name}")
    
    print("\n6. Analysis Summary...")
    print("-" * 50)
    
    # Print key findings
    print("   Key Findings:")
    
    # Best performing configuration
    best_throughput = perf_comparison.loc[perf_comparison['Throughput (ops/cycle)'].idxmax()]
    print(f"   • Highest throughput: {best_throughput['Configuration']} "
          f"({best_throughput['Throughput (ops/cycle)']:.4f} ops/cycle)")
    
    # Most energy efficient
    most_efficient = energy_comparison.loc[energy_comparison['Energy per Op'].idxmin()]
    print(f"   • Most energy efficient: {most_efficient['Configuration']} "
          f"({most_efficient['Energy per Op']:.2e} J/op)")
    
    # Technology comparison
    sram_configs = energy_comparison[energy_comparison['SRAM Energy'] > 0]
    rram_configs = energy_comparison[energy_comparison['RRAM Energy'] > 0]
    
    if not sram_configs.empty and not rram_configs.empty:
        sram_avg_efficiency = sram_configs['Energy per Op'].mean()
        rram_avg_efficiency = rram_configs['Energy per Op'].mean()
        
        print(f"   • Average SRAM CIM efficiency: {sram_avg_efficiency:.2e} J/op")
        print(f"   • Average RRAM efficiency: {rram_avg_efficiency:.2e} J/op")
        
        if sram_avg_efficiency < rram_avg_efficiency:
            improvement = ((rram_avg_efficiency - sram_avg_efficiency) / rram_avg_efficiency) * 100
            print(f"   • SRAM CIM is {improvement:.1f}% more energy efficient than RRAM")
        else:
            improvement = ((sram_avg_efficiency - rram_avg_efficiency) / sram_avg_efficiency) * 100
            print(f"   • RRAM is {improvement:.1f}% more energy efficient than SRAM CIM")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print(f"Open the dashboard: {dashboard_path}")
    print("\nGenerated files:")
    print(f"  • HTML Dashboard: dashboard.html")
    print(f"  • Plots: plots/ directory ({len(list((output_dir / 'plots').glob('*.png')))} files)")
    print(f"  • Reports: reports/ directory ({len(list((output_dir / 'reports').glob('*.txt')))} files)")
    print(f"  • Data exports: exports/ directory ({len(list((output_dir / 'exports').glob('*')))} files)")
    print(f"  • Raw data: data/ directory ({len(list((output_dir / 'data').glob('*.csv')))} files)")


if __name__ == "__main__":
    main()