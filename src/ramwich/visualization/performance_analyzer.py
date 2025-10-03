#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from ..stats import StatsDict


@dataclass
class PerformanceMetrics:
    """Container for performance analysis metrics"""
    throughput: float  # Operations per second
    latency: float     # Average latency in cycles
    efficiency: float  # Computational efficiency (%)
    utilization: float # Hardware utilization (%)
    energy_per_op: float # Energy per operation
    area_efficiency: float # Operations per unit area


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis tool for RAMwich simulation results.
    
    Provides detailed analysis of timing, throughput, efficiency, and comparative metrics
    across different configurations and neural network architectures.
    """
    
    def __init__(self):
        self.results_history: List[Dict[str, Any]] = []
        self.comparison_data: Dict[str, List[PerformanceMetrics]] = {}
        
    def analyze_simulation_results(self, 
                                 stats_dict: StatsDict, 
                                 simulation_cycles: int,
                                 config_name: str = "default") -> PerformanceMetrics:
        """
        Analyze simulation results and compute performance metrics.
        
        Args:
            stats_dict: Statistics from RAMwich simulation
            simulation_cycles: Total simulation cycles
            config_name: Name of the configuration for tracking
            
        Returns:
            PerformanceMetrics object with computed metrics
        """
        # Calculate total operations
        total_ops = 0
        total_energy = 0
        total_area = 0
        
        for component_name, stats in stats_dict.items():
            total_ops += stats.activation_count
            total_energy += stats.dynamic_energy + stats.leakage_energy
            total_area += stats.area
            
        # Calculate performance metrics
        throughput = total_ops / max(simulation_cycles, 1)  # ops per cycle
        latency = simulation_cycles / max(total_ops, 1)     # cycles per op
        energy_per_op = total_energy / max(total_ops, 1)
        area_efficiency = total_ops / max(total_area, 1e-10)
        
        # Calculate efficiency (simplified as ops/cycle normalized)
        max_theoretical_throughput = 1.0  # Assume 1 op per cycle as theoretical max
        efficiency = min(100.0, (throughput / max_theoretical_throughput) * 100)
        
        # Calculate utilization (based on active vs idle cycles)
        utilization = min(100.0, (total_ops / max(simulation_cycles, 1)) * 100)
        
        metrics = PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            efficiency=efficiency,
            utilization=utilization,
            energy_per_op=energy_per_op,
            area_efficiency=area_efficiency
        )
        
        # Store results for historical analysis
        self.results_history.append({
            'config_name': config_name,
            'metrics': metrics,
            'stats_dict': stats_dict,
            'simulation_cycles': simulation_cycles
        })
        
        return metrics
    
    def compare_configurations(self, 
                             config_results: Dict[str, Tuple[StatsDict, int]]) -> pd.DataFrame:
        """
        Compare performance across multiple configurations.
        
        Args:
            config_results: Dict mapping config names to (stats_dict, cycles) tuples
            
        Returns:
            DataFrame with comparative analysis
        """
        comparison_data = []
        
        for config_name, (stats_dict, cycles) in config_results.items():
            metrics = self.analyze_simulation_results(stats_dict, cycles, config_name)
            
            comparison_data.append({
                'Configuration': config_name,
                'Throughput (ops/cycle)': metrics.throughput,
                'Latency (cycles/op)': metrics.latency,
                'Efficiency (%)': metrics.efficiency,
                'Utilization (%)': metrics.utilization,
                'Energy per Op': metrics.energy_per_op,
                'Area Efficiency': metrics.area_efficiency
            })
            
        return pd.DataFrame(comparison_data)
    
    def analyze_neural_network_performance(self, 
                                         network_results: Dict[str, Dict[str, Tuple[StatsDict, int]]]) -> pd.DataFrame:
        """
        Analyze performance across different neural network architectures.
        
        Args:
            network_results: Nested dict {network_name: {config_name: (stats_dict, cycles)}}
            
        Returns:
            DataFrame with network performance analysis
        """
        network_data = []
        
        for network_name, config_results in network_results.items():
            for config_name, (stats_dict, cycles) in config_results.items():
                metrics = self.analyze_simulation_results(stats_dict, cycles, f"{network_name}_{config_name}")
                
                network_data.append({
                    'Network': network_name,
                    'Configuration': config_name,
                    'Throughput': metrics.throughput,
                    'Latency': metrics.latency,
                    'Efficiency': metrics.efficiency,
                    'Energy_per_Op': metrics.energy_per_op,
                    'Area_Efficiency': metrics.area_efficiency
                })
                
        return pd.DataFrame(network_data)
    
    def plot_performance_comparison(self, 
                                  comparison_df: pd.DataFrame, 
                                  metric: str = 'Throughput',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create performance comparison plots.
        
        Args:
            comparison_df: DataFrame from compare_configurations or analyze_neural_network_performance
            metric: Metric to plot ('Throughput', 'Latency', 'Efficiency', etc.)
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'Network' in comparison_df.columns:
            # Network comparison plot
            sns.barplot(data=comparison_df, x='Network', y=metric, hue='Configuration', ax=ax)
            ax.set_title(f'{metric} Comparison Across Neural Networks')
            ax.set_xlabel('Neural Network Architecture')
        else:
            # Configuration comparison plot
            sns.barplot(data=comparison_df, x='Configuration', y=metric, ax=ax)
            ax.set_title(f'{metric} Comparison Across Configurations')
            ax.set_xlabel('Configuration')
            
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_efficiency_analysis(self, 
                               comparison_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive efficiency analysis plots.
        
        Args:
            comparison_df: DataFrame with performance metrics
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Throughput vs Energy efficiency
        has_legend_data = False
        if 'Network' in comparison_df.columns and len(comparison_df) > 0:
            networks = comparison_df['Network'].unique()
            for network in networks:
                network_data = comparison_df[comparison_df['Network'] == network]
                if len(network_data) > 0:
                    ax1.scatter(network_data['Energy_per_Op'], network_data['Throughput'], 
                               label=network, s=100, alpha=0.7)
                    has_legend_data = True
        elif len(comparison_df) > 0:
            # Only add label if we have data points
            ax1.scatter(comparison_df['Energy per Op'], comparison_df['Throughput (ops/cycle)'], 
                       s=100, alpha=0.7, label='Configurations')
            has_legend_data = True
            
        ax1.set_xlabel('Energy per Operation')
        ax1.set_ylabel('Throughput (ops/cycle)')
        ax1.set_title('Energy Efficiency vs Throughput')
        # Only show legend if we actually have labeled data
        if has_legend_data and len(ax1.get_children()) > 0:
            try:
                ax1.legend()
            except UserWarning:
                pass  # Ignore legend warnings
        ax1.grid(True, alpha=0.3)
        
        # Area efficiency comparison
        metric_col = 'Area_Efficiency' if 'Area_Efficiency' in comparison_df.columns else 'Area Efficiency'
        if 'Network' in comparison_df.columns:
            sns.barplot(data=comparison_df, x='Network', y=metric_col, hue='Configuration', ax=ax2)
        else:
            sns.barplot(data=comparison_df, x='Configuration', y=metric_col, ax=ax2)
        ax2.set_title('Area Efficiency Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Latency distribution
        latency_col = 'Latency' if 'Latency' in comparison_df.columns else 'Latency (cycles/op)'
        ax3.hist(comparison_df[latency_col], bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Latency (cycles/op)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Latency Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Efficiency vs Utilization
        eff_col = 'Efficiency' if 'Efficiency' in comparison_df.columns else 'Efficiency (%)'
        util_col = 'Utilization' if 'Utilization' in comparison_df.columns else 'Utilization (%)'
        
        if util_col in comparison_df.columns:
            ax4.scatter(comparison_df[util_col], comparison_df[eff_col], s=100, alpha=0.7)
            ax4.set_xlabel('Utilization (%)')
            ax4.set_ylabel('Efficiency (%)')
            ax4.set_title('Efficiency vs Utilization')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Utilization data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Utilization Analysis')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_performance_report(self, 
                                  comparison_df: pd.DataFrame,
                                  report_path: str = "performance_report.txt") -> str:
        """
        Generate a comprehensive performance analysis report.
        
        Args:
            comparison_df: DataFrame with performance metrics
            report_path: Path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAMwich Performance Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in comparison_df.columns:
                mean_val = comparison_df[col].mean()
                std_val = comparison_df[col].std()
                min_val = comparison_df[col].min()
                max_val = comparison_df[col].max()
                
                report_lines.append(f"{col}:")
                report_lines.append(f"  Mean: {mean_val:.6f}")
                report_lines.append(f"  Std:  {std_val:.6f}")
                report_lines.append(f"  Min:  {min_val:.6f}")
                report_lines.append(f"  Max:  {max_val:.6f}")
                report_lines.append("")
        
        # Best performing configurations
        report_lines.append("BEST PERFORMING CONFIGURATIONS")
        report_lines.append("-" * 40)
        
        if 'Throughput' in comparison_df.columns:
            best_throughput = comparison_df.loc[comparison_df['Throughput'].idxmax()]
            report_lines.append(f"Highest Throughput: {best_throughput.name}")
            report_lines.append(f"  Value: {best_throughput['Throughput']:.6f} ops/cycle")
            report_lines.append("")
        
        if 'Energy_per_Op' in comparison_df.columns:
            best_energy = comparison_df.loc[comparison_df['Energy_per_Op'].idxmin()]
            report_lines.append(f"Most Energy Efficient: {best_energy.name}")
            report_lines.append(f"  Energy per Op: {best_energy['Energy_per_Op']:.6f}")
            report_lines.append("")
        
        # Detailed results table
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(comparison_df.to_string())
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        return report_content
    
    def plot_timeline_analysis(self, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance metrics over time/iterations.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.results_history:
            raise ValueError("No historical data available. Run analyze_simulation_results first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract timeline data
        iterations = list(range(len(self.results_history)))
        throughputs = [result['metrics'].throughput for result in self.results_history]
        latencies = [result['metrics'].latency for result in self.results_history]
        efficiencies = [result['metrics'].efficiency for result in self.results_history]
        energy_per_ops = [result['metrics'].energy_per_op for result in self.results_history]
        
        # Plot trends
        ax1.plot(iterations, throughputs, 'o-', linewidth=2, markersize=6)
        ax1.set_title('Throughput Over Time')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Throughput (ops/cycle)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations, latencies, 'o-', linewidth=2, markersize=6, color='orange')
        ax2.set_title('Latency Over Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Latency (cycles/op)')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(iterations, efficiencies, 'o-', linewidth=2, markersize=6, color='green')
        ax3.set_title('Efficiency Over Time')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Efficiency (%)')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(iterations, energy_per_ops, 'o-', linewidth=2, markersize=6, color='red')
        ax4.set_title('Energy per Operation Over Time')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Energy per Op')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig