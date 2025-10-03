#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import csv
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from ..stats import StatsDict
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .energy_analyzer import EnergyAnalyzer, EnergyBreakdown


class ResultsExporter:
    """
    Comprehensive results export utility for RAMwich simulation data.
    
    Supports multiple export formats including CSV, JSON, Excel, and custom formats
    for integration with external analysis tools and research workflows.
    """
    
    def __init__(self, output_dir: str = "ramwich_exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_stats_to_csv(self, 
                           stats_dict: StatsDict, 
                           filename: str = "simulation_stats.csv") -> str:
        """
        Export simulation statistics to CSV format.
        
        Args:
            stats_dict: Statistics from RAMwich simulation
            filename: Output filename
            
        Returns:
            Path to exported CSV file
        """
        export_path = self.output_dir / filename
        
        # Prepare data for CSV export
        csv_data = []
        for component_name, stats in stats_dict.items():
            csv_data.append({
                'Component': component_name,
                'Activation_Count': stats.activation_count,
                'Dynamic_Energy': stats.dynamic_energy,
                'Leakage_Energy': stats.leakage_energy,
                'Total_Energy': stats.dynamic_energy + stats.leakage_energy,
                'Area': stats.area,
                'Energy_per_Activation': (stats.dynamic_energy + stats.leakage_energy) / max(stats.activation_count, 1)
            })
        
        # Write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(export_path, index=False)
        
        return str(export_path)
    
    def export_comparison_to_excel(self, 
                                 comparison_data: Dict[str, StatsDict],
                                 filename: str = "comparison_analysis.xlsx") -> str:
        """
        Export comparison data to Excel with multiple sheets.
        
        Args:
            comparison_data: Dict mapping config names to StatsDict
            filename: Output filename
            
        Returns:
            Path to exported Excel file
        """
        export_path = self.output_dir / filename
        
        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for config_name, stats_dict in comparison_data.items():
                total_energy = sum(stats.dynamic_energy + stats.leakage_energy for stats in stats_dict.values())
                total_activations = sum(stats.activation_count for stats in stats_dict.values())
                total_area = sum(stats.area for stats in stats_dict.values())
                
                summary_data.append({
                    'Configuration': config_name,
                    'Total_Energy': total_energy,
                    'Total_Activations': total_activations,
                    'Total_Area': total_area,
                    'Energy_per_Op': total_energy / max(total_activations, 1),
                    'Area_Efficiency': total_activations / max(total_area, 1e-10)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual configuration sheets
            for config_name, stats_dict in comparison_data.items():
                config_data = []
                for component_name, stats in stats_dict.items():
                    config_data.append({
                        'Component': component_name,
                        'Activation_Count': stats.activation_count,
                        'Dynamic_Energy': stats.dynamic_energy,
                        'Leakage_Energy': stats.leakage_energy,
                        'Total_Energy': stats.dynamic_energy + stats.leakage_energy,
                        'Area': stats.area
                    })
                
                config_df = pd.DataFrame(config_data)
                # Excel sheet names have character limits
                sheet_name = config_name[:31] if len(config_name) > 31 else config_name
                config_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return str(export_path)
    
    def export_performance_metrics(self, 
                                 performance_metrics: List[PerformanceMetrics],
                                 config_names: List[str],
                                 filename: str = "performance_metrics.json") -> str:
        """
        Export performance metrics to JSON format.
        
        Args:
            performance_metrics: List of PerformanceMetrics objects
            config_names: Corresponding configuration names
            filename: Output filename
            
        Returns:
            Path to exported JSON file
        """
        export_path = self.output_dir / filename
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': []
        }
        
        for config_name, metrics in zip(config_names, performance_metrics):
            export_data['metrics'].append({
                'configuration': config_name,
                'throughput': metrics.throughput,
                'latency': metrics.latency,
                'efficiency': metrics.efficiency,
                'utilization': metrics.utilization,
                'energy_per_op': metrics.energy_per_op,
                'area_efficiency': metrics.area_efficiency
            })
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_path)
    
    def export_energy_breakdown(self, 
                              energy_breakdowns: List[EnergyBreakdown],
                              filename: str = "energy_breakdown.csv") -> str:
        """
        Export energy breakdown analysis to CSV.
        
        Args:
            energy_breakdowns: List of EnergyBreakdown objects
            filename: Output filename
            
        Returns:
            Path to exported CSV file
        """
        export_path = self.output_dir / filename
        
        csv_data = []
        for breakdown in energy_breakdowns:
            csv_data.append({
                'Component': breakdown.component_name,
                'Dynamic_Energy': breakdown.dynamic_energy,
                'Leakage_Energy': breakdown.leakage_energy,
                'Total_Energy': breakdown.total_energy,
                'Percentage': breakdown.percentage,
                'Energy_per_Activation': breakdown.energy_per_activation
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(export_path, index=False)
        
        return str(export_path)
    
    def export_neural_network_results(self, 
                                     network_results: Dict[str, Dict[str, Any]],
                                     filename: str = "neural_network_results.json") -> str:
        """
        Export neural network analysis results to JSON.
        
        Args:
            network_results: Nested dict with network analysis results
            filename: Output filename
            
        Returns:
            Path to exported JSON file
        """
        export_path = self.output_dir / filename
        
        # Convert any non-serializable objects to serializable format
        serializable_results = {}
        
        for network_name, network_data in network_results.items():
            serializable_results[network_name] = {}
            
            for config_name, config_data in network_data.items():
                if isinstance(config_data, tuple) and len(config_data) == 2:
                    # Handle (stats_dict, cycles) tuple
                    stats_dict, cycles = config_data
                    serializable_results[network_name][config_name] = {
                        'simulation_cycles': cycles,
                        'components': {
                            component: {
                                'activation_count': stats.activation_count,
                                'dynamic_energy': stats.dynamic_energy,
                                'leakage_energy': stats.leakage_energy,
                                'area': stats.area
                            }
                            for component, stats in stats_dict.items()
                        }
                    }
                else:
                    # Handle other data formats
                    serializable_results[network_name][config_name] = config_data
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'neural_networks': serializable_results
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_path)
    
    def export_plots_metadata(self, 
                            plots_info: List[Dict[str, Any]],
                            filename: str = "plots_metadata.json") -> str:
        """
        Export metadata about generated plots.
        
        Args:
            plots_info: List of plot information dictionaries
            filename: Output filename
            
        Returns:
            Path to exported JSON file
        """
        export_path = self.output_dir / filename
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'plots': plots_info
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(export_path)
    
    def create_research_dataset(self, 
                              comparison_data: Dict[str, StatsDict],
                              performance_metrics: Optional[Dict[str, PerformanceMetrics]] = None,
                              energy_breakdowns: Optional[Dict[str, List[EnergyBreakdown]]] = None,
                              filename: str = "research_dataset.csv") -> str:
        """
        Create a comprehensive research dataset combining all metrics.
        
        Args:
            comparison_data: Configuration comparison data
            performance_metrics: Optional performance metrics
            energy_breakdowns: Optional energy breakdown data
            filename: Output filename
            
        Returns:
            Path to exported research dataset
        """
        export_path = self.output_dir / filename
        
        dataset_rows = []
        
        for config_name, stats_dict in comparison_data.items():
            # Basic statistics
            total_energy = sum(stats.dynamic_energy + stats.leakage_energy for stats in stats_dict.values())
            total_dynamic = sum(stats.dynamic_energy for stats in stats_dict.values())
            total_leakage = sum(stats.leakage_energy for stats in stats_dict.values())
            total_activations = sum(stats.activation_count for stats in stats_dict.values())
            total_area = sum(stats.area for stats in stats_dict.values())
            
            # Component-specific metrics
            sram_energy = sum(stats.dynamic_energy + stats.leakage_energy 
                            for name, stats in stats_dict.items() 
                            if 'SRAM' in name.upper())
            rram_energy = sum(stats.dynamic_energy + stats.leakage_energy 
                            for name, stats in stats_dict.items() 
                            if 'RRAM' in name.upper())
            adc_energy = sum(stats.dynamic_energy + stats.leakage_energy 
                           for name, stats in stats_dict.items() 
                           if 'ADC' in name.upper())
            dac_energy = sum(stats.dynamic_energy + stats.leakage_energy 
                           for name, stats in stats_dict.items() 
                           if 'DAC' in name.upper())
            
            row_data = {
                'Configuration': config_name,
                'Total_Energy': total_energy,
                'Dynamic_Energy': total_dynamic,
                'Leakage_Energy': total_leakage,
                'Total_Activations': total_activations,
                'Total_Area': total_area,
                'Energy_per_Op': total_energy / max(total_activations, 1),
                'Area_Efficiency': total_activations / max(total_area, 1e-10),
                'Dynamic_Ratio': (total_dynamic / max(total_energy, 1e-10)) * 100,
                'Leakage_Ratio': (total_leakage / max(total_energy, 1e-10)) * 100,
                'SRAM_Energy': sram_energy,
                'RRAM_Energy': rram_energy,
                'ADC_Energy': adc_energy,
                'DAC_Energy': dac_energy,
                'Has_SRAM': 1 if sram_energy > 0 else 0,
                'Has_RRAM': 1 if rram_energy > 0 else 0
            }
            
            # Add performance metrics if available
            if performance_metrics and config_name in performance_metrics:
                perf = performance_metrics[config_name]
                row_data.update({
                    'Throughput': perf.throughput,
                    'Latency': perf.latency,
                    'Efficiency': perf.efficiency,
                    'Utilization': perf.utilization
                })
            
            # Add energy breakdown metrics if available
            if energy_breakdowns and config_name in energy_breakdowns:
                breakdowns = energy_breakdowns[config_name]
                # Add dominant component information
                if breakdowns:
                    dominant = max(breakdowns, key=lambda x: x.total_energy)
                    row_data.update({
                        'Dominant_Component': dominant.component_name,
                        'Dominant_Energy_Percentage': dominant.percentage
                    })
            
            dataset_rows.append(row_data)
        
        # Create DataFrame and export
        df = pd.DataFrame(dataset_rows)
        df.to_csv(export_path, index=False)
        
        return str(export_path)
    
    def export_configuration_summary(self, 
                                   comparison_data: Dict[str, StatsDict],
                                   filename: str = "configuration_summary.txt") -> str:
        """
        Export a human-readable configuration summary report.
        
        Args:
            comparison_data: Configuration comparison data
            filename: Output filename
            
        Returns:
            Path to exported summary file
        """
        export_path = self.output_dir / filename
        
        lines = []
        lines.append("=" * 80)
        lines.append("RAMwich Configuration Summary Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Configurations Analyzed: {len(comparison_data)}")
        lines.append("")
        
        # Overall statistics
        all_energies = []
        all_areas = []
        all_activations = []
        
        for stats_dict in comparison_data.values():
            total_energy = sum(stats.dynamic_energy + stats.leakage_energy for stats in stats_dict.values())
            total_area = sum(stats.area for stats in stats_dict.values())
            total_activations = sum(stats.activation_count for stats in stats_dict.values())
            
            all_energies.append(total_energy)
            all_areas.append(total_area)
            all_activations.append(total_activations)
        
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Energy Range: {min(all_energies):.2e} - {max(all_energies):.2e} J")
        lines.append(f"Area Range: {min(all_areas):.2e} - {max(all_areas):.2e} units")
        lines.append(f"Operations Range: {min(all_activations)} - {max(all_activations)}")
        lines.append("")
        
        # Configuration details
        lines.append("CONFIGURATION DETAILS")
        lines.append("-" * 40)
        
        for config_name, stats_dict in comparison_data.items():
            lines.append(f"\n{config_name}:")
            lines.append("-" * len(config_name))
            
            total_energy = sum(stats.dynamic_energy + stats.leakage_energy for stats in stats_dict.values())
            total_area = sum(stats.area for stats in stats_dict.values())
            total_activations = sum(stats.activation_count for stats in stats_dict.values())
            
            lines.append(f"  Total Energy: {total_energy:.2e} J")
            lines.append(f"  Total Area: {total_area:.2e} units")
            lines.append(f"  Total Operations: {total_activations}")
            lines.append(f"  Energy Efficiency: {total_energy/max(total_activations,1):.2e} J/op")
            lines.append(f"  Area Efficiency: {total_activations/max(total_area,1e-10):.2e} ops/unit")
            
            # Component breakdown
            lines.append("  Components:")
            for component_name, stats in stats_dict.items():
                component_energy = stats.dynamic_energy + stats.leakage_energy
                percentage = (component_energy / max(total_energy, 1e-10)) * 100
                lines.append(f"    {component_name}: {component_energy:.2e} J ({percentage:.1f}%)")
        
        # Write to file
        with open(export_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(export_path)
    
    def export_all_formats(self, 
                         comparison_data: Dict[str, StatsDict],
                         performance_metrics: Optional[Dict[str, PerformanceMetrics]] = None,
                         energy_breakdowns: Optional[Dict[str, List[EnergyBreakdown]]] = None,
                         base_filename: str = "ramwich_analysis") -> Dict[str, str]:
        """
        Export data in all available formats.
        
        Args:
            comparison_data: Configuration comparison data
            performance_metrics: Optional performance metrics
            energy_breakdowns: Optional energy breakdown data
            base_filename: Base filename for exports
            
        Returns:
            Dictionary mapping format names to export paths
        """
        export_paths = {}
        
        # CSV export
        first_config = list(comparison_data.values())[0]
        export_paths['csv'] = self.export_stats_to_csv(first_config, f"{base_filename}.csv")
        
        # Excel export
        export_paths['excel'] = self.export_comparison_to_excel(comparison_data, f"{base_filename}.xlsx")
        
        # Research dataset
        export_paths['dataset'] = self.create_research_dataset(
            comparison_data, performance_metrics, energy_breakdowns, f"{base_filename}_dataset.csv"
        )
        
        # Summary report
        export_paths['summary'] = self.export_configuration_summary(comparison_data, f"{base_filename}_summary.txt")
        
        # Performance metrics (if available)
        if performance_metrics:
            config_names = list(performance_metrics.keys())
            metrics_list = list(performance_metrics.values())
            export_paths['performance'] = self.export_performance_metrics(
                metrics_list, config_names, f"{base_filename}_performance.json"
            )
        
        # Energy breakdowns (if available)
        if energy_breakdowns:
            # Export first breakdown as example
            first_breakdown = list(energy_breakdowns.values())[0]
            export_paths['energy'] = self.export_energy_breakdown(first_breakdown, f"{base_filename}_energy.csv")
        
        return export_paths