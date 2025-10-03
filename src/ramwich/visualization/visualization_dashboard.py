#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
import json

from .performance_analyzer import PerformanceAnalyzer
from .energy_analyzer import EnergyAnalyzer
from ..stats import StatsDict


class VisualizationDashboard:
    """
    Comprehensive visualization dashboard for RAMwich simulation results.
    
    Provides an integrated interface for performance analysis, energy analysis,
    and comparative visualization across different configurations and neural networks.
    """
    
    def __init__(self, output_dir: str = "ramwich_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.energy_analyzer = EnergyAnalyzer()
        
        # Storage for dashboard data
        self.dashboard_data = {
            'configurations': {},
            'neural_networks': {},
            'comparisons': [],
            'metadata': {}
        }
        
    def add_simulation_result(self, 
                            config_name: str,
                            stats_dict: StatsDict,
                            simulation_cycles: int,
                            network_name: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None):
        """
        Add simulation results to the dashboard.
        
        Args:
            config_name: Name of the configuration
            stats_dict: Statistics from simulation
            simulation_cycles: Total simulation cycles
            network_name: Optional neural network name
            metadata: Optional additional metadata
        """
        result_data = {
            'stats_dict': stats_dict,
            'simulation_cycles': simulation_cycles,
            'metadata': metadata or {}
        }
        
        if network_name:
            if network_name not in self.dashboard_data['neural_networks']:
                self.dashboard_data['neural_networks'][network_name] = {}
            self.dashboard_data['neural_networks'][network_name][config_name] = result_data
        else:
            self.dashboard_data['configurations'][config_name] = result_data
            
    def generate_comprehensive_dashboard(self, 
                                       dashboard_title: str = "RAMwich Analysis Dashboard") -> str:
        """
        Generate a comprehensive analysis dashboard with all visualizations.
        
        Args:
            dashboard_title: Title for the dashboard
            
        Returns:
            Path to the generated HTML dashboard
        """
        # Create analysis directories
        plots_dir = self.output_dir / "plots"
        reports_dir = self.output_dir / "reports"
        data_dir = self.output_dir / "data"
        
        for dir_path in [plots_dir, reports_dir, data_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Generate all analyses and plots
        self._generate_all_analyses(plots_dir, reports_dir, data_dir)
        
        # Create HTML dashboard
        html_path = self.output_dir / "dashboard.html"
        self._create_html_dashboard(html_path, dashboard_title, plots_dir, reports_dir)
        
        return str(html_path)
    
    def _generate_all_analyses(self, plots_dir: Path, reports_dir: Path, data_dir: Path):
        """Generate all analysis plots and reports."""
        
        # Configuration comparison analysis
        if self.dashboard_data['configurations']:
            config_results = {
                name: (data['stats_dict'], data['simulation_cycles'])
                for name, data in self.dashboard_data['configurations'].items()
            }
            
            # Performance analysis
            perf_comparison = self.performance_analyzer.compare_configurations(config_results)
            perf_comparison.to_csv(data_dir / "performance_comparison.csv", index=False)
            
            # Generate performance plots
            for metric in ['Throughput (ops/cycle)', 'Latency (cycles/op)', 'Efficiency (%)', 'Energy per Op']:
                if metric in perf_comparison.columns:
                    fig = self.performance_analyzer.plot_performance_comparison(
                        perf_comparison, metric, 
                        save_path=plots_dir / f"performance_{metric.split(' ')[0].lower()}.png"
                    )
                    plt.close(fig)
            
            # Efficiency analysis
            fig = self.performance_analyzer.plot_efficiency_analysis(
                perf_comparison, save_path=plots_dir / "efficiency_analysis.png"
            )
            plt.close(fig)
            
            # Energy analysis
            energy_stats = {name: data['stats_dict'] for name, data in self.dashboard_data['configurations'].items()}
            energy_comparison = self.energy_analyzer.compare_energy_efficiency(energy_stats)
            energy_comparison.to_csv(data_dir / "energy_comparison.csv", index=False)
            
            # Energy plots
            fig = self.energy_analyzer.plot_energy_comparison_bar(
                energy_comparison, save_path=plots_dir / "energy_comparison.png"
            )
            plt.close(fig)
            
            # Generate reports
            perf_report = self.performance_analyzer.generate_performance_report(
                perf_comparison, str(reports_dir / "performance_report.txt")
            )
            
            # Energy breakdown for first configuration
            first_config = list(self.dashboard_data['configurations'].values())[0]
            energy_breakdowns = self.energy_analyzer.analyze_energy_breakdown(first_config['stats_dict'])
            
            fig = self.energy_analyzer.plot_energy_breakdown_pie(
                energy_breakdowns, save_path=plots_dir / "energy_breakdown_pie.png"
            )
            plt.close(fig)
            
            energy_report = self.energy_analyzer.generate_energy_report(
                energy_breakdowns, energy_comparison, str(reports_dir / "energy_report.txt")
            )
        
        # Neural network analysis
        if self.dashboard_data['neural_networks']:
            network_results = {
                network: {config: (data['stats_dict'], data['simulation_cycles'])
                         for config, data in configs.items()}
                for network, configs in self.dashboard_data['neural_networks'].items()
            }
            
            # Network performance analysis
            network_perf = self.performance_analyzer.analyze_neural_network_performance(network_results)
            network_perf.to_csv(data_dir / "network_performance.csv", index=False)
            
            # Network comparison plots
            for metric in ['Throughput', 'Latency', 'Efficiency', 'Energy_per_Op']:
                if metric in network_perf.columns:
                    fig = self.performance_analyzer.plot_performance_comparison(
                        network_perf, metric, 
                        save_path=plots_dir / f"network_{metric.lower()}.png"
                    )
                    plt.close(fig)
            
            # Network efficiency analysis
            fig = self.performance_analyzer.plot_efficiency_analysis(
                network_perf, save_path=plots_dir / "network_efficiency_analysis.png"
            )
            plt.close(fig)
        
        # Timeline analysis if available
        if self.performance_analyzer.results_history:
            fig = self.performance_analyzer.plot_timeline_analysis(
                save_path=plots_dir / "timeline_analysis.png"
            )
            plt.close(fig)
    
    def _create_html_dashboard(self, html_path: Path, title: str, plots_dir: Path, reports_dir: Path):
        """Create HTML dashboard file."""
        
        # Get relative paths for HTML
        plots_rel = plots_dir.relative_to(html_path.parent)
        reports_rel = reports_dir.relative_to(html_path.parent)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-container {{
            text-align: center;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .plot-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .report-link {{
            display: inline-block;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 5px;
            transition: background-color 0.3s;
        }}
        .report-link:hover {{
            background-color: #2980b9;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .navigation {{
            background-color: #34495e;
            padding: 15px;
            margin: -30px -30px 30px -30px;
            border-radius: 10px 10px 0 0;
        }}
        .nav-link {{
            color: white;
            text-decoration: none;
            margin: 0 15px;
            padding: 8px 15px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-link:hover {{
            background-color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <a href="#overview" class="nav-link">Overview</a>
            <a href="#performance" class="nav-link">Performance</a>
            <a href="#energy" class="nav-link">Energy</a>
            <a href="#networks" class="nav-link">Neural Networks</a>
            <a href="#reports" class="nav-link">Reports</a>
        </div>
        
        <h1>{title}</h1>
        
        <div id="overview" class="section">
            <h2>📊 Analysis Overview</h2>
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{len(self.dashboard_data['configurations'])}</div>
                    <div class="stat-label">Configurations Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(self.dashboard_data['neural_networks'])}</div>
                    <div class="stat-label">Neural Networks Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(self.performance_analyzer.results_history)}</div>
                    <div class="stat-label">Simulation Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(list(plots_dir.glob('*.png')))}</div>
                    <div class="stat-label">Generated Plots</div>
                </div>
            </div>
        </div>
        
        <div id="performance" class="section">
            <h2>⚡ Performance Analysis</h2>
            <div class="plot-grid">
        """
        
        # Add performance plots
        perf_plots = [
            ("performance_throughput.png", "Throughput Comparison"),
            ("performance_latency.png", "Latency Analysis"),
            ("performance_efficiency.png", "Efficiency Metrics"),
            ("efficiency_analysis.png", "Comprehensive Efficiency Analysis"),
            ("timeline_analysis.png", "Performance Timeline")
        ]
        
        for plot_file, plot_title in perf_plots:
            plot_path = plots_dir / plot_file
            if plot_path.exists():
                html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_title}</div>
                    <img src="{plots_rel / plot_file}" alt="{plot_title}">
                </div>
                """
        
        html_content += """
            </div>
        </div>
        
        <div id="energy" class="section">
            <h2>🔋 Energy Analysis</h2>
            <div class="plot-grid">
        """
        
        # Add energy plots
        energy_plots = [
            ("energy_comparison.png", "Energy Consumption Comparison"),
            ("energy_breakdown_pie.png", "Energy Breakdown by Component"),
        ]
        
        for plot_file, plot_title in energy_plots:
            plot_path = plots_dir / plot_file
            if plot_path.exists():
                html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_title}</div>
                    <img src="{plots_rel / plot_file}" alt="{plot_title}">
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
        
        # Add neural networks section if available
        if self.dashboard_data['neural_networks']:
            html_content += """
            <div id="networks" class="section">
                <h2>🧠 Neural Network Analysis</h2>
                <div class="plot-grid">
            """
            
            network_plots = [
                ("network_throughput.png", "Network Throughput Comparison"),
                ("network_latency.png", "Network Latency Analysis"),
                ("network_efficiency.png", "Network Efficiency Metrics"),
                ("network_efficiency_analysis.png", "Comprehensive Network Analysis")
            ]
            
            for plot_file, plot_title in network_plots:
                plot_path = plots_dir / plot_file
                if plot_path.exists():
                    html_content += f"""
                    <div class="plot-container">
                        <div class="plot-title">{plot_title}</div>
                        <img src="{plots_rel / plot_file}" alt="{plot_title}">
                    </div>
                    """
            
            html_content += """
                </div>
            </div>
            """
        
        # Add reports section
        html_content += f"""
        <div id="reports" class="section">
            <h2>📋 Detailed Reports</h2>
            <p>Download comprehensive analysis reports:</p>
        """
        
        # Add report links
        report_files = [
            ("performance_report.txt", "Performance Analysis Report"),
            ("energy_report.txt", "Energy Analysis Report")
        ]
        
        for report_file, report_title in report_files:
            report_path = reports_dir / report_file
            if report_path.exists():
                html_content += f"""
                <a href="{reports_rel / report_file}" class="report-link" download>{report_title}</a>
                """
        
        html_content += """
        </div>
        
        <div class="section">
            <h2>ℹ️ About This Analysis</h2>
            <p>This dashboard was generated by the RAMwich visualization system. It provides comprehensive analysis of:</p>
            <ul>
                <li><strong>Performance Metrics:</strong> Throughput, latency, efficiency, and utilization analysis</li>
                <li><strong>Energy Analysis:</strong> Power consumption, energy breakdown, and efficiency comparisons</li>
                <li><strong>Neural Network Evaluation:</strong> Architecture-specific performance and energy analysis</li>
                <li><strong>Comparative Studies:</strong> SRAM CIM vs RRAM technology comparisons</li>
            </ul>
            <p>All plots and reports are available for download and further analysis.</p>
        </div>
    </div>
    
    <script>
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
        """
        
        # Write HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def create_comparison_dashboard(self, 
                                  comparison_configs: Dict[str, Tuple[StatsDict, int]],
                                  title: str = "Configuration Comparison Dashboard") -> str:
        """
        Create a focused comparison dashboard for specific configurations.
        
        Args:
            comparison_configs: Dict mapping config names to (stats_dict, cycles) tuples
            title: Dashboard title
            
        Returns:
            Path to generated dashboard
        """
        # Clear existing data and add comparison configs
        self.dashboard_data['configurations'] = {}
        
        for config_name, (stats_dict, cycles) in comparison_configs.items():
            self.add_simulation_result(config_name, stats_dict, cycles)
        
        return self.generate_comprehensive_dashboard(title)
    
    def export_dashboard_data(self, export_path: str = None) -> str:
        """
        Export dashboard data to JSON for external analysis.
        
        Args:
            export_path: Optional path for export file
            
        Returns:
            Path to exported JSON file
        """
        if export_path is None:
            export_path = str(self.output_dir / "dashboard_data.json")
        
        # Prepare data for JSON serialization (convert StatsDict to regular dict)
        export_data = {
            'configurations': {},
            'neural_networks': {},
            'metadata': self.dashboard_data['metadata']
        }
        
        # Convert configurations
        for config_name, data in self.dashboard_data['configurations'].items():
            stats_dict = data['stats_dict']
            export_data['configurations'][config_name] = {
                'simulation_cycles': data['simulation_cycles'],
                'metadata': data['metadata'],
                'stats': {
                    component: {
                        'activation_count': stats.activation_count,
                        'dynamic_energy': stats.dynamic_energy,
                        'leakage_energy': stats.leakage_energy,
                        'area': stats.area
                    }
                    for component, stats in stats_dict.items()
                }
            }
        
        # Convert neural networks
        for network_name, configs in self.dashboard_data['neural_networks'].items():
            export_data['neural_networks'][network_name] = {}
            for config_name, data in configs.items():
                stats_dict = data['stats_dict']
                export_data['neural_networks'][network_name][config_name] = {
                    'simulation_cycles': data['simulation_cycles'],
                    'metadata': data['metadata'],
                    'stats': {
                        component: {
                            'activation_count': stats.activation_count,
                            'dynamic_energy': stats.dynamic_energy,
                            'leakage_energy': stats.leakage_energy,
                            'area': stats.area
                        }
                        for component, stats in stats_dict.items()
                    }
                }
        
        # Write JSON file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_path