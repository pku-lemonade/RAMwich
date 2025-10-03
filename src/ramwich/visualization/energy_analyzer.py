#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from ..stats import StatsDict


@dataclass
class EnergyBreakdown:
    """Container for energy analysis breakdown"""
    component_name: str
    dynamic_energy: float
    leakage_energy: float
    total_energy: float
    percentage: float
    energy_per_activation: float


class EnergyAnalyzer:
    """
    Comprehensive energy analysis tool for RAMwich simulation results.
    
    Provides detailed energy breakdown, efficiency analysis, and power consumption
    visualization across different components and configurations.
    """
    
    def __init__(self):
        self.energy_history: List[Dict[str, Any]] = []
        
    def analyze_energy_breakdown(self, 
                                stats_dict: StatsDict,
                                analysis_name: str = "default") -> List[EnergyBreakdown]:
        """
        Analyze energy consumption breakdown by component.
        
        Args:
            stats_dict: Statistics from RAMwich simulation
            analysis_name: Name for this analysis
            
        Returns:
            List of EnergyBreakdown objects for each component
        """
        breakdowns = []
        total_system_energy = 0
        
        # First pass: calculate total energy
        for component_name, stats in stats_dict.items():
            total_energy = stats.dynamic_energy + stats.leakage_energy
            total_system_energy += total_energy
            
        # Second pass: create breakdowns with percentages
        for component_name, stats in stats_dict.items():
            dynamic_energy = stats.dynamic_energy
            leakage_energy = stats.leakage_energy
            total_energy = dynamic_energy + leakage_energy
            percentage = (total_energy / max(total_system_energy, 1e-10)) * 100
            energy_per_activation = total_energy / max(stats.activation_count, 1)
            
            breakdown = EnergyBreakdown(
                component_name=component_name,
                dynamic_energy=dynamic_energy,
                leakage_energy=leakage_energy,
                total_energy=total_energy,
                percentage=percentage,
                energy_per_activation=energy_per_activation
            )
            breakdowns.append(breakdown)
            
        # Store for historical analysis
        self.energy_history.append({
            'analysis_name': analysis_name,
            'breakdowns': breakdowns,
            'total_energy': total_system_energy
        })
        
        # Sort by total energy (highest first)
        breakdowns.sort(key=lambda x: x.total_energy, reverse=True)
        
        return breakdowns
    
    def compare_energy_efficiency(self, 
                                config_results: Dict[str, StatsDict]) -> pd.DataFrame:
        """
        Compare energy efficiency across multiple configurations.
        
        Args:
            config_results: Dict mapping config names to StatsDict
            
        Returns:
            DataFrame with energy efficiency comparison
        """
        comparison_data = []
        
        for config_name, stats_dict in config_results.items():
            breakdowns = self.analyze_energy_breakdown(stats_dict, config_name)
            
            total_energy = sum(b.total_energy for b in breakdowns)
            total_dynamic = sum(b.dynamic_energy for b in breakdowns)
            total_leakage = sum(b.leakage_energy for b in breakdowns)
            total_activations = sum(stats.activation_count for stats in stats_dict.values())
            
            # Find dominant components
            sram_energy = sum(b.total_energy for b in breakdowns if 'SRAM' in b.component_name.upper())
            rram_energy = sum(b.total_energy for b in breakdowns if 'RRAM' in b.component_name.upper())
            adc_energy = sum(b.total_energy for b in breakdowns if 'ADC' in b.component_name.upper())
            dac_energy = sum(b.total_energy for b in breakdowns if 'DAC' in b.component_name.upper())
            
            comparison_data.append({
                'Configuration': config_name,
                'Total Energy': total_energy,
                'Dynamic Energy': total_dynamic,
                'Leakage Energy': total_leakage,
                'Energy per Op': total_energy / max(total_activations, 1),
                'Dynamic Ratio (%)': (total_dynamic / max(total_energy, 1e-10)) * 100,
                'Leakage Ratio (%)': (total_leakage / max(total_energy, 1e-10)) * 100,
                'SRAM Energy': sram_energy,
                'RRAM Energy': rram_energy,
                'ADC Energy': adc_energy,
                'DAC Energy': dac_energy
            })
            
        return pd.DataFrame(comparison_data)
    
    def plot_energy_breakdown_pie(self, 
                                breakdowns: List[EnergyBreakdown],
                                title: str = "Energy Breakdown by Component",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create pie chart showing energy breakdown by component.
        
        Args:
            breakdowns: List of EnergyBreakdown objects
            title: Chart title
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for pie chart
        labels = [b.component_name for b in breakdowns]
        sizes = [b.percentage for b in breakdowns]
        colors = plt.cm.Set3(np.linspace(0, 1, len(breakdowns)))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 10})
        
        # Enhance appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend with energy values
        legend_labels = [f"{b.component_name}: {b.total_energy:.2e} J" for b in breakdowns]
        ax.legend(wedges, legend_labels, title="Components", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_energy_comparison_bar(self, 
                                 comparison_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart comparing energy consumption across configurations.
        
        Args:
            comparison_df: DataFrame from compare_energy_efficiency
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total energy comparison
        sns.barplot(data=comparison_df, x='Configuration', y='Total Energy', ax=ax1)
        ax1.set_title('Total Energy Consumption')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylabel('Energy (J)')
        
        # Dynamic vs Leakage energy
        energy_types = ['Dynamic Energy', 'Leakage Energy']
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax2.bar(x - width/2, comparison_df['Dynamic Energy'], width, label='Dynamic', alpha=0.8)
        ax2.bar(x + width/2, comparison_df['Leakage Energy'], width, label='Leakage', alpha=0.8)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Energy (J)')
        ax2.set_title('Dynamic vs Leakage Energy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df['Configuration'], rotation=45)
        ax2.legend()
        
        # Energy per operation
        sns.barplot(data=comparison_df, x='Configuration', y='Energy per Op', ax=ax3)
        ax3.set_title('Energy Efficiency (Energy per Operation)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylabel('Energy per Op (J/op)')
        
        # Component breakdown (stacked bar)
        component_cols = ['SRAM Energy', 'RRAM Energy', 'ADC Energy', 'DAC Energy']
        available_cols = [col for col in component_cols if col in comparison_df.columns]
        
        if available_cols:
            bottom = np.zeros(len(comparison_df))
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            
            for i, col in enumerate(available_cols):
                ax4.bar(comparison_df['Configuration'], comparison_df[col], 
                       bottom=bottom, label=col.replace(' Energy', ''), 
                       color=colors[i % len(colors)], alpha=0.8)
                bottom += comparison_df[col]
                
            ax4.set_title('Energy Breakdown by Component Type')
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Energy (J)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Component breakdown\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Component Energy Breakdown')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_power_analysis(self, 
                          breakdowns: List[EnergyBreakdown],
                          simulation_time: float = 1.0,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create power analysis plots (assuming simulation_time in seconds).
        
        Args:
            breakdowns: List of EnergyBreakdown objects
            simulation_time: Total simulation time in seconds
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Power consumption by component
        components = [b.component_name for b in breakdowns]
        dynamic_power = [b.dynamic_energy / simulation_time for b in breakdowns]
        leakage_power = [b.leakage_energy / simulation_time for b in breakdowns]
        
        x = np.arange(len(components))
        width = 0.35
        
        ax1.bar(x - width/2, dynamic_power, width, label='Dynamic Power', alpha=0.8)
        ax1.bar(x + width/2, leakage_power, width, label='Leakage Power', alpha=0.8)
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Power (W)')
        ax1.set_title('Power Consumption by Component')
        ax1.set_xticks(x)
        ax1.set_xticklabels(components, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Power efficiency (power per activation)
        power_per_activation = [b.energy_per_activation / simulation_time for b in breakdowns]
        
        ax2.bar(components, power_per_activation, alpha=0.8, color='green')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Power per Activation (W/activation)')
        ax2.set_title('Power Efficiency by Component')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def analyze_sram_vs_rram_energy(self, 
                                  comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze energy efficiency comparison between SRAM CIM and RRAM.
        
        Args:
            comparison_df: DataFrame with SRAM and RRAM energy data
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        if 'SRAM Energy' in comparison_df.columns and 'RRAM Energy' in comparison_df.columns:
            sram_configs = comparison_df[comparison_df['SRAM Energy'] > 0]
            rram_configs = comparison_df[comparison_df['RRAM Energy'] > 0]
            
            if not sram_configs.empty and not rram_configs.empty:
                analysis['sram_avg_energy'] = sram_configs['SRAM Energy'].mean()
                analysis['rram_avg_energy'] = rram_configs['RRAM Energy'].mean()
                analysis['sram_avg_efficiency'] = sram_configs['Energy per Op'].mean()
                analysis['rram_avg_efficiency'] = rram_configs['Energy per Op'].mean()
                
                # Calculate efficiency ratio
                if analysis['rram_avg_efficiency'] > 0:
                    analysis['efficiency_ratio'] = analysis['sram_avg_efficiency'] / analysis['rram_avg_efficiency']
                    analysis['sram_more_efficient'] = analysis['efficiency_ratio'] < 1.0
                else:
                    analysis['efficiency_ratio'] = None
                    analysis['sram_more_efficient'] = None
                    
                # Dynamic vs leakage analysis
                sram_dynamic_ratio = sram_configs['Dynamic Ratio (%)'].mean()
                rram_dynamic_ratio = rram_configs['Dynamic Ratio (%)'].mean()
                
                analysis['sram_dynamic_ratio'] = sram_dynamic_ratio
                analysis['rram_dynamic_ratio'] = rram_dynamic_ratio
                analysis['sram_more_dynamic'] = sram_dynamic_ratio > rram_dynamic_ratio
                
        return analysis
    
    def generate_energy_report(self, 
                             breakdowns: List[EnergyBreakdown],
                             comparison_df: Optional[pd.DataFrame] = None,
                             report_path: str = "energy_report.txt") -> str:
        """
        Generate comprehensive energy analysis report.
        
        Args:
            breakdowns: List of EnergyBreakdown objects
            comparison_df: Optional DataFrame with comparison data
            report_path: Path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAMwich Energy Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Energy breakdown summary
        report_lines.append("ENERGY BREAKDOWN BY COMPONENT")
        report_lines.append("-" * 50)
        
        total_energy = sum(b.total_energy for b in breakdowns)
        total_dynamic = sum(b.dynamic_energy for b in breakdowns)
        total_leakage = sum(b.leakage_energy for b in breakdowns)
        
        report_lines.append(f"Total System Energy: {total_energy:.6e} J")
        report_lines.append(f"Total Dynamic Energy: {total_dynamic:.6e} J ({(total_dynamic/total_energy)*100:.1f}%)")
        report_lines.append(f"Total Leakage Energy: {total_leakage:.6e} J ({(total_leakage/total_energy)*100:.1f}%)")
        report_lines.append("")
        
        for breakdown in breakdowns:
            report_lines.append(f"{breakdown.component_name}:")
            report_lines.append(f"  Total Energy: {breakdown.total_energy:.6e} J ({breakdown.percentage:.1f}%)")
            report_lines.append(f"  Dynamic: {breakdown.dynamic_energy:.6e} J")
            report_lines.append(f"  Leakage: {breakdown.leakage_energy:.6e} J")
            report_lines.append(f"  Energy per Activation: {breakdown.energy_per_activation:.6e} J/op")
            report_lines.append("")
        
        # Comparison analysis if available
        if comparison_df is not None:
            report_lines.append("CONFIGURATION COMPARISON")
            report_lines.append("-" * 40)
            
            # Find most efficient configuration
            most_efficient = comparison_df.loc[comparison_df['Energy per Op'].idxmin()]
            report_lines.append(f"Most Energy Efficient: {most_efficient['Configuration']}")
            report_lines.append(f"  Energy per Op: {most_efficient['Energy per Op']:.6e} J/op")
            report_lines.append("")
            
            # SRAM vs RRAM analysis
            sram_rram_analysis = self.analyze_sram_vs_rram_energy(comparison_df)
            if sram_rram_analysis:
                report_lines.append("SRAM CIM vs RRAM ANALYSIS")
                report_lines.append("-" * 30)
                
                if 'efficiency_ratio' in sram_rram_analysis and sram_rram_analysis['efficiency_ratio']:
                    ratio = sram_rram_analysis['efficiency_ratio']
                    more_efficient = "SRAM CIM" if sram_rram_analysis['sram_more_efficient'] else "RRAM"
                    report_lines.append(f"Energy Efficiency Ratio (SRAM/RRAM): {ratio:.3f}")
                    report_lines.append(f"More Efficient Technology: {more_efficient}")
                    report_lines.append("")
                
                if 'sram_dynamic_ratio' in sram_rram_analysis:
                    report_lines.append(f"SRAM Dynamic Energy Ratio: {sram_rram_analysis['sram_dynamic_ratio']:.1f}%")
                    report_lines.append(f"RRAM Dynamic Energy Ratio: {sram_rram_analysis['rram_dynamic_ratio']:.1f}%")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report to file
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        return report_content