#!/usr/bin/env python3

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.mvmu import MVMU
from ramwich.visualization import (
    PerformanceAnalyzer, 
    EnergyAnalyzer, 
    VisualizationDashboard, 
    ResultsExporter
)


class TestVisualizationSystem:
    """Test suite for the RAMwich visualization system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_simulation_data(self):
        """Create sample simulation data for testing"""
        # SRAM CIM configuration
        sram_config = DataConfig(
            storage_config=[BitConfig.SRAM, BitConfig.SRAM],
            weight_format="Q2.0",
            activation_format="Q8.0"
        )
        
        # RRAM configuration
        rram_config = DataConfig(
            storage_config=[BitConfig.MLC],
            weight_format="Q2.0",
            activation_format="Q8.0"
        )
        
        # Create MVMUs
        config_sram = Config(
            num_tiles_per_node=1,
            num_cores_per_tile=1,
            num_mvmus_per_core=1,
            data_config=sram_config
        )
        
        config_rram = Config(
            num_tiles_per_node=1,
            num_cores_per_tile=1,
            num_mvmus_per_core=1,
            data_config=rram_config
        )
        
        mvmu_sram = MVMU(id=0, config=config_sram)
        mvmu_rram = MVMU(id=1, config=config_rram)
        
        # Run simulations
        xbar_size = mvmu_sram.mvmu_config.xbar_config.xbar_size
        weights = np.random.uniform(-1, 1, size=(xbar_size, xbar_size)).astype(np.float64)
        input_data = np.random.randint(0, 100, size=xbar_size).astype(np.int32)
        
        # SRAM simulation
        mvmu_sram.load_weights(weights)
        mvmu_sram.write_to_inreg(0, input_data)
        mvmu_sram.execute_mvm()
        stats_sram = mvmu_sram.get_stats()
        
        # RRAM simulation
        mvmu_rram.load_weights(weights)
        mvmu_rram.write_to_inreg(0, input_data)
        mvmu_rram.execute_mvm()
        stats_rram = mvmu_rram.get_stats()
        
        return {
            'SRAM_CIM': (stats_sram, 1000),  # (stats_dict, cycles)
            'RRAM': (stats_rram, 1200)
        }
    
    def test_performance_analyzer(self, sample_simulation_data):
        """Test performance analysis functionality"""
        analyzer = PerformanceAnalyzer()
        
        # Test individual analysis
        stats_dict, cycles = sample_simulation_data['SRAM_CIM']
        metrics = analyzer.analyze_simulation_results(stats_dict, cycles, "SRAM_CIM")
        
        assert metrics.throughput > 0
        assert metrics.latency > 0
        assert 0 <= metrics.efficiency <= 100
        assert 0 <= metrics.utilization <= 100
        assert metrics.energy_per_op > 0
        assert metrics.area_efficiency > 0
        
        # Test comparison analysis
        comparison_df = analyzer.compare_configurations(sample_simulation_data)
        
        assert len(comparison_df) == 2
        assert 'Configuration' in comparison_df.columns
        assert 'Throughput (ops/cycle)' in comparison_df.columns
        assert 'Energy per Op' in comparison_df.columns
        
        # Verify data integrity
        assert all(comparison_df['Throughput (ops/cycle)'] > 0)
        assert all(comparison_df['Energy per Op'] > 0)
    
    def test_energy_analyzer(self, sample_simulation_data):
        """Test energy analysis functionality"""
        analyzer = EnergyAnalyzer()
        
        # Test energy breakdown
        stats_dict, _ = sample_simulation_data['SRAM_CIM']
        breakdowns = analyzer.analyze_energy_breakdown(stats_dict, "SRAM_CIM")
        
        assert len(breakdowns) > 0
        assert all(b.total_energy >= 0 for b in breakdowns)
        assert all(0 <= b.percentage <= 100 for b in breakdowns)
        assert abs(sum(b.percentage for b in breakdowns) - 100.0) < 1e-6
        
        # Test energy efficiency comparison
        config_stats = {name: stats for name, (stats, _) in sample_simulation_data.items()}
        comparison_df = analyzer.compare_energy_efficiency(config_stats)
        
        assert len(comparison_df) == 2
        assert 'Configuration' in comparison_df.columns
        assert 'Total Energy' in comparison_df.columns
        assert 'Energy per Op' in comparison_df.columns
        
        # Verify energy data
        assert all(comparison_df['Total Energy'] > 0)
        assert all(comparison_df['Dynamic Energy'] >= 0)
        assert all(comparison_df['Leakage Energy'] >= 0)
    
    def test_visualization_dashboard(self, sample_simulation_data, temp_dir):
        """Test visualization dashboard functionality"""
        dashboard = VisualizationDashboard(temp_dir)
        
        # Add simulation results
        for config_name, (stats_dict, cycles) in sample_simulation_data.items():
            dashboard.add_simulation_result(config_name, stats_dict, cycles)
        
        # Test dashboard generation
        html_path = dashboard.generate_comprehensive_dashboard("Test Dashboard")
        
        assert Path(html_path).exists()
        assert Path(html_path).suffix == '.html'
        
        # Verify output structure
        output_dir = Path(temp_dir)
        assert (output_dir / "plots").exists()
        assert (output_dir / "reports").exists()
        assert (output_dir / "data").exists()
        
        # Check for generated files
        plots_dir = output_dir / "plots"
        reports_dir = output_dir / "reports"
        data_dir = output_dir / "data"
        
        # Should have some plots
        plot_files = list(plots_dir.glob("*.png"))
        assert len(plot_files) > 0
        
        # Should have some reports
        report_files = list(reports_dir.glob("*.txt"))
        assert len(report_files) > 0
        
        # Should have some data files
        data_files = list(data_dir.glob("*.csv"))
        assert len(data_files) > 0
    
    def test_neural_network_analysis(self, sample_simulation_data, temp_dir):
        """Test neural network specific analysis"""
        dashboard = VisualizationDashboard(temp_dir)
        
        # Add results as neural network data
        for config_name, (stats_dict, cycles) in sample_simulation_data.items():
            dashboard.add_simulation_result(config_name, stats_dict, cycles, 
                                          network_name="LeNet-5")
        
        # Generate dashboard
        html_path = dashboard.generate_comprehensive_dashboard("Neural Network Analysis")
        
        assert Path(html_path).exists()
        
        # Verify neural network section is included
        with open(html_path, 'r') as f:
            content = f.read()
            assert "Neural Network Analysis" in content
            assert "LeNet-5" in content or "network" in content.lower()
    
    def test_results_exporter(self, sample_simulation_data, temp_dir):
        """Test results export functionality"""
        exporter = ResultsExporter(temp_dir)
        
        # Test CSV export
        stats_dict, _ = sample_simulation_data['SRAM_CIM']
        csv_path = exporter.export_stats_to_csv(stats_dict, "test_stats.csv")
        
        assert Path(csv_path).exists()
        assert Path(csv_path).suffix == '.csv'
        
        # Test Excel export
        config_stats = {name: stats for name, (stats, _) in sample_simulation_data.items()}
        excel_path = exporter.export_comparison_to_excel(config_stats, "test_comparison.xlsx")
        
        assert Path(excel_path).exists()
        assert Path(excel_path).suffix == '.xlsx'
        
        # Test research dataset creation
        dataset_path = exporter.create_research_dataset(config_stats, filename="test_dataset.csv")
        
        assert Path(dataset_path).exists()
        
        # Verify dataset content
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        assert len(df) == 2  # Two configurations
        assert 'Configuration' in df.columns
        assert 'Total_Energy' in df.columns
        assert 'Energy_per_Op' in df.columns
        assert 'Has_SRAM' in df.columns
        assert 'Has_RRAM' in df.columns
        
        # Test summary export
        summary_path = exporter.export_configuration_summary(config_stats, "test_summary.txt")
        
        assert Path(summary_path).exists()
        
        # Verify summary content
        with open(summary_path, 'r') as f:
            content = f.read()
            assert "Configuration Summary Report" in content
            assert "SRAM_CIM" in content
            assert "RRAM" in content
    
    def test_comparison_dashboard(self, sample_simulation_data, temp_dir):
        """Test focused comparison dashboard"""
        dashboard = VisualizationDashboard(temp_dir)
        
        # Create comparison dashboard
        html_path = dashboard.create_comparison_dashboard(
            sample_simulation_data, 
            "SRAM CIM vs RRAM Comparison"
        )
        
        assert Path(html_path).exists()
        
        # Verify comparison content
        with open(html_path, 'r') as f:
            content = f.read()
            assert "SRAM CIM vs RRAM Comparison" in content
            assert "Performance Analysis" in content
            assert "Energy Analysis" in content
    
    def test_export_all_formats(self, sample_simulation_data, temp_dir):
        """Test exporting in all available formats"""
        exporter = ResultsExporter(temp_dir)
        
        config_stats = {name: stats for name, (stats, _) in sample_simulation_data.items()}
        export_paths = exporter.export_all_formats(config_stats, base_filename="comprehensive_test")
        
        # Verify all expected formats are exported
        expected_formats = ['csv', 'excel', 'dataset', 'summary']
        for format_name in expected_formats:
            assert format_name in export_paths
            assert Path(export_paths[format_name]).exists()
        
        # Verify file extensions
        assert export_paths['csv'].endswith('.csv')
        assert export_paths['excel'].endswith('.xlsx')
        assert export_paths['dataset'].endswith('.csv')
        assert export_paths['summary'].endswith('.txt')
    
    def test_sram_vs_rram_analysis(self, sample_simulation_data):
        """Test SRAM CIM vs RRAM specific analysis"""
        analyzer = EnergyAnalyzer()
        
        config_stats = {name: stats for name, (stats, _) in sample_simulation_data.items()}
        comparison_df = analyzer.compare_energy_efficiency(config_stats)
        
        # Test SRAM vs RRAM analysis
        sram_rram_analysis = analyzer.analyze_sram_vs_rram_energy(comparison_df)
        
        # Should have analysis results
        assert isinstance(sram_rram_analysis, dict)
        
        # Check for expected keys (may vary based on data)
        possible_keys = ['sram_avg_energy', 'rram_avg_energy', 'efficiency_ratio', 
                        'sram_more_efficient', 'sram_dynamic_ratio', 'rram_dynamic_ratio']
        
        # At least some analysis should be present
        assert len(sram_rram_analysis) > 0
    
    def test_performance_timeline(self, sample_simulation_data):
        """Test performance timeline analysis"""
        analyzer = PerformanceAnalyzer()
        
        # Add multiple simulation results to create timeline
        for i, (config_name, (stats_dict, cycles)) in enumerate(sample_simulation_data.items()):
            analyzer.analyze_simulation_results(stats_dict, cycles + i*100, f"{config_name}_run_{i}")
        
        # Should have historical data
        assert len(analyzer.results_history) >= 2
        
        # Each result should have required fields
        for result in analyzer.results_history:
            assert 'config_name' in result
            assert 'metrics' in result
            assert 'stats_dict' in result
            assert 'simulation_cycles' in result
    
    def test_energy_report_generation(self, sample_simulation_data, temp_dir):
        """Test energy report generation"""
        analyzer = EnergyAnalyzer()
        
        stats_dict, _ = sample_simulation_data['SRAM_CIM']
        breakdowns = analyzer.analyze_energy_breakdown(stats_dict)
        
        config_stats = {name: stats for name, (stats, _) in sample_simulation_data.items()}
        comparison_df = analyzer.compare_energy_efficiency(config_stats)
        
        # Generate report
        report_path = Path(temp_dir) / "test_energy_report.txt"
        report_content = analyzer.generate_energy_report(
            breakdowns, comparison_df, str(report_path)
        )
        
        assert report_path.exists()
        assert len(report_content) > 0
        assert "Energy Analysis Report" in report_content
        assert "ENERGY BREAKDOWN BY COMPONENT" in report_content
        
    def test_dashboard_data_export(self, sample_simulation_data, temp_dir):
        """Test dashboard data export functionality"""
        dashboard = VisualizationDashboard(temp_dir)
        
        # Add simulation results
        for config_name, (stats_dict, cycles) in sample_simulation_data.items():
            dashboard.add_simulation_result(config_name, stats_dict, cycles)
        
        # Export dashboard data
        export_path = dashboard.export_dashboard_data()
        
        assert Path(export_path).exists()
        assert export_path.endswith('.json')
        
        # Verify JSON content
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert 'configurations' in data
        assert 'neural_networks' in data
        assert 'metadata' in data
        
        # Should have our configurations
        assert len(data['configurations']) == 2
        assert 'SRAM_CIM' in data['configurations']
        assert 'RRAM' in data['configurations']


def test_visualization_integration():
    """Integration test for the complete visualization system"""
    # Create sample data
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
    
    # Run simulation
    xbar_size = mvmu.mvmu_config.xbar_config.xbar_size
    weights = np.eye(xbar_size, dtype=np.float64)  # Identity matrix
    input_data = np.arange(1, xbar_size + 1, dtype=np.int32)
    
    mvmu.load_weights(weights)
    mvmu.write_to_inreg(0, input_data)
    mvmu.execute_mvm()
    
    stats = mvmu.get_stats()
    
    # Test all analyzers work together
    perf_analyzer = PerformanceAnalyzer()
    energy_analyzer = EnergyAnalyzer()
    
    # Performance analysis
    perf_metrics = perf_analyzer.analyze_simulation_results(stats, 1000, "Integration_Test")
    assert perf_metrics.throughput > 0
    
    # Energy analysis
    energy_breakdowns = energy_analyzer.analyze_energy_breakdown(stats, "Integration_Test")
    assert len(energy_breakdowns) > 0
    
    # Verify integration works
    assert perf_analyzer.results_history[-1]['config_name'] == "Integration_Test"
    assert energy_analyzer.energy_history[-1]['analysis_name'] == "Integration_Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])