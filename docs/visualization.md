# Visualization System

RAMwich provides comprehensive visualization and analysis tools to help you understand simulation results, compare configurations, and optimize your designs.

## 🎨 Overview

The visualization system offers:
- **Interactive Dashboards**: HTML-based analysis interfaces
- **Performance Analysis**: Throughput, latency, and efficiency metrics
- **Energy Analysis**: Component-wise power consumption breakdown
- **Comparative Studies**: Multi-configuration analysis
- **Export Capabilities**: Multiple output formats for further analysis

## 🚀 Quick Start

### Generate Basic Dashboard
```bash
python run.py --config config.yaml --ops ops.json --visualize
```

### Run Visualization Demos
```bash
# Quick demo
cd examples
python quick_visualization_demo.py

# Comprehensive analysis
python visualization_demo.py
```

## 📊 Dashboard Components

### Performance Analysis Dashboard

#### Throughput vs Energy Efficiency
- **X-axis**: Energy per operation (J/op)
- **Y-axis**: Throughput (operations/cycle)
- **Insights**: Identify optimal efficiency points

#### Latency Distribution
- **Histogram**: Distribution of operation latencies
- **Statistics**: Mean, median, 95th percentile
- **Bottleneck Analysis**: Identify performance limiters

#### Hardware Utilization
- **Utilization %**: MVMU, memory, NoC usage
- **Timeline View**: Utilization over simulation time
- **Resource Efficiency**: Identify underutilized components

### Energy Analysis Dashboard

#### Component Energy Breakdown
```
┌─────────────────────────────────────┐
│        Energy Breakdown             │
├─────────────────────────────────────┤
│  MVMU Computation    ████████ 45%   │
│  SRAM Access         ██████   30%   │
│  NoC Communication   ███     15%    │
│  Control Logic       ██      10%    │
└─────────────────────────────────────┘
```

#### Dynamic vs Leakage Power
- **Stacked Bar Charts**: Power consumption by component
- **Time Series**: Power consumption over time
- **Efficiency Metrics**: Power per operation

#### Technology Comparison
- **SRAM vs RRAM**: Energy efficiency comparison
- **Precision Analysis**: Impact of bit-width on power
- **Scaling Studies**: Power vs performance trade-offs

## 🔧 Programmatic Usage

### Basic Visualization
```python
from ramwich.visualization import VisualizationDashboard

# Create dashboard
dashboard = VisualizationDashboard("output_dir")

# Add simulation results
dashboard.add_simulation_result("Config1", stats_dict, cycles)

# Generate dashboard
html_path = dashboard.generate_comprehensive_dashboard("My Analysis")
```

### Performance Analysis
```python
from ramwich.visualization import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze single configuration
metrics = analyzer.analyze_performance(stats_dict, cycles)
print(f"Throughput: {metrics.throughput:.2f} ops/cycle")
print(f"Energy Efficiency: {metrics.energy_efficiency:.2e} J/op")

# Compare multiple configurations
comparison_df = analyzer.compare_configurations(multi_config_data)
```

### Energy Analysis
```python
from ramwich.visualization import EnergyAnalyzer

analyzer = EnergyAnalyzer()

# Component energy breakdown
breakdowns = analyzer.analyze_energy_breakdown(stats_dict, "SRAM_CIM")

# Technology comparison
comparison = analyzer.compare_energy_efficiency({
    "SRAM": sram_stats,
    "RRAM": rram_stats
})
```

### Export Results
```python
from ramwich.visualization import ResultsExporter

exporter = ResultsExporter("output_dir")

# Export all formats
export_paths = exporter.export_all_formats(
    config_stats, 
    base_filename="analysis"
)

# Available formats: CSV, Excel, JSON, Dataset, Summary
```

## 📈 Visualization Types

### Interactive Plots

#### Scatter Plots
- **Energy vs Throughput**: Efficiency analysis
- **Area vs Performance**: Design space exploration
- **Latency vs Utilization**: Resource optimization

#### Bar Charts
- **Component Comparison**: Energy, area, performance
- **Configuration Comparison**: Multi-config analysis
- **Technology Comparison**: SRAM vs RRAM vs Mixed

#### Time Series
- **Performance Timeline**: Metrics over simulation time
- **Power Consumption**: Dynamic power tracking
- **Utilization Timeline**: Resource usage patterns

#### Heatmaps
- **Configuration Space**: Parameter sweep results
- **Correlation Analysis**: Metric relationships
- **Bottleneck Analysis**: Performance hotspots

### Static Plots

#### Publication-Quality Figures
```python
# Generate publication plots
dashboard.create_publication_plots(
    output_dir="plots/",
    format="pdf",  # or "png", "svg"
    dpi=300
)
```

#### Custom Styling
```python
# Apply custom styling
dashboard.set_style({
    'figure_size': (10, 6),
    'color_palette': 'viridis',
    'font_size': 12,
    'line_width': 2
})
```

## 📋 Analysis Reports

### Automated Report Generation

#### Performance Report
```
================================================================================
PERFORMANCE ANALYSIS REPORT
================================================================================

Configuration: SRAM_CIM_4bit
Simulation Date: 2024-01-15 14:30:00

SUMMARY METRICS:
  Throughput:           16.58 ops/cycle
  Energy Efficiency:    7.25e-01 J/op
  Area Efficiency:      0.47 ops/cycle/mm²
  Average Latency:      0.06 cycles/op

COMPONENT BREAKDOWN:
  MVMU Utilization:     85.2%
  Memory Utilization:   67.8%
  NoC Utilization:      23.4%

RECOMMENDATIONS:
  • Consider increasing MVMU parallelism
  • Memory bandwidth is adequate
  • NoC has spare capacity for scaling
================================================================================
```

#### Energy Report
```
================================================================================
ENERGY ANALYSIS REPORT
================================================================================

Total Energy Consumption: 8.187e-05 J

BREAKDOWN BY COMPONENT:
  MVMU Computation:     3.456e-05 J (42.2%)
  SRAM Access:          2.134e-05 J (26.1%)
  NoC Communication:    1.234e-05 J (15.1%)
  Control Logic:        8.765e-06 J (10.7%)
  DRAM Access:          4.567e-06 J (5.6%)
  Other:                2.345e-07 J (0.3%)

POWER ANALYSIS:
  Average Power:        6.67e-04 W
  Peak Power:           1.23e-03 W
  Power Efficiency:     15.2 GOPS/W

OPTIMIZATION OPPORTUNITIES:
  • SRAM leakage dominates (93.1% of total)
  • Consider power gating for idle components
  • Dynamic voltage scaling could reduce active power
================================================================================
```

## 🎯 Advanced Features

### Multi-Configuration Analysis

#### Configuration Sweep Visualization
```python
# Analyze parameter sweep results
sweep_results = load_sweep_data("sweep_results/")
dashboard.create_sweep_analysis(sweep_results, parameters=[
    "num_tiles", "num_cores", "storage_technology"
])
```

#### Pareto Frontier Analysis
```python
# Find optimal configurations
pareto_configs = analyzer.find_pareto_frontier(
    configurations,
    objectives=["energy", "performance", "area"]
)
dashboard.plot_pareto_frontier(pareto_configs)
```

### Neural Network Analysis

#### Layer-wise Performance
```python
# Analyze per-layer metrics
layer_analysis = analyzer.analyze_neural_network(
    network_stats,
    network_type="LeNet-5"
)
dashboard.create_layer_analysis_dashboard(layer_analysis)
```

#### Architecture Comparison
```python
# Compare different neural networks
network_comparison = analyzer.compare_neural_networks({
    "LeNet-5": lenet_stats,
    "ResNet-20": resnet_stats,
    "MobileNet": mobilenet_stats
})
```

### Custom Analysis

#### Define Custom Metrics
```python
def custom_efficiency_metric(stats):
    """Custom efficiency calculation"""
    ops_per_second = stats['total_operations'] / stats['simulation_time']
    energy_per_op = stats['total_energy'] / stats['total_operations']
    return ops_per_second / energy_per_op

# Register custom metric
analyzer.register_custom_metric("custom_efficiency", custom_efficiency_metric)
```

#### Custom Visualizations
```python
def create_custom_plot(data):
    """Create custom visualization"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['x'], data['y'])
    ax.set_xlabel('Custom X Metric')
    ax.set_ylabel('Custom Y Metric')
    return fig

# Add to dashboard
dashboard.add_custom_plot("Custom Analysis", create_custom_plot)
```

## 📤 Export Options

### Data Export Formats

#### CSV Export
```csv
Configuration,Throughput,Energy,Area,Latency
SRAM_CIM_2bit,7.92,7.25e-01,35.52,0.126
RRAM_2bit,7.92,7.25e-01,35.52,0.126
Mixed_SRAM_RRAM,7.92,7.25e-01,35.52,0.126
```

#### Excel Export
- **Multiple Sheets**: Separate sheets for different analyses
- **Formatted Tables**: Professional formatting
- **Charts**: Embedded visualizations

#### JSON Export
```json
{
  "configurations": {
    "SRAM_CIM_2bit": {
      "throughput": 7.915,
      "energy_per_op": 7.25e-01,
      "total_area": 35.52,
      "components": {...}
    }
  },
  "analysis": {
    "best_energy": "SRAM_CIM_2bit",
    "best_performance": "SRAM_CIM_4bit",
    "pareto_optimal": ["SRAM_CIM_2bit", "Mixed_SRAM_RRAM"]
  }
}
```

### Research Dataset Export
```python
# Export research-ready dataset
exporter.export_research_dataset(
    results,
    metadata={
        "experiment": "SRAM CIM Analysis",
        "date": "2024-01-15",
        "parameters": ["storage_tech", "precision", "architecture"]
    },
    format="csv"  # Compatible with R, Python, MATLAB
)
```

## 🎨 Customization

### Styling Options

#### Color Schemes
```python
# Predefined color schemes
dashboard.set_color_scheme("academic")  # Black/white for papers
dashboard.set_color_scheme("colorful")  # Vibrant colors
dashboard.set_color_scheme("colorblind") # Colorblind-friendly
```

#### Layout Options
```python
# Dashboard layout
dashboard.set_layout({
    'columns': 2,
    'plot_height': 400,
    'plot_width': 600,
    'spacing': 20
})
```

### Interactive Features

#### Zoom and Pan
- **Mouse Controls**: Zoom with wheel, pan with drag
- **Keyboard Shortcuts**: Arrow keys for navigation
- **Reset View**: Double-click to reset zoom

#### Data Filtering
- **Configuration Filter**: Show/hide specific configurations
- **Metric Filter**: Focus on specific performance ranges
- **Component Filter**: Analyze specific hardware components

#### Export from Dashboard
- **Plot Export**: Save individual plots as PNG/PDF
- **Data Export**: Download underlying data as CSV
- **Report Export**: Generate PDF reports

## 🔍 Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Install visualization dependencies
pip install matplotlib seaborn plotly
```

#### Large Dataset Performance
```python
# Optimize for large datasets
dashboard.set_performance_mode(
    sample_size=1000,  # Subsample large datasets
    cache_plots=True,  # Cache generated plots
    lazy_loading=True  # Load data on demand
)
```

#### Memory Issues
```python
# Reduce memory usage
dashboard.set_memory_optimization(
    compress_data=True,
    streaming_mode=True,
    max_memory_gb=4
)
```

## 📚 Examples

### Complete Analysis Workflow
```python
from ramwich.visualization import *

# 1. Load simulation results
results = load_simulation_results("results/")

# 2. Create analyzers
perf_analyzer = PerformanceAnalyzer()
energy_analyzer = EnergyAnalyzer()

# 3. Perform analysis
performance_metrics = perf_analyzer.analyze_performance(results)
energy_breakdown = energy_analyzer.analyze_energy_breakdown(results)

# 4. Create dashboard
dashboard = VisualizationDashboard("analysis_output/")
dashboard.add_performance_analysis(performance_metrics)
dashboard.add_energy_analysis(energy_breakdown)

# 5. Generate outputs
html_path = dashboard.generate_comprehensive_dashboard("Complete Analysis")
dashboard.export_all_formats("analysis_results")

print(f"Analysis complete! Open {html_path} to view results.")
```