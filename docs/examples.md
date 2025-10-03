# Examples and Tutorials

This guide provides comprehensive examples for using RAMwich, from basic simulations to advanced analysis workflows.

## 🚀 Quick Start Examples

### Basic Simulation
```bash
# Run a simple SRAM CIM simulation
python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json
```

### With Visualization
```bash
# Generate analysis dashboard
python run.py --config examples/mlp_l4_mnist/config.yaml --ops examples/mlp_l4_mnist/ops.json --visualize
```

### Complete Analysis
```bash
# Full simulation with all outputs
python run.py --config examples/mlp_l4_mnist/config.yaml \
              --ops examples/mlp_l4_mnist/ops.json \
              --weights examples/mlp_l4_mnist/weights.npz \
              --activation examples/mlp_l4_mnist/activation.npy \
              --visualize --json-output results.json
```

## 📊 Visualization Examples

### Quick Visualization Demo
```bash
cd examples
python quick_visualization_demo.py
```

**What it does:**
- Creates SRAM CIM configuration
- Runs performance simulation
- Generates interactive HTML dashboard
- Exports analysis in multiple formats

**Output:**
- `quick_demo_output/dashboard.html` - Interactive dashboard
- 8 visualization plots
- 2 detailed analysis reports

### Comprehensive Analysis Demo
```bash
cd examples
python visualization_demo.py
```

**What it does:**
- Compares 4 different configurations
- Tests LeNet-5 and ResNet-20 architectures
- Performs SRAM CIM vs RRAM analysis
- Creates comprehensive visualization suite

**Output:**
- `visualization_demo_output/dashboard.html` - Full analysis dashboard
- 13 visualization plots
- Multiple export formats (CSV, Excel, JSON)
- Detailed performance and energy reports

## 🧠 Neural Network Examples

### LeNet-5 Simulation
```python
from ramwich.config import Config
from ramwich.config.data_config import DataConfig, BitConfig
from ramwich.simulator import Simulator

# LeNet-5 SRAM CIM configuration
data_config = DataConfig(
    storage_config=[BitConfig.SRAM, BitConfig.SRAM],
    weight_format="Q2.0",
    activation_format="Q8.0"
)

config = Config(
    num_tiles_per_node=2,
    num_cores_per_tile=4,
    num_mvmus_per_core=2,
    data_config=data_config
)

# Run simulation
simulator = Simulator(config)
results = simulator.run_neural_network("LeNet-5", input_data)
```

### ResNet-20 with Mixed Precision
```python
# ResNet-20 with SRAM CIM + RRAM hybrid
mixed_config = DataConfig(
    storage_config=[BitConfig.SRAM, BitConfig.MLC],  # Mixed technology
    weight_format="Q3.0",
    activation_format="Q8.0"
)

config = Config(
    num_tiles_per_node=4,
    num_cores_per_tile=8,
    num_mvmus_per_core=2,
    data_config=mixed_config
)

simulator = Simulator(config)
results = simulator.run_neural_network("ResNet-20", input_data)
```

## 📈 Analysis Examples

### Performance Analysis
```python
from ramwich.visualization import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze single configuration
metrics = analyzer.analyze_performance(simulation_stats, cycles)
print(f"Throughput: {metrics.throughput:.2f} ops/cycle")
print(f"Energy Efficiency: {metrics.energy_efficiency:.2e} J/op")

# Compare multiple configurations
comparison = analyzer.compare_configurations({
    "SRAM_2bit": sram_stats,
    "RRAM_2bit": rram_stats,
    "Mixed": mixed_stats
})
```

### Energy Analysis
```python
from ramwich.visualization import EnergyAnalyzer

analyzer = EnergyAnalyzer()

# Component energy breakdown
breakdowns = analyzer.analyze_energy_breakdown(stats, "SRAM_CIM")
for breakdown in breakdowns:
    print(f"{breakdown.component_name}: {breakdown.total_energy:.2e} J ({breakdown.percentage:.1f}%)")

# Technology comparison
comparison = analyzer.compare_energy_efficiency({
    "SRAM_CIM": sram_stats,
    "RRAM": rram_stats
})
```

### Custom Analysis
```python
from ramwich.visualization import VisualizationDashboard

# Create custom dashboard
dashboard = VisualizationDashboard("custom_output")

# Add multiple simulation results
for config_name, (stats, cycles) in simulation_results.items():
    dashboard.add_simulation_result(config_name, stats, cycles)

# Generate comprehensive analysis
html_path = dashboard.generate_comprehensive_dashboard("Custom Analysis")
print(f"Dashboard created: {html_path}")
```

## 🔧 Configuration Examples

### SRAM CIM Configuration
```yaml
# sram_cim_config.yaml
num_tiles_per_node: 2
num_cores_per_tile: 4
num_mvmus_per_core: 2

data_config:
  storage_config: ["SRAM", "SRAM"]
  weight_format: "Q2.0"
  activation_format: "Q8.0"

mvmu_config:
  xbar_size: 128
  adc_resolution: 8
  dac_resolution: 8
```

### Mixed Technology Configuration
```yaml
# mixed_config.yaml
num_tiles_per_node: 4
num_cores_per_tile: 8
num_mvmus_per_core: 1

data_config:
  storage_config: ["SRAM", "MLC"]  # SRAM + RRAM hybrid
  weight_format: "Q3.0"
  activation_format: "Q8.0"

mvmu_config:
  xbar_size: 256
  adc_resolution: 8
  dac_resolution: 8
```

### High-Performance Configuration
```yaml
# high_perf_config.yaml
num_tiles_per_node: 8
num_cores_per_tile: 8
num_mvmus_per_core: 4

data_config:
  storage_config: ["SRAM", "SRAM"]
  weight_format: "Q4.0"
  activation_format: "Q8.0"

mvmu_config:
  xbar_size: 256
  adc_resolution: 8
  dac_resolution: 8
```

## 📊 Batch Processing Examples

### Configuration Sweep
```python
import yaml
import subprocess
from itertools import product

# Parameter ranges
tiles = [1, 2, 4, 8]
storage_configs = [["SRAM", "SRAM"], ["RRAM", "RRAM"], ["SRAM", "MLC"]]
weight_formats = ["Q2.0", "Q3.0", "Q4.0"]

base_config = {
    'num_tiles_per_node': 1,
    'num_cores_per_tile': 4,
    'num_mvmus_per_core': 2,
    'data_config': {
        'storage_config': ["SRAM", "SRAM"],
        'weight_format': "Q2.0",
        'activation_format': "Q8.0"
    }
}

results = []
for i, (t, s, w) in enumerate(product(tiles, storage_configs, weight_formats)):
    config = base_config.copy()
    config['num_tiles_per_node'] = t
    config['data_config']['storage_config'] = s
    config['data_config']['weight_format'] = w
    
    # Save configuration
    config_file = f"sweep_config_{i}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run simulation
    result = subprocess.run([
        'python', 'run.py',
        '--config', config_file,
        '--ops', 'ops.json',
        '--quiet',
        '--json-output', f'sweep_results_{i}.json'
    ], capture_output=True, text=True)
    
    results.append({
        'config_id': i,
        'tiles': t,
        'storage': s,
        'weight_format': w,
        'success': result.returncode == 0
    })

print(f"Completed {len(results)} simulations")
```

### Parallel Batch Processing
```python
import concurrent.futures
import json

def run_simulation(config_params):
    """Run a single simulation with given parameters"""
    config_id, config_file, ops_file = config_params
    
    result = subprocess.run([
        'python', 'run.py',
        '--config', config_file,
        '--ops', ops_file,
        '--quiet',
        '--json-output', f'results_{config_id}.json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        with open(f'results_{config_id}.json', 'r') as f:
            return json.load(f)
    else:
        return None

# Prepare simulation parameters
simulation_params = [
    (i, f'config_{i}.yaml', 'ops.json') 
    for i in range(10)
]

# Run simulations in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_simulation, simulation_params))

# Filter successful results
successful_results = [r for r in results if r is not None]
print(f"Successful simulations: {len(successful_results)}")
```

## 📈 Analysis Workflows

### Complete Analysis Pipeline
```python
from ramwich.visualization import *
import pandas as pd

def complete_analysis_workflow(simulation_results, output_dir):
    """Complete analysis workflow from simulation results to final report"""
    
    # 1. Initialize analyzers
    perf_analyzer = PerformanceAnalyzer()
    energy_analyzer = EnergyAnalyzer()
    dashboard = VisualizationDashboard(output_dir)
    exporter = ResultsExporter(output_dir)
    
    # 2. Perform analysis
    performance_metrics = {}
    energy_breakdowns = {}
    
    for config_name, (stats, cycles) in simulation_results.items():
        # Performance analysis
        perf_metrics = perf_analyzer.analyze_performance(stats, cycles)
        performance_metrics[config_name] = perf_metrics
        
        # Energy analysis
        energy_breakdown = energy_analyzer.analyze_energy_breakdown(stats, config_name)
        energy_breakdowns[config_name] = energy_breakdown
        
        # Add to dashboard
        dashboard.add_simulation_result(config_name, stats, cycles)
    
    # 3. Comparative analysis
    comparison_df = perf_analyzer.compare_configurations(simulation_results)
    energy_comparison = energy_analyzer.compare_energy_efficiency(
        {name: stats for name, (stats, _) in simulation_results.items()}
    )
    
    # 4. Generate visualizations
    html_path = dashboard.generate_comprehensive_dashboard("Complete Analysis")
    
    # 5. Export results
    export_paths = exporter.export_all_formats(
        {name: stats for name, (stats, _) in simulation_results.items()},
        base_filename="complete_analysis"
    )
    
    # 6. Generate summary report
    summary = generate_summary_report(performance_metrics, energy_breakdowns, comparison_df)
    
    return {
        'dashboard_path': html_path,
        'export_paths': export_paths,
        'summary': summary,
        'performance_metrics': performance_metrics,
        'energy_breakdowns': energy_breakdowns
    }

def generate_summary_report(performance_metrics, energy_breakdowns, comparison_df):
    """Generate a text summary report"""
    report = []
    report.append("=" * 80)
    report.append("RAMWICH ANALYSIS SUMMARY REPORT")
    report.append("=" * 80)
    
    # Best performing configurations
    best_throughput = comparison_df.loc[comparison_df['Throughput (ops/cycle)'].idxmax()]
    best_energy = comparison_df.loc[comparison_df['Energy per Op'].idxmin()]
    
    report.append(f"\nBEST CONFIGURATIONS:")
    report.append(f"  Highest Throughput: {best_throughput['Configuration']} ({best_throughput['Throughput (ops/cycle)']:.2f} ops/cycle)")
    report.append(f"  Most Energy Efficient: {best_energy['Configuration']} ({best_energy['Energy per Op']:.2e} J/op)")
    
    # Configuration comparison
    report.append(f"\nCONFIGURATION COMPARISON:")
    for _, row in comparison_df.iterrows():
        report.append(f"  {row['Configuration']}:")
        report.append(f"    Throughput: {row['Throughput (ops/cycle)']:.2f} ops/cycle")
        report.append(f"    Energy: {row['Energy per Op']:.2e} J/op")
        report.append(f"    Area: {row['Area Efficiency']:.2f} ops/cycle/mm²")
    
    report.append("=" * 80)
    return "\n".join(report)

# Usage example
simulation_results = load_simulation_results("batch_results/")
analysis_results = complete_analysis_workflow(simulation_results, "final_analysis/")
print(f"Analysis complete! Dashboard: {analysis_results['dashboard_path']}")
```

### Research Publication Workflow
```python
def research_publication_workflow(results, paper_title):
    """Generate publication-ready figures and data"""
    
    # Create publication directory
    pub_dir = f"publication_{paper_title.replace(' ', '_').lower()}"
    os.makedirs(pub_dir, exist_ok=True)
    
    # Generate high-quality figures
    dashboard = VisualizationDashboard(pub_dir)
    dashboard.set_style({
        'figure_size': (8, 6),
        'dpi': 300,
        'color_scheme': 'academic',
        'font_size': 14
    })
    
    # Create specific plots for paper
    plots = [
        'energy_efficiency_comparison',
        'throughput_vs_area',
        'technology_comparison',
        'scaling_analysis'
    ]
    
    for plot_type in plots:
        fig_path = dashboard.create_publication_plot(results, plot_type, format='pdf')
        print(f"Generated: {fig_path}")
    
    # Export data tables
    exporter = ResultsExporter(pub_dir)
    data_files = exporter.export_research_dataset(
        results,
        metadata={
            'title': paper_title,
            'date': datetime.now().isoformat(),
            'description': 'SRAM CIM simulation results'
        }
    )
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    with open(f"{pub_dir}/results_table.tex", 'w') as f:
        f.write(latex_table)
    
    return pub_dir

def generate_latex_table(results):
    """Generate LaTeX table for publication"""
    # Implementation for LaTeX table generation
    pass
```

## 🎯 Best Practices

### Configuration Management
```python
# Use configuration templates
def create_config_from_template(template_name, **overrides):
    """Create configuration from template with overrides"""
    templates = {
        'sram_cim_basic': {
            'num_tiles_per_node': 1,
            'num_cores_per_tile': 4,
            'data_config': {
                'storage_config': ['SRAM', 'SRAM'],
                'weight_format': 'Q2.0'
            }
        },
        'high_performance': {
            'num_tiles_per_node': 8,
            'num_cores_per_tile': 8,
            'data_config': {
                'storage_config': ['SRAM', 'SRAM'],
                'weight_format': 'Q4.0'
            }
        }
    }
    
    config = templates[template_name].copy()
    config.update(overrides)
    return config
```

### Error Handling
```python
def robust_simulation(config_file, ops_file, max_retries=3):
    """Run simulation with error handling and retries"""
    for attempt in range(max_retries):
        try:
            result = subprocess.run([
                'python', 'run.py',
                '--config', config_file,
                '--ops', ops_file,
                '--json-output', 'results.json'
            ], check=True, capture_output=True, text=True)
            
            # Load and validate results
            with open('results.json', 'r') as f:
                data = json.load(f)
            
            if validate_results(data):
                return data
            else:
                raise ValueError("Invalid simulation results")
                
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Brief delay before retry

def validate_results(data):
    """Validate simulation results"""
    required_fields = ['total_dynamic_energy', 'total_leakage_energy', 'total_area']
    return all(field in data for field in required_fields)
```

### Performance Optimization
```python
# Use caching for repeated analyses
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_analysis(config_hash, stats_hash):
    """Cache analysis results for repeated configurations"""
    # Perform expensive analysis
    pass

# Batch processing optimization
def optimized_batch_processing(configs, max_workers=None):
    """Optimized batch processing with resource management"""
    if max_workers is None:
        max_workers = min(len(configs), os.cpu_count())
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_simulation, config): config 
            for config in configs
        }
        
        # Process results as they complete
        results = {}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            try:
                result = future.result()
                results[config['name']] = result
            except Exception as e:
                print(f"Configuration {config['name']} failed: {e}")
    
    return results
```