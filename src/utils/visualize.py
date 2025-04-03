import logging

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def summarize_results(simulator_stats):
    """Print and visualize simulation results using the hierarchical stats object"""
    # Print summary statistics
    print("\n--- Simulation Results ---")

    if not simulator_stats:
        logger.warning("No statistics available from simulation")
        return

    # Extract overall statistics
    overall_stats = simulator_stats.get('stats', {})
    components = simulator_stats.get('components', [])

    print(f"Total operations completed: {sum(overall_stats.get('op_counts', {}).values())}")
    print(f"Total execution time: {overall_stats.get('total_execution_time', 0)}")

    # Create visualization
    visualize_results(simulator_stats)

def visualize_results(simulator_stats):
    """Print and visualize simulation results with component-specific statistics"""
    overall_stats = simulator_stats.get('stats', {})
    components = simulator_stats.get('components', [])

    if not overall_stats or not components:
        logger.warning("Insufficient statistics data for visualization")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot operation counts by type
    op_counts = overall_stats.get('op_counts', {})
    op_types = list(op_counts.keys())
    counts = list(op_counts.values())

    axs[0, 0].bar(op_types, counts)
    axs[0, 0].set_title('Operation Counts by Type')
    axs[0, 0].set_ylabel('Count')

    # Plot average execution times by type (if available)
    avg_times = [
        overall_stats.get('avg_load_time', 0),
        overall_stats.get('avg_set_time', 0),
        overall_stats.get('avg_alu_time', 0),
        overall_stats.get('avg_mvm_time', 0)
    ]

    axs[0, 1].bar(op_types, avg_times)
    axs[0, 1].set_title('Average Execution Time by Type')
    axs[0, 1].set_ylabel('Time (simulation units)')

    # Plot operations per component
    all_components = []
    all_op_counts = []

    # Collect operation stats from hierarchy
    for node_stats in components:
        node_id = node_stats.get('component_id', 'unknown')
        node_op_count = sum(node_stats.get('stats', {}).get('op_counts', {}).values())
        all_components.append(f"Node {node_id}")
        all_op_counts.append(node_op_count)

        # Process tiles within the node
        for tile_stats in node_stats.get('components', []):
            tile_id = tile_stats.get('component_id', 'unknown')
            tile_op_count = sum(tile_stats.get('stats', {}).get('op_counts', {}).values())
            all_components.append(f"Tile {node_id}.{tile_id}")
            all_op_counts.append(tile_op_count)

            # Process cores within the tile
            for core_stats in tile_stats.get('components', []):
                core_id = core_stats.get('component_id', 'unknown')
                core_op_count = sum(core_stats.get('stats', {}).get('op_counts', {}).values())
                all_components.append(f"Core {node_id}.{tile_id}.{core_id}")
                all_op_counts.append(core_op_count)

    # For a large number of components, only show top N
    if len(all_components) > 10:
        indices = np.argsort(all_op_counts)[-10:]  # Top 10
        all_components = [all_components[i] for i in indices]
        all_op_counts = [all_op_counts[i] for i in indices]

    axs[1, 0].barh(all_components, all_op_counts)
    axs[1, 0].set_title('Operations per Component')
    axs[1, 0].set_xlabel('Operations')

    # Timeline of operation completion (if available)
    if 'completion_times' in overall_stats:
        completion_times = np.array(overall_stats['completion_times'])
        op_indices = np.arange(len(completion_times))

        axs[1, 1].plot(op_indices, completion_times, marker='o', linestyle='-', alpha=0.7)
        axs[1, 1].set_title('Operation Completion Timeline')
        axs[1, 1].set_xlabel('Operation Index')
        axs[1, 1].set_ylabel('Completion Time')
    else:
        axs[1, 1].text(0.5, 0.5, 'Timeline data not available',
                      horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()

    # Print component-level statistics
    print("\n--- Component Statistics ---")
    _print_hierarchical_stats(components)

def _print_hierarchical_stats(components, indent=0):
    """Helper function to print hierarchical statistics with proper indentation"""
    for component in components:
        component_type = component.get('component_type', 'unknown')
        component_id = component.get('component_id', 'unknown')
        stats = component.get('stats', {})

        prefix = "  " * indent
        print(f"{prefix}{component_type.capitalize()} {component_id}:")
        print(f"{prefix}  Total operations: {sum(stats.get('op_counts', {}).values())}")
        print(f"{prefix}  Total execution time: {stats.get('total_execution_time', 0)}")

        if 'components' in component:
            _print_hierarchical_stats(component['components'], indent + 1)
