import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def summarize_results(stats, nodes):
    """Print and visualize simulation results"""
    if not stats['completion_times']:
        logger.warning("No operations were completed during simulation")
        return

    # Print summary statistics
    print("\n--- Simulation Results ---")
    print(f"Total operations completed: {sum(len(times) for times in stats.values()) - len(stats['completion_times'])}")
    print(f"Simulation end time: {max(stats['completion_times'])}")

    for op_type in ['load', 'set', 'alu', 'mvm']:
        times_key = f"{op_type}_times"
        if times_key in stats and stats[times_key]:
            avg_time = np.mean(stats[times_key])
            max_time = max(stats[times_key])
            print(f"{op_type.upper()} operations: {len(stats[times_key])}, Avg time: {avg_time:.2f}, Max time: {max_time}")

    # Create visualization
    visualize_results(stats, nodes)

def visualize_results(stats, nodes):
    """Print and visualize simulation results with component-specific statistics"""
    if not stats['completion_times']:
        logger.warning("No operations were completed during simulation")
        return

    # Create some visualizations of the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot operation counts by type
    op_types = ['load', 'set', 'alu', 'mvm']
    op_counts = [len(stats.get(f"{op}_times", [])) for op in op_types]

    axs[0, 0].bar(op_types, op_counts)
    axs[0, 0].set_title('Operation Counts by Type')
    axs[0, 0].set_ylabel('Count')

    # Plot average execution times by type
    avg_times = []
    for op in op_types:
        times = stats.get(f"{op}_times", [])
        avg_times.append(np.mean(times) if times else 0)

    axs[0, 1].bar(op_types, avg_times)
    axs[0, 1].set_title('Average Execution Time by Type')
    axs[0, 1].set_ylabel('Time (simulation units)')

    # Plot operations per tile/core
    all_components = []
    all_op_counts = []

    # Collect operation stats from hierarchy
    for node in nodes:
        node_stats = node.get_stats(include_components=False)

        for tile in node.tiles:
            tile_stats = tile.get_stats(include_components=False)
            component_label = f"Tile {tile.id}"
            all_components.append(component_label)
            all_op_counts.append(tile_stats['stats']['operations'])

            for core in tile.cores:
                core_stats = core.get_stats(include_components=False)
                component_label = f"Core {tile.id}.{core.id}"
                all_components.append(component_label)
                all_op_counts.append(core_stats['stats']['operations'])

    # For a large number of components, only show top N
    if len(all_components) > 10:
        indices = np.argsort(all_op_counts)[-10:]  # Top 10
        all_components = [all_components[i] for i in indices]
        all_op_counts = [all_op_counts[i] for i in indices]

    axs[1, 0].barh(all_components, all_op_counts)
    axs[1, 0].set_title('Operations per Component')
    axs[1, 0].set_xlabel('Operations')

    # Timeline of operation completion
    completion_times = np.array(stats['completion_times'])
    op_indices = np.arange(len(completion_times))

    axs[1, 1].plot(op_indices, completion_times, marker='o', linestyle='-', alpha=0.7)
    axs[1, 1].set_title('Operation Completion Timeline')
    axs[1, 1].set_xlabel('Operation Index')
    axs[1, 1].set_ylabel('Completion Time')

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()

    # Print component-level statistics
    print("\n--- Component Statistics ---")
    for node in nodes:
        node_stats = node.get_stats(include_components=False)
        print(f"\nNode {node.id}:")
        print(f"  Total operations: {node_stats['stats']['operations']}")
        print(f"  Total execution time: {node_stats['stats']['total_execution_time']}")

        for tile in node.tiles:
            tile_stats = tile.get_stats(include_components=False)
            print(f"\n  Tile {tile.id}:")
            print(f"    Total operations: {tile_stats['stats']['operations']}")
            print(f"    Total execution time: {tile_stats['stats']['total_execution_time']}")

            for core in tile.cores:
                core_stats = core.get_stats(include_components=False)
                print(f"\n    Core {core.id}:")
                print(f"      Operations: LOAD={core_stats['stats']['load_operations']}, "
                      f"SET={core_stats['stats']['set_operations']}, "
                      f"ALU={core_stats['stats']['alu_operations']}, "
                      f"MVM={core_stats['stats']['mvm_operations']}")
                print(f"      Total execution time: {core_stats['stats']['total_execution_time']}")
