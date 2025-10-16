"""
Analysis script for DDP all-reduce benchmark results.

Generates:
1. Tables comparing different configurations
2. Plots showing performance trends
3. Key observations and insights
"""

import json
import csv
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Using basic CSV parsing.")


def load_results(csv_path):
    """Load benchmark results from CSV."""
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["data_size_mb"] = float(row["data_size_mb"])
            row["world_size"] = int(row["world_size"])
            row["mean_time"] = float(row["mean_time"])
            row["std_time"] = float(row["std_time"])
            row["min_time"] = float(row["min_time"])
            row["max_time"] = float(row["max_time"])
            row["bandwidth_mbps"] = float(row["bandwidth_mbps"])
            results.append(row)
    return results


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Backend':<10} {'Device':<8} {'Data Size':<12} {'Processes':<10} {'Mean Time':<12} {'Bandwidth':<15}")
    print("-" * 100)

    for r in results:
        data_size_str = (
            f"{r['data_size_mb']:.0f} MB" if r["data_size_mb"] < 1000 else f"{r['data_size_mb'] / 1000:.1f} GB"
        )
        print(
            f"{r['backend']:<10} {r['device']:<8} {data_size_str:<12} {r['world_size']:<10} "
            f"{r['mean_time']:.6f}s   {r['bandwidth_mbps']:>10.2f} MB/s"
        )
    print("=" * 100 + "\n")


def generate_observations(results):
    """Generate key observations from the benchmark results."""
    observations = []

    # Separate by backend
    gloo_results = [r for r in results if r["backend"] == "gloo"]
    nccl_results = [r for r in results if r["backend"] == "nccl"]

    print("\n" + "=" * 100)
    print("KEY OBSERVATIONS")
    print("=" * 100 + "\n")

    # Observation 1: Backend comparison
    if gloo_results and nccl_results:
        # Compare same config (if available)
        for world_size in [2, 4, 6]:
            gloo_1gb = [r for r in gloo_results if r["data_size_mb"] == 1000 and r["world_size"] == world_size]
            nccl_1gb = [r for r in nccl_results if r["data_size_mb"] == 1000 and r["world_size"] == world_size]

            if gloo_1gb and nccl_1gb:
                speedup = gloo_1gb[0]["mean_time"] / nccl_1gb[0]["mean_time"]
                obs = (
                    f"1. NCCL vs Gloo Performance (1GB, {world_size} processes):\n"
                    f"   - NCCL (GPU): {nccl_1gb[0]['mean_time']:.4f}s "
                    f"({nccl_1gb[0]['bandwidth_mbps']:.2f} MB/s)\n"
                    f"   - Gloo (CPU): {gloo_1gb[0]['mean_time']:.4f}s "
                    f"({gloo_1gb[0]['bandwidth_mbps']:.2f} MB/s)\n"
                    f"   - Speedup: {speedup:.2f}x faster with NCCL\n"
                    f"   - Insight: {'NCCL shows significant advantage for GPU-based all-reduce, especially with larger data' if speedup > 2 else 'Performance difference is moderate'}"
                )
                observations.append(obs)
                print(obs)
                break

    # Observation 2: Scaling with number of processes
    print("\n2. Scaling with Number of Processes:")
    for backend in ["gloo", "nccl"]:
        backend_results = [r for r in results if r["backend"] == backend and r["data_size_mb"] == 100]
        if backend_results:
            backend_results.sort(key=lambda x: x["world_size"])
            print(f"   {backend.upper()} + {'GPU' if backend == 'nccl' else 'CPU'} (100MB data):")
            for r in backend_results:
                print(f"     {r['world_size']} processes: {r['mean_time']:.6f}s ({r['bandwidth_mbps']:.2f} MB/s)")

            if len(backend_results) >= 2:
                time_increase = backend_results[-1]["mean_time"] / backend_results[0]["mean_time"]
                if time_increase < 1.5:
                    insight = "Good scaling - communication overhead is well managed"
                elif time_increase < 3:
                    insight = "Moderate scaling - some communication overhead with more processes"
                else:
                    insight = "Poor scaling - significant communication overhead"
                print(f"     Insight: {insight}")

    # Observation 3: Impact of data size
    print("\n3. Impact of Data Size on Performance:")
    for backend in ["gloo", "nccl"]:
        backend_2proc = [r for r in results if r["backend"] == backend and r["world_size"] == 2]
        if backend_2proc:
            backend_2proc.sort(key=lambda x: x["data_size_mb"])
            print(f"   {backend.upper()} + {'GPU' if backend == 'nccl' else 'CPU'} (2 processes):")
            for r in backend_2proc:
                data_str = (
                    f"{r['data_size_mb']:.0f}MB" if r["data_size_mb"] < 1000 else f"{r['data_size_mb'] / 1000:.1f}GB"
                )
                print(f"     {data_str:>6}: {r['mean_time']:.6f}s ({r['bandwidth_mbps']:.2f} MB/s)")

            # Check if bandwidth improves with larger data
            bandwidths = [r["bandwidth_mbps"] for r in backend_2proc]
            if len(bandwidths) >= 2 and bandwidths[-1] > bandwidths[0] * 1.2:
                print(
                    f"     Insight: Bandwidth increases with data size - better amortization of communication overhead"
                )
            elif len(bandwidths) >= 2 and bandwidths[-1] < bandwidths[0] * 0.8:
                print(f"     Insight: Bandwidth decreases with data size - hitting hardware limits")
            else:
                print(f"     Insight: Bandwidth relatively consistent across data sizes")

    print("\n" + "=" * 100 + "\n")

    return observations


def create_plots(results, output_dir):
    """Create visualization plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Separate by backend
    gloo_results = [r for r in results if r["backend"] == "gloo"]
    nccl_results = [r for r in results if r["backend"] == "nccl"]

    # Plot 1: Time vs Data Size for different world sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (backend, backend_results) in enumerate([("Gloo + CPU", gloo_results), ("NCCL + GPU", nccl_results)]):
        if not backend_results:
            continue

        ax = axes[idx]
        for world_size in [2, 4, 6]:
            data = [r for r in backend_results if r["world_size"] == world_size]
            if data:
                data.sort(key=lambda x: x["data_size_mb"])
                sizes = [r["data_size_mb"] for r in data]
                times = [r["mean_time"] for r in data]
                ax.plot(sizes, times, marker="o", label=f"{world_size} processes", linewidth=2)

        ax.set_xlabel("Data Size (MB)", fontsize=12)
        ax.set_ylabel("Mean Time (seconds)", fontsize=12)
        ax.set_title(f"{backend}", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_datasize.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'time_vs_datasize.png'}")
    plt.close()

    # Plot 2: Bandwidth comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    data_sizes = sorted(set(r["data_size_mb"] for r in results))
    width = 0.15
    x = range(len(data_sizes))

    for i, world_size in enumerate([2, 4, 6]):
        gloo_bw = []
        nccl_bw = []

        for data_size in data_sizes:
            gloo = [r for r in gloo_results if r["data_size_mb"] == data_size and r["world_size"] == world_size]
            nccl = [r for r in nccl_results if r["data_size_mb"] == data_size and r["world_size"] == world_size]

            gloo_bw.append(gloo[0]["bandwidth_mbps"] if gloo else 0)
            nccl_bw.append(nccl[0]["bandwidth_mbps"] if nccl else 0)

        offset = (i - 1) * width
        ax.bar([xi + offset - width / 2 for xi in x], gloo_bw, width, label=f"Gloo {world_size}proc", alpha=0.8)
        ax.bar([xi + offset + width / 2 for xi in x], nccl_bw, width, label=f"NCCL {world_size}proc", alpha=0.8)

    ax.set_xlabel("Data Size", fontsize=12)
    ax.set_ylabel("Bandwidth (MB/s)", fontsize=12)
    ax.set_title("All-Reduce Bandwidth Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(ds)}MB" if ds < 1000 else f"{ds / 1000:.1f}GB" for ds in data_sizes])
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "bandwidth_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'bandwidth_comparison.png'}")
    plt.close()

    # Plot 3: Scaling efficiency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (backend, backend_results, title) in enumerate(
        [("gloo", gloo_results, "Gloo + CPU"), ("nccl", nccl_results, "NCCL + GPU")]
    ):
        if not backend_results:
            continue

        ax = axes[idx]
        for data_size in [1, 10, 100, 1000]:
            data = [r for r in backend_results if r["data_size_mb"] == data_size]
            if data:
                data.sort(key=lambda x: x["world_size"])
                world_sizes = [r["world_size"] for r in data]
                # Ideal: time should stay constant (or decrease)
                # Reality: time increases due to overhead
                # Plot normalized time (relative to 2 processes)
                base_time = next((r["mean_time"] for r in data if r["world_size"] == 2), 1)
                norm_times = [r["mean_time"] / base_time for r in data]

                label = f"{int(data_size)}MB" if data_size < 1000 else f"{data_size / 1000:.1f}GB"
                ax.plot(world_sizes, norm_times, marker="o", label=label, linewidth=2)

        ax.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="Ideal (no overhead)")
        ax.set_xlabel("Number of Processes", fontsize=12)
        ax.set_ylabel("Normalized Time (relative to 2 proc)", fontsize=12)
        ax.set_title(f"Scaling Efficiency: {title}", fontsize=14, fontweight="bold")
        ax.set_xticks([2, 4, 6])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_efficiency.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'scaling_efficiency.png'}")
    plt.close()


def main():
    """Main analysis function."""
    results_dir = Path("benchmark_results_allreduce")
    csv_path = results_dir / "allreduce_benchmark_results.csv"

    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        print("Please run benchmark_allreduce.py first.")
        sys.exit(1)

    # Load results
    print(f"Loading results from {csv_path}...")
    results = load_results(csv_path)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results.\n")

    # Print summary table
    print_summary_table(results)

    # Generate observations
    generate_observations(results)

    # Create plots
    print("Generating plots...")
    create_plots(results, results_dir)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Results directory: {results_dir}")
    print(f"  - allreduce_benchmark_results.csv (data)")
    print(f"  - allreduce_benchmark_results.json (detailed data)")
    if HAS_MATPLOTLIB:
        print(f"  - time_vs_datasize.png (plot)")
        print(f"  - bandwidth_comparison.png (plot)")
        print(f"  - scaling_efficiency.png (plot)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
