"""
Comprehensive benchmark script for DDP all-reduce operations.

Tests various combinations of:
- Backend: Gloo (CPU), NCCL (GPU)
- Data sizes: 1MB, 10MB, 100MB, 1GB
- Number of processes: 2, 4, 6

Results are saved to CSV for analysis.
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend, master_port="29507"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def benchmark_allreduce(
    rank, world_size, backend, device_type, data_size_mb, warmup_steps, measurement_steps, master_port, results_dict
):
    """
    Run all-reduce benchmark for given configuration.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend to use (gloo or nccl)
        device_type: Device type (cpu or cuda)
        data_size_mb: Size of data in MB
        warmup_steps: Number of warmup iterations
        measurement_steps: Number of measurement iterations
        master_port: Port for master process
        results_dict: Shared dictionary to store results

    Returns:
        List of timing measurements (only from rank 0)
    """
    try:
        setup(rank, world_size, backend, master_port)

        # Calculate number of float32 elements for target size
        # 1 MB = 1024 * 1024 bytes, float32 = 4 bytes
        num_elements = int(data_size_mb * 1024 * 1024 / 4)

        # Set device
        if device_type == "cuda":
            if not torch.cuda.is_available():
                if rank == 0:
                    print(f"CUDA not available, skipping GPU test")
                return None
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(device)
            dist.barrier(device_ids=[rank])
        else:
            device = torch.device("cpu")

        # Warmup phase
        for _ in range(warmup_steps):
            data = torch.rand(num_elements, dtype=torch.float32, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dist.all_reduce(data, async_op=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Measurement phase
        times = []
        for _ in range(measurement_steps):
            data = torch.rand(num_elements, dtype=torch.float32, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            dist.all_reduce(data, async_op=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Convert times to tensor for gathering (NCCL requires tensors, not Python objects)
        times_tensor = torch.tensor(times, dtype=torch.float64, device=device)

        # Gather all times from all ranks using tensor operations
        all_times_tensor = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        dist.all_gather(all_times_tensor, times_tensor)

        # Convert back to lists
        all_times = [[t.item() for t in rank_times] for rank_times in all_times_tensor]

        dist.destroy_process_group()

        # Store results in shared dict (only from rank 0)
        if rank == 0:
            # Compute mean time across all ranks for each iteration
            avg_times = []
            for i in range(measurement_steps):
                iter_times = [all_times[r][i] for r in range(world_size)]
                avg_times.append(sum(iter_times) / len(iter_times))
            results_dict["times"] = avg_times

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark: {e}")


def run_benchmark_config(
    backend, device_type, data_size_mb, world_size, warmup_steps=3, measurement_steps=10, config_idx=0
):
    """
    Run benchmark for a specific configuration using multiprocessing.

    Returns:
        Dictionary with results
    """
    # Use sequential port allocation to avoid conflicts
    # Each config gets its own port, incrementing from base
    master_port = str(30000 + config_idx)

    print(f"\n{'=' * 70}")
    print(f"Config: {backend.upper()} + {device_type.upper()} | Data: {data_size_mb}MB | Processes: {world_size}")
    print(f"{'=' * 70}")

    # Run using multiprocessing
    try:
        # Use a manager to collect results
        from multiprocessing import Manager

        manager = Manager()
        results_dict = manager.dict()

        mp.spawn(
            fn=benchmark_allreduce,
            args=(
                world_size,
                backend,
                device_type,
                data_size_mb,
                warmup_steps,
                measurement_steps,
                master_port,
                results_dict,
            ),
            nprocs=world_size,
            join=True,
        )

        # Give time for port cleanup before next configuration
        time.sleep(2)

        if "times" in results_dict:
            times = list(results_dict["times"])
            mean_time = sum(times) / len(times)
            std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
            min_time = min(times)
            max_time = max(times)

            # Calculate bandwidth (MB/s)
            bandwidth_mbps = data_size_mb / mean_time

            print(f"  Mean: {mean_time:.6f}s | Std: {std_time:.6f}s | Min: {min_time:.6f}s | Max: {max_time:.6f}s")
            print(f"  Bandwidth: {bandwidth_mbps:.2f} MB/s")

            return {
                "backend": backend,
                "device": device_type,
                "data_size_mb": data_size_mb,
                "world_size": world_size,
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min_time,
                "max_time": max_time,
                "bandwidth_mbps": bandwidth_mbps,
                "raw_times": times,
            }
        else:
            print(f"  SKIPPED or FAILED")
            return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    """Run all benchmark combinations and save results."""

    # Configuration
    DATA_SIZES_MB = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    WORLD_SIZES = [2, 4]
    WARMUP_STEPS = 5
    MEASUREMENT_STEPS = 10

    # Check GPU availability
    has_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if has_cuda else 0

    print(f"\n{'=' * 70}")
    print(f"DDP All-Reduce Comprehensive Benchmark")
    print(f"{'=' * 70}")
    print(f"CUDA Available: {has_cuda}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Data sizes: {DATA_SIZES_MB} MB")
    print(f"World sizes: {WORLD_SIZES}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measurement steps: {MEASUREMENT_STEPS}")
    print(f"{'=' * 70}\n")

    # Configurations to test
    configs = []

    # Gloo + CPU (always available)
    for data_size in DATA_SIZES_MB:
        for world_size in WORLD_SIZES:
            configs.append(("gloo", "cpu", data_size, world_size))

    # NCCL + GPU (only if GPUs available)
    if has_cuda:
        for data_size in DATA_SIZES_MB:
            for world_size in WORLD_SIZES:
                # Only test if we have enough GPUs
                if world_size <= num_gpus:
                    configs.append(("nccl", "cuda", data_size, world_size))
                else:
                    print(f"Skipping NCCL with {world_size} processes (only {num_gpus} GPUs available)")
    else:
        print("Skipping all NCCL + GPU tests (no CUDA available)\n")

    # Run all configurations
    results = []
    total_configs = len(configs)

    for idx, (backend, device, data_size, world_size) in enumerate(configs, 1):
        print(f"\n[{idx}/{total_configs}] Testing configuration...")
        result = run_benchmark_config(
            backend=backend,
            device_type=device,
            data_size_mb=data_size,
            world_size=world_size,
            warmup_steps=WARMUP_STEPS,
            measurement_steps=MEASUREMENT_STEPS,
            config_idx=idx,  # Pass unique port for each config
        )

        if result is not None:
            results.append(result)

    # Save results to CSV
    output_dir = Path("benchmark_results_allreduce")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "allreduce_benchmark_results.csv"
    json_path = output_dir / "allreduce_benchmark_results.json"

    # Save as CSV
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "device",
                    "data_size_mb",
                    "world_size",
                    "mean_time",
                    "std_time",
                    "min_time",
                    "max_time",
                    "bandwidth_mbps",
                ],
            )
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.items() if k != "raw_times"}
                writer.writerow(row)

        # Save as JSON (with raw times)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"Benchmark Complete!")
        print(f"{'=' * 70}")
        print(f"Results saved to:")
        print(f"  CSV:  {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"Total configurations tested: {len(results)}")
        print(f"{'=' * 70}\n")
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
