#!/usr/bin/env python3
"""
Benchmark bucketed DDP implementation with different bucket sizes.
Compares results to non-bucketed DDP implementations.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from einops import rearrange
from cs336_basics.nn_utils import cross_entropy
from ddp.ddp_train import DDPBucketParameters, DDPIndividualParameters, DDPFlattenedGradients


def setup(rank, world_size, backend):
    """Initialize distributed process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25903"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed process group"""
    dist.destroy_process_group()


def train_bucketed(
    rank,
    world_size,
    backend,
    device_type,
    model_config,
    x,
    y,
    bucket_size_mb,
    warmup_steps,
    measurement_steps,
    output_file,
):
    """Training function for bucketed DDP"""
    setup(rank, world_size, backend)
    torch.manual_seed(42)

    # Set device
    if device_type == "cuda":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Create model
    model = BasicsTransformerLM(**model_config)
    model.to(device)

    # Wrap with bucketed DDP
    model = DDPBucketParameters(model, bucket_size_mb, world_size)

    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    # Split data for this rank
    min_batch = len(x) // world_size
    x_local = x[rank * min_batch : (rank + 1) * min_batch].to(device)
    y_local = y[rank * min_batch : (rank + 1) * min_batch].to(device)

    # Warmup phase
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x_local)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y_local, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    # Measurement phase
    step_times = []
    if device_type == "cuda":
        torch.cuda.synchronize()

    for step in range(measurement_steps):
        step_start = time.time()

        optimizer.zero_grad()
        logits = model(x_local)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y_local, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)

    # Save results to file (only from rank 0)
    if rank == 0:
        result = {
            "bucket_size_mb": bucket_size_mb,
            "avg_time": float(np.mean(step_times)),
            "std_time": float(np.std(step_times)),
            "min_time": float(np.min(step_times)),
            "max_time": float(np.max(step_times)),
            "all_times": [float(t) for t in step_times],
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    cleanup()


def train_non_bucketed(
    rank,
    world_size,
    backend,
    device_type,
    model_config,
    x,
    y,
    use_flattened,
    warmup_steps,
    measurement_steps,
    output_file,
):
    """Training function for non-bucketed DDP (individual or flattened)"""
    setup(rank, world_size, backend)
    torch.manual_seed(42)

    # Set device
    if device_type == "cuda":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Create model
    model = BasicsTransformerLM(**model_config)
    model.to(device)

    # Wrap with DDP
    if use_flattened:
        model = DDPFlattenedGradients(model, world_size)
        method_name = "Flattened"
    else:
        model = DDPIndividualParameters(model, world_size)
        method_name = "Individual"

    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    # Split data for this rank
    min_batch = len(x) // world_size
    x_local = x[rank * min_batch : (rank + 1) * min_batch].to(device)
    y_local = y[rank * min_batch : (rank + 1) * min_batch].to(device)

    # Warmup phase
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x_local)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y_local, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        if not use_flattened:
            model.finish_gradient_synchronization()
        optimizer.step()

    # Measurement phase
    step_times = []
    if device_type == "cuda":
        torch.cuda.synchronize()

    for step in range(measurement_steps):
        step_start = time.time()

        optimizer.zero_grad()
        logits = model(x_local)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y_local, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        if not use_flattened:
            model.finish_gradient_synchronization()
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)

    # Save results to file (only from rank 0)
    if rank == 0:
        result = {
            "method": method_name,
            "avg_time": float(np.mean(step_times)),
            "std_time": float(np.std(step_times)),
            "min_time": float(np.min(step_times)),
            "max_time": float(np.max(step_times)),
            "all_times": [float(t) for t in step_times],
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Benchmark bucketed DDP implementation")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="DDP backend")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device type")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--measurement-steps", type=int, default=10, help="Number of measurement steps")

    # XL model configuration as default
    parser.add_argument("--d-model", type=int, default=1600, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=48, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=25, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=6400, help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000, help="RoPE theta")

    # Bucket sizes to test
    parser.add_argument(
        "--bucket-sizes",
        type=float,
        nargs="+",
        default=[1, 10, 100, 1000],
        help="Bucket sizes in MB to test",
    )

    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmark_results_bucketed", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Model configuration
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    print(f"\n{'=' * 80}")
    print(f"BUCKETED DDP BENCHMARKING")
    print(f"{'=' * 80}")
    print(f"Model Configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  context_length: {args.context_length}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"\nTraining Configuration:")
    print(f"  world_size: {args.world_size}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  backend: {args.backend}")
    print(f"  device: {args.device}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  measurement_steps: {args.measurement_steps}")
    print(f"\nBucket sizes to test: {args.bucket_sizes} MB")
    print(f"{'=' * 80}\n")

    # Prepare data
    torch.manual_seed(42)
    random_data = np.random.randint(low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32)
    x, y = get_batch(
        dataset=random_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    all_results = {"config": vars(args), "results": []}

    # Benchmark individual parameters (baseline)
    print(f"\n{'=' * 80}")
    print(f"Benchmarking: Individual Parameters (Baseline)")
    print(f"{'=' * 80}")

    output_file = output_dir / "temp_individual.json"
    mp.spawn(
        fn=train_non_bucketed,
        args=(
            args.world_size,
            args.backend,
            args.device,
            model_config,
            x,
            y,
            False,  # use_flattened
            args.warmup_steps,
            args.measurement_steps,
            str(output_file),
        ),
        nprocs=args.world_size,
        join=True,
    )

    with open(output_file, "r") as f:
        result = json.load(f)
    all_results["results"].append(result)
    print(f"Results: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
    output_file.unlink()  # Delete temp file

    # Benchmark flattened gradients (baseline)
    print(f"\n{'=' * 80}")
    print(f"Benchmarking: Flattened Gradients (Baseline)")
    print(f"{'=' * 80}")

    output_file = output_dir / "temp_flattened.json"
    mp.spawn(
        fn=train_non_bucketed,
        args=(
            args.world_size,
            args.backend,
            args.device,
            model_config,
            x,
            y,
            True,  # use_flattened
            args.warmup_steps,
            args.measurement_steps,
            str(output_file),
        ),
        nprocs=args.world_size,
        join=True,
    )

    with open(output_file, "r") as f:
        result = json.load(f)
    all_results["results"].append(result)
    print(f"Results: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
    output_file.unlink()  # Delete temp file

    # Benchmark bucketed DDP with different bucket sizes
    for bucket_size_mb in args.bucket_sizes:
        print(f"\n{'=' * 80}")
        print(f"Benchmarking: Bucketed DDP (bucket_size={bucket_size_mb} MB)")
        print(f"{'=' * 80}")

        output_file = output_dir / f"temp_bucketed_{bucket_size_mb}.json"
        mp.spawn(
            fn=train_bucketed,
            args=(
                args.world_size,
                args.backend,
                args.device,
                model_config,
                x,
                y,
                bucket_size_mb,
                args.warmup_steps,
                args.measurement_steps,
                str(output_file),
            ),
            nprocs=args.world_size,
            join=True,
        )

        with open(output_file, "r") as f:
            result = json.load(f)
        all_results["results"].append(result)
        print(f"Results: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")
        output_file.unlink()  # Delete temp file

    # Save final results
    results_file = output_dir / "bucketed_ddp_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"SUMMARY OF RESULTS")
    print(f"{'=' * 80}")
    for result in all_results["results"]:
        if "bucket_size_mb" in result:
            label = f"Bucketed ({result['bucket_size_mb']} MB)"
        else:
            label = result["method"]
        print(f"{label:30s}: {result['avg_time']:.4f}s ± {result['std_time']:.4f}s")

    print(f"\nResults saved to: {results_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
