import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def distributed(rank, world_size, backend, device_type, data_size, warmup_steps, measurement_steps):
    """Run distributed all-reduce operation.

    Args:
        rank: Process rank,
        world_size: Total number of processes
        backend: DDP backend (gloo, nccl, etc.)
        device_type: Device type (cpu or cuda)
        data_size: Size of data tensor
        is_warmup: If True, suppress output (warmup phase)

    Returns:
        Average time across all ranks (only for rank 0)
    """
    setup(rank, world_size, backend)
    # Set device with rank-specific GPU assignment
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    torch.cuda.set_device(device)
    dist.barrier(device_ids=[rank])

    for _ in range(warmup_steps):
        data = torch.rand(data_size * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.all_reduce(data, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(measurement_steps):
        data = torch.rand(data_size * 1024 * 1024 // 4, dtype=torch.float32, device=device)

        # Ensure CUDA is synchronized before timing if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        dist.all_reduce(data, async_op=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    print(f"Process {rank} finished in {(sum(times) / measurement_steps):.6f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed all-reduce benchmarking with warmup and measurement phases."
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", choices=["gloo", "nccl", "mpi"], help="Distributed backend to use."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device type to use for tensors."
    )
    parser.add_argument("--data-size", type=int, default=1, help="Size of the tensor for all-reduce (in KB).")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes to spawn.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup iterations.")
    parser.add_argument("--measurement-steps", type=int, default=5, help="Number of measurement iterations.")

    args = parser.parse_args()

    world_size = args.world_size

    print(f"\n{'=' * 60}")
    print(f"DDP All-Reduce Benchmark")
    print(f"{'=' * 60}")
    print(f"Configuration:")
    print(f"  Backend: {args.backend}")
    print(f"  Device: {args.device}")
    print(f"  Data size: {args.data_size} KB")
    print(f"  World size: {world_size} processes")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Measurement steps: {args.measurement_steps}")
    print(f"{'=' * 60}\n")

    # Warmup phase
    mp.spawn(
        fn=distributed,
        args=(
            world_size,
            args.backend,
            args.device,
            args.data_size,
            args.warmup_steps,
            args.measurement_steps,
        ),  # is_warmup=True
        nprocs=world_size,
        join=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Benchmark complete!")
    print(f"Note: Times shown above are averages across all {world_size} processes")
    print(f"{'=' * 60}\n")
