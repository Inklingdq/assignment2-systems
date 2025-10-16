import torch
import torch.distributed as dist
import os
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from einops import rearrange
from cs336_basics.nn_utils import cross_entropy
import numpy as np
import torch.multiprocessing as mp
import copy
import time
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
import torch.cuda.nvtx as nvtx


class DDPIndividualParameters(torch.nn.Module):
    """
    DDP implementation that communicates each parameter's gradient individually.

    This class wraps an arbitrary PyTorch nn.Module and handles:
    1. Broadcasting weights from rank 0 before training
    2. Asynchronous gradient synchronization during backward pass
    3. Overlapping communication with backward computation
    """

    def __init__(self, module: torch.nn.Module, world_size=None) -> None:
        """
        Construct a DDP container for gradient synchronization.

        Args:
            module: PyTorch nn.Module to be parallelized
            world_size: Number of processes (optional, will auto-detect if None)
        """
        super().__init__()
        self.module = module

        # Get world size (number of processes)
        if world_size is not None:
            self.world_size = world_size
        elif dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        # Track async communication handles
        self._grad_async_handles = []

        # Broadcast parameters from rank 0 to all other ranks
        # This ensures all ranks start with the same initial weights
        if self.world_size > 1:
            self._broadcast_parameters()

        # Register hooks for all-reduce (only for parameters that require gradients)
        if self.world_size > 1:
            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_hook(param))

        # For benchmarking/timing purposes
        self._comm_time = 0.0

    def _broadcast_parameters(self) -> None:
        """
        Broadcast all parameters from rank 0 to all other ranks.
        This ensures all ranks have the same initial model state.
        """
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _make_hook(self, param):
        """
        Create a backward hook for a parameter that launches async all-reduce.

        Args:
            param: Parameter for which to create the hook

        Returns:
            Hook function that will be called when param's gradient is ready
        """

        @torch.utils.hooks.unserializable_hook
        def hook(param):
            """
            Backward hook that launches asynchronous gradient all-reduce.

            This hook is called automatically during backward() when this
            parameter's gradient is ready. It launches an async all-reduce
            operation and stores the handle for later synchronization.

            Args:
                grad: The gradient tensor

            Returns:
                The gradient tensor (unchanged)
            """
            if param.grad is None:
                return param.grad

            # Ensure gradient is contiguous for efficient communication
            if not param.grad.is_contiguous():
                grad_contig = param.grad.contiguous()
                param.grad.copy_(grad_contig)

            # For timing (optional, can be removed for production)
            # if hasattr(self, "_timing_enabled") and self._timing_enabled:
            #     torch.cuda.synchronize()
            #     start_time = time.time()

            # Launch asynchronous all-reduce (SUM operation)
            # async_op=True means this returns immediately without blocking
            # We need to operate on the grad tensor in-place
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)

            # Store the handle so we can wait for it later
            self._grad_async_handles.append(handle)

            # For timing (optional)
            # if hasattr(self, "_timing_enabled") and self._timing_enabled:
            #     torch.cuda.synchronize()
            #     elapsed = time.time() - start_time
            #     self._comm_time += elapsed

            return None

        return hook

    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous gradient communications to complete.

        This method should be called after backward() and before optimizer.step()
        to ensure all gradients have been synchronized across ranks.

        After all communications complete, gradients are averaged by dividing
        by the world size.
        """
        # Wait for all async all-reduce operations to complete
        for i in range(len(self._grad_async_handles) - 1, -1, -1):
            self._grad_async_handles[i].wait()

        # Average the gradients by dividing by world size
        # (all-reduce uses SUM, so we need to divide to get the average)
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)

        # Clear the handle list for the next iteration
        self._grad_async_handles.clear()

    def reset_comm_time(self):
        """Reset the accumulated communication time (for benchmarking)"""
        self._comm_time = 0.0
        self._timing_enabled = True

    def get_comm_time(self):
        """Get the accumulated communication time (for benchmarking)"""
        return self._comm_time

    def forward(self, *args, **kwargs):
        """
        Forward pass through the wrapped module.

        Args:
            *inputs: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module

        Returns:
            Output from the wrapped module's forward pass
        """
        return self.module(*args, **kwargs)


class DDPBucketParameters(torch.nn.Module):
    """
    DDP implementation that groups gradients into buckets for communication.

    Instead of communicating each gradient individually, this implementation
    groups gradients into buckets of a specified size and communicates each
    bucket once all its gradients are ready.
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float, world_size=None) -> None:
        """
        Construct a DDP container with gradient bucketing.

        Args:
            module: PyTorch nn.Module to be parallelized
            bucket_size_mb: Target bucket size in megabytes
            world_size: Number of processes (optional, will auto-detect if None)
        """
        super().__init__()
        self.module = module

        # Get world size
        if world_size is not None:
            self.world_size = world_size
        elif dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self._bucket_size_bytes = bucket_size_mb * 1024 * 1024

        # Track buckets: list of lists of gradients
        self.buckets = []
        self.bucket_params = []  # Parameters corresponding to each bucket
        self.current_bucket_size = 0

        # Track async communication handles
        self._grad_async_handles = []

        # Broadcast parameters from rank 0 to all other ranks
        if self.world_size > 1:
            self._broadcast_parameters()

        # Register hooks for bucketed gradient synchronization
        if self.world_size > 1:
            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_hook(param))

        self._comm_time = 0.0

    def _broadcast_parameters(self) -> None:
        """Broadcast all parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _grad_size_bytes(self, grad):
        """Calculate gradient size in bytes."""
        return grad.numel() * grad.element_size()

    def _make_hook(self, param):
        """
        Create a backward hook that adds gradients to buckets.

        Args:
            param: Parameter for which to create the hook

        Returns:
            Hook function that will be called when param's gradient is ready
        """

        @torch.utils.hooks.unserializable_hook
        def hook(param):
            if param.grad is None:
                return None

            # Ensure gradient is contiguous
            if not param.grad.is_contiguous():
                grad_contig = param.grad.contiguous()
                param.grad.copy_(grad_contig)

            grad_size = self._grad_size_bytes(param.grad)

            # If current bucket exists and adding this gradient would exceed bucket size,
            # finalize the current bucket
            if self.buckets and self.current_bucket_size + grad_size > self._bucket_size_bytes:
                # Finalize current bucket
                self._finalize_bucket()
                # Start a new bucket after finalization
                self.buckets.append([])
                self.bucket_params.append([])
                self.current_bucket_size = 0

            # Add gradient to current bucket (or start new bucket if empty)
            if not self.buckets:
                self.buckets.append([])
                self.bucket_params.append([])
                self.current_bucket_size = 0

            self.buckets[-1].append(param.grad)
            self.bucket_params[-1].append(param)
            self.current_bucket_size += grad_size

            return None

        return hook

    def _finalize_bucket(self):
        """Finalize the current bucket by launching all-reduce."""
        if not self.buckets or not self.buckets[-1]:
            return

        # Make copies of the bucket contents before finalizing
        # (we need these for unflattening later)
        grads_copy = list(self.buckets[-1])
        params_copy = list(self.bucket_params[-1])

        # Flatten all gradients in the current bucket
        flat_grad = torch._utils._flatten_dense_tensors(grads_copy)

        # Launch async all-reduce
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self._grad_async_handles.append((handle, flat_grad, grads_copy, params_copy))

        # Reset for next bucket
        self.current_bucket_size = 0

    def finish_gradient_synchronization(self):
        """
        Wait for all bucket communications to complete and copy results back.
        """
        # Finalize any remaining bucket
        self._finalize_bucket()

        # Measure time spent waiting for communications
        torch.cuda.synchronize()
        wait_start = time.time()

        # Wait for all async operations and unflatten results
        for handle, flat_grad, grads, params in self._grad_async_handles:
            handle.wait()

            # Average the gradients
            flat_grad.div_(self.world_size)

            # Unflatten back to parameter gradients
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)

            # Copy unflattened gradients back to parameters
            for param, unflat_grad in zip(params, unflat_grads):
                param.grad.data.copy_(unflat_grad)

        torch.cuda.synchronize()
        self._comm_time = time.time() - wait_start

        # Clear for next iteration
        self._grad_async_handles.clear()
        self.buckets.clear()
        self.bucket_params.clear()
        self.current_bucket_size = 0

    def reset_comm_time(self):
        """Reset the accumulated communication time (for benchmarking)"""
        self._comm_time = 0.0

    def get_comm_time(self):
        """Get the accumulated communication time (for benchmarking)"""
        return self._comm_time

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)


class DDPFlattenedGradients(torch.nn.Module):
    """DDP implementation that flattens all gradients into a single tensor for communication"""

    def __init__(self, module: torch.nn.Module, world_size) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        self._comm_time = 0.0

        # Build a list of parameters that require gradients
        self.params = [p for p in module.parameters() if p.requires_grad]

        # Register a single hook on the first parameter's gradient
        # During backward, gradients are computed in reverse order, so the first parameter
        # in the list (typically embedding layer) receives its gradient LAST
        # This ensures all gradients are computed before we do the all-reduce
        if len(self.params) > 0:
            self.params[0].register_hook(self._all_reduce_hook)

    def _all_reduce_hook(self, grad):
        """Hook that performs all-reduce on flattened gradients"""
        if self.world_size == 1:
            return grad

        # Collect all gradients and their corresponding parameters
        params_with_grad = [p for p in self.params if p.grad is not None]
        grads = [p.grad for p in params_with_grad]

        if len(grads) == 0:
            return grad

        # Flatten all gradients into a single buffer using PyTorch's utility
        flat_grad = torch._utils._flatten_dense_tensors(grads)

        # All-reduce the flattened buffer
        torch.cuda.synchronize()
        start_time = time.time()
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        self._comm_time += elapsed

        # Average the gradients
        flat_grad.div_(self.world_size)

        # Unflatten back to parameter gradients using PyTorch's utility
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)

        # Copy unflattened gradients back to parameters (now properly aligned)
        for param, unflat_grad in zip(params_with_grad, unflat_grads):
            param.grad.data.copy_(unflat_grad)

        return grad

    def reset_comm_time(self):
        """Reset the accumulated communication time"""
        self._comm_time = 0.0

    def get_comm_time(self):
        """Get the accumulated communication time"""
        return self._comm_time

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25902"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(
    rank,
    world_size,
    backend,
    device,
    model,
    x,
    y,
    warmup_steps=5,
    measurement_steps=10,
    benchmark_mode=False,
    use_flattened=False,
    use_bucket=False,
    bucket_size_mb=1,
    enable_profiling=False,
    profile_output_dir="/tmp/nsight_profiles",
):
    setup(rank, world_size, backend)
    torch.manual_seed(42)  # Use same seed for all ranks for deterministic comparison
    if device == "cuda":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)

    model.to(device)
    if use_flattened:
        model = DDPFlattenedGradients(model, world_size)
        if benchmark_mode and rank == 0:
            print("Using DDPFlattenedGradients (single all-reduce call)")
    else:
        if use_bucket:
            model = DDPBucketParameters(model, bucket_size_mb, world_size)
            if benchmark_mode and rank == 0:
                print(f"Using DDPBucketParameters (bucket size: {bucket_size_mb}MB)")
        else:
            model = DDPIndividualParameters(model, world_size)
            if benchmark_mode and rank == 0:
                print("Using DDPIndividualParameters (one all-reduce per parameter)")

    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    min_batch = len(x) // world_size
    x = x[rank * min_batch : (rank + 1) * min_batch].to(device)
    y = y[rank * min_batch : (rank + 1) * min_batch].to(device)

    # Warmup phase
    for _ in range(warmup_steps):
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        if not use_flattened and hasattr(model, "finish_gradient_synchronization"):
            model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

    # Measurement phase with NVTX annotations for Nsight Systems
    step_times = []
    comm_times = []

    # Start CUDA profiler (for nsys profiling)
    if enable_profiling:
        torch.cuda.profiler.start()

    for step in range(measurement_steps):
        model.reset_comm_time()
        torch.cuda.synchronize()
        step_start = time.time()

        nvtx.range_push(f"step_{step}")

        nvtx.range_push("forward")
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("gradient_sync")
        if not use_flattened and hasattr(model, "finish_gradient_synchronization"):
            model.finish_gradient_synchronization()
        nvtx.range_pop()

        nvtx.range_push("optimizer_step")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_pop()  # step

        torch.cuda.synchronize()
        step_time = time.time() - step_start
        comm_time = model.get_comm_time()
        step_times.append(step_time)
        comm_times.append(comm_time)

        if benchmark_mode and rank == 0:
            print(
                f"[Step {step}] Total: {step_time:.4f}s, Comm: {comm_time:.4f}s, Comm%: {(comm_time / step_time * 100):.2f}%"
            )

    # Stop CUDA profiler
    if enable_profiling:
        torch.cuda.profiler.stop()
        if rank == 0:
            print(f"\nNsight profiling complete for rank {rank}")
            print(f"Use nsys to capture: nsys profile [options] python your_script.py")

    # Print results from rank 0
    if rank == 0:
        if benchmark_mode:
            avg_step_time = np.mean(step_times)
            avg_comm_time = np.mean(comm_times)
            comm_percentage = (avg_comm_time / avg_step_time) * 100
            print(f"\n{'=' * 60}")
            print(f"DDP Benchmark Results (Rank 0)")
            print(f"{'=' * 60}")
            print(f"Average step time:          {avg_step_time:.4f}s")
            print(f"Average communication time: {avg_comm_time:.4f}s")
            print(f"Communication overhead:     {comm_percentage:.2f}%")
            print(f"Computation time:           {avg_step_time - avg_comm_time:.4f}s")
            print(f"{'=' * 60}\n")

        torch.save(model.module.state_dict(), "/tmp/ddp_trained_model.pt")

    cleanup()


def compare_ddp_and_non_ddp(model, args):
    print("Testing DDP training vs. non-DDP training...")
    non_parallel_model = copy.deepcopy(model)

    random_data = np.random.randint(low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32)
    x, y = get_batch(
        dataset=random_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    mp.spawn(
        fn=train,
        args=(
            args.world_size,
            args.backend,
            args.device,
            model,
            x,
            y,
            args.warmup_steps,
            args.measurement_steps,
            False,
        ),
        nprocs=args.world_size,
        join=True,
    )

    # Load trained weights back to parent's model
    model.load_state_dict(torch.load("/tmp/ddp_trained_model.pt"))

    # For non-parallel (single process) training, call train directly:
    train(
        0,  # rank
        1,  # world_size
        args.backend,
        args.device,
        non_parallel_model,
        x,
        y,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        benchmark_mode=False,
    )

    # Validate that the models are the same
    for non_parallel_param, parallel_param in zip(non_parallel_model.parameters(), model.parameters()):
        np.testing.assert_allclose(
            non_parallel_param.detach().cpu().numpy(),
            parallel_param.detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-5,
        )

    print("Success!")


def train_bucketed(
    rank,
    world_size,
    backend,
    device,
    model,
    x,
    y,
    bucket_size_mb,
    warmup_steps=5,
    measurement_steps=10,
):
    """Training function specifically for bucketed DDP benchmarking"""
    setup(rank, world_size, backend)
    torch.manual_seed(42)
    if device == "cuda":
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)

    model.to(device)
    model = DDPBucketParameters(model, bucket_size_mb, world_size)

    optimizer = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    min_batch = len(x) // world_size
    x = x[rank * min_batch : (rank + 1) * min_batch].to(device)
    y = y[rank * min_batch : (rank + 1) * min_batch].to(device)

    # Warmup phase
    for _ in range(warmup_steps):
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

    # Measurement phase
    step_times = []
    torch.cuda.synchronize() if device.type == "cuda" else None

    for step in range(measurement_steps):
        step_start = time.time()

        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize() if device.type == "cuda" else None
        step_time = time.time() - step_start
        step_times.append(step_time)

    # Return results from all ranks
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)

    cleanup()

    return avg_step_time, std_step_time, rank


def benchmark_bucketed_ddp(model, args, bucket_sizes_mb):
    """Benchmark bucketed DDP with different bucket sizes"""
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
    print(f"\nBucket sizes to test: {bucket_sizes_mb} MB")
    print(f"{'=' * 80}\n")

    # Prepare data
    random_data = np.random.randint(low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32)
    x, y = get_batch(
        dataset=random_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    results = []

    for bucket_size_mb in bucket_sizes_mb:
        print(f"\n{'=' * 80}")
        print(f"Testing bucket size: {bucket_size_mb} MB")
        print(f"{'=' * 80}")

        # Create fresh model for each test
        test_model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        )

        # Run benchmark
        mp.spawn(
            fn=train_bucketed,
            args=(
                args.world_size,
                args.backend,
                args.device,
                test_model,
                x,
                y,
                bucket_size_mb,
                args.warmup_steps,
                args.measurement_steps,
            ),
            nprocs=args.world_size,
            join=True,
        )

        # Note: In multiprocessing, we can't easily return values from spawn
        # So we'll need to use a different approach or save results to files
        print(f"Completed benchmark for bucket size {bucket_size_mb} MB\n")

    print(f"\n{'=' * 80}")
    print(f"BENCHMARKING COMPLETE")
    print(f"{'=' * 80}\n")


def benchmark_ddp(model, args):
    """Benchmark DDP training with XL model size"""
    print(f"\n{'=' * 60}")
    print(f"DDP Benchmarking - XL Model")
    print(f"{'=' * 60}")
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
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  measurement_steps: {args.measurement_steps}")
    print(f"{'=' * 60}\n")

    random_data = np.random.randint(low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32)
    x, y = get_batch(
        dataset=random_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    mp.spawn(
        fn=train,
        args=(
            args.world_size,
            args.backend,
            args.device,
            model,
            x,
            y,
            args.warmup_steps,
            args.measurement_steps,
            True,
            args.use_flattened,
            args.use_bucket,
            args.bucket_size_mb,
            args.enable_profiling,
            args.profile_output_dir,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training and Benchmarking")
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["compare", "benchmark"],
        help="Mode: compare (validation) or benchmark",
    )
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--backend", type=str, default="nccl", help="DDP backend")
    parser.add_argument("--device", type=str, default="cuda", help="Device type")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--measurement_steps", type=int, default=10, help="Number of measurement steps")
    parser.add_argument(
        "--use-flattened",
        action="store_true",
        help="Use flattened gradient communication (single all-reduce) instead of per-parameter",
    )

    # XL model configuration as default
    parser.add_argument("--d_model", type=int, default=1600, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=25, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=6400, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000, help="RoPE theta")
    parser.add_argument("--bucket_size_mb", type=int, default=1, help="Bucket size in MB")
    parser.add_argument("--use_bucket", action="store_true", help="Benchmark bucketed DDP")
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable Nsight profiling to visualize compute-communication overlap",
    )
    parser.add_argument(
        "--profile_output_dir", type=str, default="/tmp/nsight_profiles", help="Directory to save profile traces"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    if args.mode == "compare":
        compare_ddp_and_non_ddp(model, args)
    elif args.mode == "benchmark":
        benchmark_ddp(model, args)
