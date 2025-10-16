import argparse
import math
import timeit
from contextlib import nullcontext

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr
from einops import rearrange


def benchmarking(args):
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    random_data = np.random.randint(
        low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32
    )
    x, y = get_batch(
        dataset=random_data,
        batch_size=4,
        context_length=args.context_length,
        device=args.device,
    )
    model.to(args.device)
    if args.jit_compile_model:
        model = torch.compile(model)
    optimizer = AdamW(model.parameters())

    # Reset peak memory stats before benchmarking
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()

    # Setup mixed precision context
    if args.use_mixed_precision:
        autocast_context = torch.autocast(device_type=args.device, dtype=torch.bfloat16)
        print(f"Running with mixed precision (BF16)")
    else:
        autocast_context = nullcontext()
        print(f"Running with full precision (FP32)")

    # Start recording memory history before any training steps (including warmup)
    torch.cuda.memory._record_memory_history(enabled=True, context="all", stacks="all")

    time = []
    for step in range(args.warmup_steps + args.measurement_steps):
        if step == 0:
            nvtx.range_push("Warm-up Phase")
        if step == args.warmup_steps:
            nvtx.range_pop()
            nvtx.range_push("Measurement Phase")

        nvtx.range_push(f"Step {step}")

        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        nvtx.range_push("Forward Pass")
        with autocast_context:
            logits = model(x)
            logits_flatten = rearrange(logits, "b c v -> (b c) v")
            y_flatten = rearrange(y, "b c -> (b c)")
            loss = cross_entropy(logits_flatten, y_flatten)
        nvtx.range_pop()

        if args.measure_forward_only:
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            time.append(end_time - start_time)
            continue

        nvtx.range_push("Backward Pass")
        loss.backward()
        clip_gradient(model.parameters(), 1)
        nvtx.range_pop()

        if not (args.measure_forward_only):
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            time.append(end_time - start_time)

        # Update lr
        current_lr = get_cosine_lr(step, 1e-4, 1e-3, 2, 10)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        nvtx.range_push("Optimizer Step")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_pop()

    nvtx.range_pop()
    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

    # Get peak memory statistics
    peak_allocated = torch.cuda.max_memory_allocated(args.device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(args.device) / (1024**3)

    precision_mode = "BF16" if args.use_mixed_precision else "FP32"
    mode = "Forward-only" if args.measure_forward_only else "Full Training"
    print(
        f"Model: d_model={args.d_model}, d_ff={args.d_ff}, num_layers={args.num_layers}, num_heads={args.num_heads}"
    )
    print(f"Context Length: {args.context_length}")
    print(f"Mode: {mode}")
    print(f"Precision: {precision_mode}")
    print(f"Peak Memory Allocated: {peak_allocated:.2f} GB")
    print(f"Peak Memory Reserved: {peak_reserved:.2f} GB")
    print("time:", time[args.warmup_steps :])
    print("mean:", np.mean(time[args.warmup_steps :]))
    print("std:", np.std(time[args.warmup_steps :]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking a language model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Steps to warmup before benchmarking",
    )
    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=10,
        help="Steps to benchmarking after warmup",
    )
    parser.add_argument(
        "--measure-forward-only",
        type=bool,
        default=False,
        help="Steps to benchmarking after warmup",
    )
    parser.add_argument(
        "--jit-compile-model",
        type=bool,
        default=False,
        help="Whether to do jit compile model.",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Whether to use mixed precision (BF16) training.",
    )
    # Add model hyperparameters
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument(
        "--context-length", type=int, default=256, help="Context length"
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    args = parser.parse_args()
    benchmarking(args)
