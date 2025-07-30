import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient, clip_gradient
from cs336_basics.optimizer import AdamW, get_cosine_lr
import numpy as np
from einops import rearrange
import torch
import timeit


def benchmarking(args):
    model = BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
        )
    random_data = np.random.randint(low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32)
    x, y = get_batch(dataset = random_data, batch_size = 4, context_length = args.context_length, device = args.device)
    model.to(args.device)
    optimizer = AdamW(model.parameters())
    time = []
    for step in range(args.warmup_steps + args.measurement_steps):
        start_time = timeit.default_timer()
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        end_time = timeit.default_timer()
        loss.backward()
        clip_gradient(model.parameters(), 1)
        if not(args.measure_forward_only):
            end_time = timeit.default_timer()
        time.append(end_time - start_time)

        # Update lr
        current_lr = get_cosine_lr(step, 1e-4, 1e-3, 2, 10)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        torch.mps.synchronize()

        optimizer.step()
        optimizer.zero_grad()
    
    print(f"{args.d_model} {args.d_ff}, {args.num_layers}, {args.num_heads}")
    print(time)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking a language model.")
    parser.add_argument("--batch-size", type=int, default = 32, help = "Batch Size")
    parser.add_argument('--device', type = str, default = "mps", help = "Device to run inference on")
    parser.add_argument('--warmup-steps', type = int, default = 5, help = "Steps to warmup before benchmarking")
    parser.add_argument('--measurement-steps', type = int, default = 10, help = "Steps to benchmarking after warmup")
    parser.add_argument('--measure-forward-only', type = bool, default = False, help = "Steps to benchmarking after warmup")

    # Add model hyperparameters
    parser.add_argument('--vocab-size', type=int, default = 10000)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=16)
    parser.add_argument('--d-ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--context-length', type=int, default=256, help='Context length')
    parser.add_argument('--rope-theta', type=float, default=10000.0)


    args = parser.parse_args()
    benchmarking(args)