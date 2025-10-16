# How to View DDP Bucket Overlap in Nsight Systems

## What You're Looking For

The goal is to see **gradient computation overlapping with gradient communication** when using bucketed DDP.

## Steps in Nsight Systems Viewer

### 1. Open the Timeline View
- The main window shows a horizontal timeline
- Time flows from left to right

### 2. Find Key Rows in the Timeline

Look for these rows (from top to bottom):

#### NVTX Ranges (colored bars):
- **step_0, step_1, step_2**: Each training iteration
- **forward**: Forward pass computation
- **backward**: Backward gradient computation  
- **gradient_sync**: Gradient synchronization (bucket finalization)
- **optimizer_step**: Optimizer parameter updates

#### CUDA API calls:
- Shows CUDA operations being launched

#### CUDA Kernels (GPU execution):
- Shows actual GPU kernel execution
- During backward: kernels execute in reverse order (Layer 7 → Layer 0)
- Look for patterns like:
  - `LayerNorm`, `Softmax`, `MatMul` kernels
  - Running from last layer to first layer

#### NCCL (Communication):
- **ncclAllReduce** operations
- These are the gradient communications happening in buckets

### 3. Zoom to See One Complete Training Step

**Method 1: Use keyboard**
- Press `F` to fit the entire trace
- Use mouse wheel to zoom in/out
- Click and drag to pan left/right

**Method 2: Select a region**
- Click and drag to select a time range
- Right-click → "Zoom to Selection"

**Each step should take ~60-80ms** (based on your benchmark output)

### 4. What Overlap Looks Like

When viewing a single backward pass:

```
Time →
[=== Backward CUDA Kernels ===========================]
  Layer 7  Layer 6  Layer 5  Layer 4  Layer 3  Layer 2  Layer 1  Layer 0
            ↑                 ↑                 ↑
            [ncclAllReduce]   [ncclAllReduce]   [ncclAllReduce]
            Bucket 1          Bucket 2          Bucket 3
```

**The KEY observation:**
- ncclAllReduce operations **start BEFORE** all backward kernels complete
- This means: **while Layer 0-2 are still computing**, Layer 7-5's gradients are already being communicated
- **This is the overlap!**

### 5. Compare with Non-Bucketed (if available)

Without bucketing, you would see:
```
Time →
[=== Backward CUDA Kernels ===]  [=== NCCL Communications ===]
                                  ↑
                                  All communication happens AFTER computation
                                  NO OVERLAP
```

## Expected Timing (from your benchmark)

From the terminal output:
- Average step time: 0.0831s (83.1ms)
- Communication time: 0.0000s (not measured in this mode)

The NVTX ranges should match these timings when you measure them in Nsight Systems.

## Troubleshooting

### If you don't see NVTX ranges:
- Check the left panel: ensure "NVTX" row is expanded
- The profiling may not have captured them - try re-running with `--enable_profiling` flag

### If you don't see NCCL operations:
- Check if "NCCL" row exists in the timeline
- NCCL communications might be very fast and hard to see - zoom in more

### If everything looks too small:
- Press `F` to fit entire trace
- Use mouse wheel to zoom in on a specific step
- Look at the time ruler at the top to confirm you're viewing ~80ms range

## Key Insight

The bucketing strategy allows DDP to:
1. Start communicating gradients as soon as a bucket is full (~1MB)
2. Continue computing gradients for other layers simultaneously
3. Achieve better GPU utilization by overlapping compute and communication

This is why bucketed DDP is more efficient than waiting for all gradients before starting communication!
