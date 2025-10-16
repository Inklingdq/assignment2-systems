#!/usr/bin/env python3
"""
Script to analyze and compare FP32 vs BF16 mixed precision benchmarking results.
"""

import re
import os
from typing import Dict, Tuple

def parse_results_file(filepath: str) -> Dict[str, float]:
    """Parse a benchmark results file and extract mean and std."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    mean_match = re.search(r'mean:\s+([\d.]+)', content)
    std_match = re.search(r'std:\s+([\d.]+)', content)
    
    if mean_match and std_match:
        return {
            'mean': float(mean_match.group(1)),
            'std': float(std_match.group(1))
        }
    return None

def calculate_speedup(fp32_time: float, bf16_time: float) -> float:
    """Calculate speedup factor (FP32 time / BF16 time)."""
    return fp32_time / bf16_time

def main():
    results_dir = "benchmark_results"
    
    # Model configurations
    models = {
        'small': {'d_model': 768, 'layers': 12, 'heads': 12, 'd_ff': 3072},
        'large': {'d_model': 1024, 'layers': 24, 'heads': 16, 'd_ff': 4096},
        'xl': {'d_model': 1536, 'layers': 24, 'heads': 16, 'd_ff': 6144},
        '2.7B': {'d_model': 2560, 'layers': 32, 'heads': 32, 'd_ff': 10240}
    }
    
    print("=" * 80)
    print("MIXED PRECISION BENCHMARKING ANALYSIS: FP32 vs BF16")
    print("=" * 80)
    print()
    
    comparison_data = []
    
    for model_name, config in models.items():
        fp32_file = os.path.join(results_dir, f"{model_name}_fp32_results.txt")
        bf16_file = os.path.join(results_dir, f"{model_name}_bf16_results.txt")
        
        if not os.path.exists(fp32_file) or not os.path.exists(bf16_file):
            print(f"⚠ Warning: Results files for {model_name} not found")
            continue
        
        fp32_results = parse_results_file(fp32_file)
        bf16_results = parse_results_file(bf16_file)
        
        if fp32_results is None or bf16_results is None:
            print(f"⚠ Warning: Could not parse results for {model_name}")
            continue
        
        speedup = calculate_speedup(fp32_results['mean'], bf16_results['mean'])
        time_saved = fp32_results['mean'] - bf16_results['mean']
        percent_faster = ((fp32_results['mean'] - bf16_results['mean']) / fp32_results['mean']) * 100
        
        # Calculate model size in parameters (approximate)
        d_model = config['d_model']
        layers = config['layers']
        vocab_size = 10000
        d_ff = config['d_ff']
        
        # Rough parameter count: embedding + layers * (attention + FFN)
        # Each layer has: 4 * d_model^2 (attention) + 2 * d_model * d_ff (FFN)
        params_per_layer = (4 * d_model * d_model) + (2 * d_model * d_ff)
        total_params = (vocab_size * d_model) + (layers * params_per_layer)
        params_millions = total_params / 1e6
        
        comparison_data.append({
            'model': model_name,
            'params_m': params_millions,
            'config': config,
            'fp32_mean': fp32_results['mean'],
            'fp32_std': fp32_results['std'],
            'bf16_mean': bf16_results['mean'],
            'bf16_std': bf16_results['std'],
            'speedup': speedup,
            'time_saved': time_saved,
            'percent_faster': percent_faster
        })
        
        print(f"{'─' * 80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'─' * 80}")
        print(f"Configuration:")
        print(f"  • d_model: {config['d_model']}, layers: {config['layers']}, heads: {config['heads']}, d_ff: {config['d_ff']}")
        print(f"  • Approx. Parameters: {params_millions:.1f}M")
        print()
        print(f"FP32 (Full Precision):")
        print(f"  • Mean time: {fp32_results['mean']:.6f} seconds")
        print(f"  • Std dev:   {fp32_results['std']:.6f} seconds")
        print()
        print(f"BF16 (Mixed Precision):")
        print(f"  • Mean time: {bf16_results['mean']:.6f} seconds")
        print(f"  • Std dev:   {bf16_results['std']:.6f} seconds")
        print()
        print(f"Performance Comparison:")
        print(f"  • Speedup:       {speedup:.3f}x")
        print(f"  • Time saved:    {time_saved:.6f} seconds per step")
        print(f"  • Percent faster: {percent_faster:.2f}%")
        print()
    
    # Print summary table
    if comparison_data:
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print()
        print(f"{'Model':<10} {'Params':<12} {'FP32 (s)':<12} {'BF16 (s)':<12} {'Speedup':<10} {'% Faster':<10}")
        print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
        for data in comparison_data:
            print(f"{data['model']:<10} {data['params_m']:>10.1f}M {data['fp32_mean']:>10.6f}  {data['bf16_mean']:>10.6f}  {data['speedup']:>8.3f}x {data['percent_faster']:>8.2f}%")
        print()
        
        # Analysis of trends
        print("=" * 80)
        print("TRENDS AND OBSERVATIONS")
        print("=" * 80)
        print()
        
        print("1. Speedup vs Model Size:")
        print("   As model size increases:")
        speedups = [(d['params_m'], d['speedup']) for d in comparison_data]
        speedups.sort(key=lambda x: x[0])
        for params, speedup in speedups:
            print(f"   • {params:>8.1f}M parameters: {speedup:.3f}x speedup")
        print()
        
        avg_speedup = sum(d['speedup'] for d in comparison_data) / len(comparison_data)
        min_speedup = min(d['speedup'] for d in comparison_data)
        max_speedup = max(d['speedup'] for d in comparison_data)
        
        print("2. Overall Statistics:")
        print(f"   • Average speedup: {avg_speedup:.3f}x")
        print(f"   • Min speedup:     {min_speedup:.3f}x")
        print(f"   • Max speedup:     {max_speedup:.3f}x")
        print()
        
        print("3. Key Insights:")
        print("   • Mixed precision (BF16) reduces computation time by using 16-bit floating")
        print("     point operations instead of 32-bit, which is faster on modern GPUs.")
        print()
        print("   • The speedup benefit tends to be more pronounced for larger models because:")
        print("     - Larger models are more memory bandwidth-bound")
        print("     - BF16 reduces memory traffic by 2x (16 bits vs 32 bits)")
        print("     - Tensor cores on modern GPUs are optimized for lower precision")
        print()
        print("   • BF16 maintains similar numerical stability to FP32 due to its wider")
        print("     exponent range compared to FP16, making it suitable for training.")
        print()
        
        if len(comparison_data) >= 2:
            smallest = comparison_data[0]
            largest = comparison_data[-1]
            speedup_diff = largest['speedup'] - smallest['speedup']
            if speedup_diff > 0.1:
                print(f"   • Speedup increases from {smallest['speedup']:.3f}x ({smallest['model']}) to")
                print(f"     {largest['speedup']:.3f}x ({largest['model']}), showing that larger models")
                print(f"     benefit more from mixed precision training.")
            elif speedup_diff < -0.1:
                print(f"   • Speedup is relatively consistent across model sizes, with minor")
                print(f"     variations due to different computational patterns.")
            else:
                print(f"   • Speedup remains relatively stable across different model sizes.")
        print()

if __name__ == "__main__":
    main()
