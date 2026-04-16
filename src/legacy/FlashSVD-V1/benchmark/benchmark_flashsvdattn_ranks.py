#!/usr/bin/env python3
"""
benchmark_flashsvdattn_ranks.py - Comprehensive benchmarking of FlashSVD attention across different head ranks

This script benchmarks:
1. Latency (forward pass time)
2. Memory usage (peak GPU memory)
3. Accuracy (relative error vs dense baseline)
4. Parameter count comparison

Across different head rank configurations for FlashSVD attention.
"""

import time
import torch
import torch.nn as nn
import math
import sys
import os

# Add current directory to path to import flashsvdattn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flashsvdattn import flash_svd_attention, FlashSVDBlock, BaselineBlock, transplant_weights
except ImportError:
    print("Error: Could not import flashsvdattn. Make sure flashsvdattn.py is in the current directory.")
    sys.exit(1)

class BenchmarkConfig:
    """Configuration for benchmarking"""
    def __init__(self):
        # Model dimensions
        self.d_model = 768
        self.n_heads = 12
        self.d_ff = 3072
        self.dh = self.d_model // self.n_heads  # 64
        
        # Sequence and batch dimensions
        self.batch_sizes = [1, 16, 64]
        self.seq_lengths = [128, 256, 512, 1024]
        
        # Rank configurations to test
        self.ranks = [64, 48, 32, 16]
        
        # Benchmark settings
        self.warmup_iters = 10
        self.timing_iters = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        
        # Filter ranks that are valid (rank <= dh)
        self.ranks = [r for r in self.ranks if r <= self.dh]

def create_attention_mask(batch_size, seq_len, mask_ratio=0.2, device="cuda"):
    """Create realistic attention mask with some padding"""
    mask = torch.ones(batch_size, 1, 1, seq_len, device=device, dtype=torch.bool)
    
    # Simulate variable sequence lengths
    for i in range(batch_size):
        if mask_ratio > 0:
            actual_len = max(1, int(seq_len * (1 - torch.rand(1).item() * mask_ratio)))
            mask[i, :, :, actual_len:] = False
    
    return mask

def benchmark_model(model, x, mask, warmup_iters=10, timing_iters=50):
    """Benchmark a model's forward pass"""
    model.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iters):
            if isinstance(model, FlashSVDBlock):
                _ = model(x, mask)
            else:  # BaselineBlock
                _ = model(x, mask)
        torch.cuda.synchronize()
        
        # Clear memory stats and time the iterations
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        
        for _ in range(timing_iters):
            if isinstance(model, FlashSVDBlock):
                output = model(x, mask)
            else:  # BaselineBlock
                output = model(x, mask)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / timing_iters
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return avg_time_ms, peak_memory_mb, output

def compute_accuracy(output_flash, output_dense):
    """Compute relative error between FlashSVD and dense outputs"""
    diff_norm = (output_flash.float() - output_dense.float()).norm()
    dense_norm = output_dense.float().norm()
    return (diff_norm / dense_norm).item()

def count_attention_parameters(d_model, n_heads, rank=None):
    """Count attention parameters for dense vs FlashSVD"""
    if rank is None:  # Dense attention
        # Q, K, V projections: 3 * (d_model * d_model + d_model)
        # Output projection: d_model * d_model + d_model
        qkv_params = 3 * (d_model * d_model + d_model)
        out_params = d_model * d_model + d_model
        return qkv_params + out_params
    else:  # FlashSVD attention
        dh = d_model // n_heads
        # P matrices: 3 * n_heads * d_model * rank
        # V matrices: 3 * n_heads * rank * dh
        # Biases: 3 * n_heads * dh
        p_params = 3 * n_heads * d_model * rank
        v_params = 3 * n_heads * rank * dh
        bias_params = 3 * n_heads * dh
        # Output projection (unchanged)
        out_params = d_model * d_model + d_model
        return p_params + v_params + bias_params + out_params

def run_rank_comparison(config):
    """Run comprehensive comparison across different ranks"""
    print("🔬 FlashSVD Attention Rank Benchmarking")
    print("=" * 80)
    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print("=" * 80)
    
    results = []
    
    # Test different batch sizes and sequence lengths
    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lengths:
            print(f"\n📊 Testing B={batch_size}, M={seq_len}")
            print("-" * 60)
            
            # Create input data
            x = torch.randn(batch_size, seq_len, config.d_model, 
                          device=config.device, dtype=config.dtype)
            mask = create_attention_mask(batch_size, seq_len, device=config.device)
            
            # Create dense baseline
            dense_model = BaselineBlock(config.d_model, config.n_heads, config.d_ff).to(config.device).to(config.dtype)
            
            # Benchmark dense model
            dense_time, dense_memory, dense_output = benchmark_model(
                dense_model, x, mask, config.warmup_iters, config.timing_iters)
            
            dense_params = count_attention_parameters(config.d_model, config.n_heads)
            
            print(f"{'Rank':<6}{'Time(ms)':<10}{'Memory(MB)':<12}{'Rel Error':<12}{'Params(K)':<12}{'Reduction%':<12}{'Speedup':<10}")
            print("-" * 80)
            
            # Dense baseline row
            print(f"{'Dense':<6}{dense_time:<10.2f}{dense_memory:<12.1f}{'0.0000':<12}{dense_params/1000:<12.0f}{'0.0':<12}{'1.00x':<10}")
            
            # Test each rank
            for rank in config.ranks:
                try:
                    # Create FlashSVD model
                    flash_model = FlashSVDBlock(config.d_model, config.n_heads, rank, config.d_ff).to(config.device).to(config.dtype)
                    
                    # Transplant weights from dense to flash model
                    transplant_weights(dense_model, flash_model)
                    
                    # Benchmark FlashSVD model
                    flash_time, flash_memory, flash_output = benchmark_model(
                        flash_model, x, mask, config.warmup_iters, config.timing_iters)
                    
                    # Compute accuracy
                    rel_error = compute_accuracy(flash_output, dense_output)
                    
                    # Count parameters
                    flash_params = count_attention_parameters(config.d_model, config.n_heads, rank)
                    param_reduction = (1 - flash_params / dense_params) * 100
                    
                    # Compute speedup
                    speedup = dense_time / flash_time
                    
                    print(f"{rank:<6}{flash_time:<10.2f}{flash_memory:<12.1f}{rel_error:<12.4f}{flash_params/1000:<12.0f}{param_reduction:<12.1f}{speedup:<10.2f}x")
                    
                    # Store results
                    results.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'rank': rank,
                        'flash_time': flash_time,
                        'dense_time': dense_time,
                        'flash_memory': flash_memory,
                        'dense_memory': dense_memory,
                        'rel_error': rel_error,
                        'flash_params': flash_params,
                        'dense_params': dense_params,
                        'param_reduction': param_reduction,
                        'speedup': speedup
                    })
                    
                except Exception as e:
                    print(f"{rank:<6}{'ERROR':<10}{'ERROR':<12}{'ERROR':<12}{'ERROR':<12}{'ERROR':<12}{'ERROR':<10}")
                    print(f"Error with rank {rank}: {e}")
                    continue
            
            print()
    
    return results

def print_summary_tables(results, config):
    """Print summary tables for LaTeX"""
    print("\n" + "=" * 100)
    print("📋 SUMMARY TABLES (LaTeX Ready)")
    print("=" * 100)
    
    # Group results by configuration
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for result in results:
        key = (result['batch_size'], result['seq_len'])
        grouped[key].append(result)
    
    # Print accuracy vs rank table
    print("\n🎯 ACCURACY vs RANK TABLE")
    print("-" * 60)
    print("Rank & B=1,M=128 & B=8,M=512 & B=32,M=1024 \\\\")
    print("\\hline")
    
    for rank in config.ranks:
        row = f"{rank}"
        for batch_size, seq_len in [(1, 128), (8, 512), (32, 1024)]:
            if (batch_size, seq_len) in grouped:
                rank_results = [r for r in grouped[(batch_size, seq_len)] if r['rank'] == rank]
                if rank_results:
                    rel_error = rank_results[0]['rel_error']
                    row += f" & {rel_error:.4f}"
                else:
                    row += " & N/A"
            else:
                row += " & N/A"
        row += " \\\\"
        print(row)
    
    # Print speedup vs rank table
    print("\n⚡ SPEEDUP vs RANK TABLE")
    print("-" * 60)
    print("Rank & B=1,M=128 & B=8,M=512 & B=32,M=1024 \\\\")
    print("\\hline")
    
    for rank in config.ranks:
        row = f"{rank}"
        for batch_size, seq_len in [(1, 128), (8, 512), (32, 1024)]:
            if (batch_size, seq_len) in grouped:
                rank_results = [r for r in grouped[(batch_size, seq_len)] if r['rank'] == rank]
                if rank_results:
                    speedup = rank_results[0]['speedup']
                    row += f" & {speedup:.2f}x"
                else:
                    row += " & N/A"
            else:
                row += " & N/A"
        row += " \\\\"
        print(row)
    
    # Print parameter reduction table
    print("\n📊 PARAMETER REDUCTION vs RANK TABLE")
    print("-" * 60)
    print("Rank & Attention Params (K) & Reduction (\\%) & Rel Error & Speedup@B8M512 \\\\")
    print("\\hline")
    
    # Use B=8, M=512 as reference configuration
    ref_key = (8, 512)
    if ref_key in grouped:
        for rank in config.ranks:
            rank_results = [r for r in grouped[ref_key] if r['rank'] == rank]
            if rank_results:
                result = rank_results[0]
                params_k = result['flash_params'] / 1000
                reduction = result['param_reduction']
                rel_error = result['rel_error']
                speedup = result['speedup']
                print(f"{rank} & {params_k:.0f} & {reduction:.1f} & {rel_error:.4f} & {speedup:.2f}x \\\\")

def run_detailed_rank_study():
    """Run detailed study focusing on rank vs performance tradeoffs"""
    print("\n🔍 DETAILED RANK STUDY")
    print("=" * 80)
    
    config = BenchmarkConfig()
    
    # Focus on specific configurations for detailed analysis
    batch_size, seq_len = 8, 512
    x = torch.randn(batch_size, seq_len, config.d_model, device=config.device, dtype=config.dtype)
    mask = create_attention_mask(batch_size, seq_len, device=config.device)
    
    # Create dense baseline
    dense_model = BaselineBlock(config.d_model, config.n_heads, config.d_ff).to(config.device).to(config.dtype)
    dense_time, dense_memory, dense_output = benchmark_model(dense_model, x, mask)
    
    print(f"Configuration: B={batch_size}, M={seq_len}, d_model={config.d_model}")
    print(f"Dense baseline: {dense_time:.2f}ms, {dense_memory:.1f}MB")
    print()
    
    rank_data = []
    
    for rank in config.ranks:
        try:
            flash_model = FlashSVDBlock(config.d_model, config.n_heads, rank, config.d_ff).to(config.device).to(config.dtype)
            transplant_weights(dense_model, flash_model)
            
            flash_time, flash_memory, flash_output = benchmark_model(flash_model, x, mask)
            rel_error = compute_accuracy(flash_output, dense_output)
            
            flash_params = count_attention_parameters(config.d_model, config.n_heads, rank)
            dense_params = count_attention_parameters(config.d_model, config.n_heads)
            param_reduction = (1 - flash_params / dense_params) * 100
            speedup = dense_time / flash_time
            
            rank_data.append({
                'rank': rank,
                'time': flash_time,
                'memory': flash_memory,
                'error': rel_error,
                'params': flash_params,
                'reduction': param_reduction,
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"Error with rank {rank}: {e}")
            continue
    
    # Print detailed analysis
    print("Rank Performance Analysis:")
    print("Rank | Time(ms) | Memory(MB) | Rel Error | Params(K) | Reduction% | Speedup")
    print("-" * 80)
    
    for data in rank_data:
        print(f"{data['rank']:4d} | {data['time']:8.2f} | {data['memory']:10.1f} | {data['error']:9.6f} | {data['params']/1000:9.0f} | {data['reduction']:10.1f} | {data['speedup']:7.2f}x")
    
    # Find optimal ranks for different criteria
    if rank_data:
        best_speed = max(rank_data, key=lambda x: x['speedup'])
        best_params = max(rank_data, key=lambda x: x['reduction'])
        best_accuracy = min(rank_data, key=lambda x: x['error'])
        
        print(f"\n🏆 OPTIMAL RANKS:")
        print(f"Best speedup: rank {best_speed['rank']} ({best_speed['speedup']:.2f}x speedup)")
        print(f"Best param reduction: rank {best_params['rank']} ({best_params['reduction']:.1f}% reduction)")
        print(f"Best accuracy: rank {best_accuracy['rank']} (rel error {best_accuracy['error']:.6f})")
        
        # Find balanced rank (good speedup + accuracy)
        balanced_ranks = [d for d in rank_data if d['error'] < 0.01 and d['speedup'] > 1.0]
        if balanced_ranks:
            balanced = max(balanced_ranks, key=lambda x: x['speedup'] * (1 - x['error']))
            print(f"Balanced choice: rank {balanced['rank']} ({balanced['speedup']:.2f}x speedup, {balanced['error']:.6f} error)")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. This benchmark requires GPU.")
        exit(1)
    
    config = BenchmarkConfig()
    
    print(f"🚀 Starting FlashSVD Attention Benchmarking")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Run comprehensive comparison
    results = run_rank_comparison(config)
    
    # Print summary tables
    print_summary_tables(results, config)
    
    # Run detailed rank study
    run_detailed_rank_study()
    
    print("\n✅ Benchmarking complete!") 