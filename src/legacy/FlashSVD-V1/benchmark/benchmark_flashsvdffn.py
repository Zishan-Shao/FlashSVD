#!/usr/bin/env python3
"""
benchmark_flashsvdffn.py - Comprehensive benchmarking of FlashSVD FFN implementations

This script benchmarks:
1. flashsvd_ffn (original single-stage implementation)  
2. flashsvd_ffn_v1 (two-stage Triton+PyTorch implementation)
3. Dense PyTorch baseline (Linear->GELU->Linear)

Across different FFN rank configurations, measuring:
- Latency (forward pass time)
- Memory usage (peak GPU memory)
- Accuracy (relative error vs dense baseline)
- Parameter count comparison
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add current directory to path to import flashsvdffn modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flashsvdffn import flashsvd_ffn
    from flashsvdffnv1 import flashsvd_ffn_v1
except ImportError as e:
    print(f"Error: Could not import FlashSVD FFN modules: {e}")
    print("Make sure flashsvdffn.py and flashsvdffnv1.py are in the current directory.")
    sys.exit(1)

class FFNBenchmarkConfig:
    """Configuration for FFN benchmarking"""
    def __init__(self):
        # BERT-base FFN dimensions
        self.d_model = 768
        self.d_ff = 3072  # 4 * d_model
        
        # Sequence and batch dimensions
        self.batch_sizes = [16, 64]
        self.seq_lengths = [128, 256, 512, 1024]
        
        # FFN rank configurations to test
        self.ranks = [768, 384, 192, 96]
        
        # Triton kernel block size configurations
        # Note: flashsvd_ffn_v1 only uses BL, BD, BR1, BR2 (no BH)
        # flashsvd_ffn uses all parameters including BH
        self.block_configs = [
            {'BL': 64, 'BD': 128, 'BH': 64, 'BR1': 32, 'BR2': 32},  # Default
            {'BL': 32, 'BD': 64, 'BH': 32, 'BR1': 16, 'BR2': 16},   # Small
            {'BL': 128, 'BD': 256, 'BH': 128, 'BR1': 64, 'BR2': 64}, # Large
        ]
        
        # Benchmark settings
        self.warmup_iters = 10
        self.timing_iters = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

class DenseFFN(nn.Module):
    """Dense baseline FFN implementation"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class FlashSVDFFN(nn.Module):
    """FlashSVD FFN wrapper for both implementations"""
    def __init__(self, d_model, d_ff, rank, version='v1', block_config=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.rank = rank
        self.version = version
        self.block_config = block_config or {'BL': 64, 'BD': 128, 'BH': 64, 'BR1': 32, 'BR2': 32}
        
        # SVD factors: x -> P @ V1 -> activation -> U2 @ V2 -> output
        # P is computed as x @ U1, where U1 is stored
        self.U1 = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V1 = nn.Parameter(torch.randn(rank, d_ff) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        
        self.U2 = nn.Parameter(torch.randn(d_ff, rank) * 0.02)
        self.V2 = nn.Parameter(torch.randn(rank, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # Compute P = x @ U1
        P = x @ self.U1  # [B, L, rank]
        
        if self.version == 'v1':
            # flashsvd_ffn_v1 only accepts BL, BD, BR1, BR2 (no BH)
            v1_config = {k: v for k, v in self.block_config.items() if k in ['BL', 'BD', 'BR1', 'BR2']}
            return flashsvd_ffn_v1(
                P, self.V1, self.U2, self.V2, self.b1, self.b2,
                **v1_config
            )
        elif self.version == 'original':
            return flashsvd_ffn(
                P, self.V1, self.U2, self.V2, self.b1, self.b2,
                **self.block_config
            )
        else:
            raise ValueError(f"Unknown version: {self.version}")

def create_svd_factors_from_dense(dense_ffn, rank):
    """Create SVD factors from a dense FFN using SVD decomposition"""
    W1 = dense_ffn.linear1.weight.data.t()  # [d_model, d_ff]
    b1 = dense_ffn.linear1.bias.data
    W2 = dense_ffn.linear2.weight.data.t()  # [d_ff, d_model]
    b2 = dense_ffn.linear2.bias.data
    
    # SVD decomposition of W1 and W2
    U1, S1, V1t = torch.linalg.svd(W1.float(), full_matrices=False)
    U2, S2, V2t = torch.linalg.svd(W2.float(), full_matrices=False)
    
    # Truncate to rank
    U1_r = (U1[:, :rank] * S1[:rank]).to(W1.dtype)
    V1_r = V1t[:rank, :].to(W1.dtype)
    U2_r = (U2[:, :rank] * S2[:rank]).to(W2.dtype)
    V2_r = V2t[:rank, :].to(W2.dtype)
    
    return U1_r, V1_r, b1, U2_r, V2_r, b2

def transplant_svd_weights(flash_ffn, dense_ffn):
    """Transplant weights from dense FFN to FlashSVD FFN"""
    U1, V1, b1, U2, V2, b2 = create_svd_factors_from_dense(dense_ffn, flash_ffn.rank)
    
    flash_ffn.U1.data.copy_(U1)
    flash_ffn.V1.data.copy_(V1)
    flash_ffn.b1.data.copy_(b1)
    flash_ffn.U2.data.copy_(U2)
    flash_ffn.V2.data.copy_(V2)
    flash_ffn.b2.data.copy_(b2)

def benchmark_ffn(ffn_model, x, warmup_iters=10, timing_iters=50):
    """Benchmark an FFN model's forward pass"""
    ffn_model.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iters):
            _ = ffn_model(x)
        torch.cuda.synchronize()
        
        # Clear memory stats and time the iterations
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        
        for _ in range(timing_iters):
            output = ffn_model(x)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / timing_iters
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return avg_time_ms, peak_memory_mb, output

def compute_ffn_accuracy(output_flash, output_dense):
    """Compute relative error between FlashSVD and dense outputs"""
    diff_norm = (output_flash.float() - output_dense.float()).norm()
    dense_norm = output_dense.float().norm()
    return (diff_norm / dense_norm).item()

def count_ffn_parameters(d_model, d_ff, rank=None):
    """Count FFN parameters for dense vs FlashSVD"""
    if rank is None:  # Dense FFN
        # Linear1: d_model * d_ff + d_ff
        # Linear2: d_ff * d_model + d_model
        return d_model * d_ff + d_ff + d_ff * d_model + d_model
    else:  # FlashSVD FFN
        # U1: d_model * rank, V1: rank * d_ff, b1: d_ff
        # U2: d_ff * rank, V2: rank * d_model, b2: d_model
        return (d_model * rank + rank * d_ff + d_ff + 
                d_ff * rank + rank * d_model + d_model)

def run_ffn_rank_comparison(config):
    """Run comprehensive comparison across different FFN ranks"""
    print("🔬 FlashSVD FFN Rank Benchmarking")
    print("=" * 80)
    print(f"FFN: d_model={config.d_model}, d_ff={config.d_ff}")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print("=" * 80)
    
    results = []
    
    # Test different batch sizes and sequence lengths
    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lengths:
            print(f"\n📊 Testing B={batch_size}, M={seq_len}")
            print("-" * 80)
            
            # Create input data
            x = torch.randn(batch_size, seq_len, config.d_model, 
                          device=config.device, dtype=config.dtype)
            
            # Create dense baseline
            dense_ffn = DenseFFN(config.d_model, config.d_ff).to(config.device).to(config.dtype)
            
            # Benchmark dense model
            dense_time, dense_memory, dense_output = benchmark_ffn(
                dense_ffn, x, config.warmup_iters, config.timing_iters)
            
            dense_params = count_ffn_parameters(config.d_model, config.d_ff)
            
            print(f"{'Version':<12}{'Rank':<6}{'Time(ms)':<10}{'Memory(MB)':<12}{'Rel Error':<12}{'Params(K)':<12}{'Reduction%':<12}{'Speedup':<10}")
            print("-" * 100)
            
            # Dense baseline row
            print(f"{'Dense':<12}{'Full':<6}{dense_time:<10.2f}{dense_memory:<12.1f}{'0.0000':<12}{dense_params/1000:<12.0f}{'0.0':<12}{'1.00x':<10}")
            
            # Test each rank for both versions
            for rank in config.ranks:
                if rank > min(config.d_model, config.d_ff):
                    continue
                    
                for version in ['v1', 'original']:
                    try:
                        # Create FlashSVD FFN
                        flash_ffn = FlashSVDFFN(config.d_model, config.d_ff, rank, 
                                              version=version, block_config=config.block_configs[0]).to(config.device).to(config.dtype)
                        
                        # Transplant weights from dense to flash model
                        transplant_svd_weights(flash_ffn, dense_ffn)
                        
                        # Benchmark FlashSVD model
                        flash_time, flash_memory, flash_output = benchmark_ffn(
                            flash_ffn, x, config.warmup_iters, config.timing_iters)
                        
                        # Compute accuracy
                        rel_error = compute_ffn_accuracy(flash_output, dense_output)
                        
                        # Count parameters
                        flash_params = count_ffn_parameters(config.d_model, config.d_ff, rank)
                        param_reduction = (1 - flash_params / dense_params) * 100
                        
                        # Compute speedup
                        speedup = dense_time / flash_time
                        
                        version_str = f"Flash-{version}"
                        print(f"{version_str:<12}{rank:<6}{flash_time:<10.2f}{flash_memory:<12.1f}{rel_error:<12.4f}{flash_params/1000:<12.0f}{param_reduction:<12.1f}{speedup:<10.2f}x")
                        
                        # Store results
                        results.append({
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'version': version,
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
                        version_str = f"Flash-{version}"
                        print(f"{version_str:<12}{rank:<6}{'ERROR':<10}{'ERROR':<12}{'ERROR':<12}{'ERROR':<12}{'ERROR':<12}{'ERROR':<10}")
                        print(f"Error with {version} rank {rank}: {e}")
                        continue
            
            print()
    
    return results

def run_block_size_analysis(config):
    """Analyze impact of different Triton block sizes"""
    print("\n🔧 BLOCK SIZE ANALYSIS")
    print("=" * 80)
    
    batch_size, seq_len = 8, 512
    rank = 192  # Medium rank for testing
    
    x = torch.randn(batch_size, seq_len, config.d_model, device=config.device, dtype=config.dtype)
    dense_ffn = DenseFFN(config.d_model, config.d_ff).to(config.device).to(config.dtype)
    _, _, dense_output = benchmark_ffn(dense_ffn, x)
    
    print(f"Testing rank={rank}, B={batch_size}, M={seq_len}")
    print(f"{'Version':<12}{'BL':<4}{'BD':<4}{'BH':<4}{'BR1':<4}{'BR2':<4}{'Time(ms)':<10}{'Memory(MB)':<12}{'Rel Error':<12}")
    print("-" * 80)
    
    for version in ['v1', 'original']:
        for i, block_config in enumerate(config.block_configs):
            try:
                flash_ffn = FlashSVDFFN(config.d_model, config.d_ff, rank, 
                                      version=version, block_config=block_config).to(config.device).to(config.dtype)
                transplant_svd_weights(flash_ffn, dense_ffn)
                
                flash_time, flash_memory, flash_output = benchmark_ffn(flash_ffn, x)
                rel_error = compute_ffn_accuracy(flash_output, dense_output)
                
                version_str = f"Flash-{version}"
                # For v1, BH is not used, so show N/A
                bh_value = block_config['BH'] if version == 'original' else 'N/A'
                print(f"{version_str:<12}{block_config['BL']:<4}{block_config['BD']:<4}{bh_value:<4}{block_config['BR1']:<4}{block_config['BR2']:<4}{flash_time:<10.2f}{flash_memory:<12.1f}{rel_error:<12.6f}")
                
            except Exception as e:
                print(f"Error with {version} block config {i}: {e}")

def print_ffn_summary_tables(results, config):
    """Print summary tables for LaTeX"""
    print("\n" + "=" * 100)
    print("📋 FFN SUMMARY TABLES (LaTeX Ready)")
    print("=" * 100)
    
    # Group results by configuration
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for result in results:
        key = (result['batch_size'], result['seq_len'], result['version'])
        grouped[key].append(result)
    
    # Version comparison table
    print("\n⚡ VERSION COMPARISON (B=8, M=512)")
    print("-" * 70)
    print("Rank & FlashSVD-v1 Time & FlashSVD-orig Time & v1 Error & orig Error & v1 Speedup & orig Speedup \\\\")
    print("\\hline")
    
    ref_key_v1 = (8, 512, 'v1')
    ref_key_orig = (8, 512, 'original')
    
    if ref_key_v1 in grouped and ref_key_orig in grouped:
        ranks_to_show = [384, 192, 96, 48, 24]
        for rank in ranks_to_show:
            v1_results = [r for r in grouped[ref_key_v1] if r['rank'] == rank]
            orig_results = [r for r in grouped[ref_key_orig] if r['rank'] == rank]
            
            if v1_results and orig_results:
                v1_r = v1_results[0]
                orig_r = orig_results[0]
                print(f"{rank} & {v1_r['flash_time']:.2f} & {orig_r['flash_time']:.2f} & {v1_r['rel_error']:.4f} & {orig_r['rel_error']:.4f} & {v1_r['speedup']:.2f}x & {orig_r['speedup']:.2f}x \\\\")
    
    # Parameter efficiency table
    print("\n📊 PARAMETER EFFICIENCY TABLE")
    print("-" * 80)
    print("Rank & FFN Params (K) & Reduction (\\%) & v1 Error & orig Error & v1 Time & orig Time \\\\")
    print("\\hline")
    
    if ref_key_v1 in grouped and ref_key_orig in grouped:
        for rank in [384, 288, 192, 144, 96, 64, 48, 32, 24, 16]:
            v1_results = [r for r in grouped[ref_key_v1] if r['rank'] == rank]
            orig_results = [r for r in grouped[ref_key_orig] if r['rank'] == rank]
            
            if v1_results and orig_results:
                v1_r = v1_results[0]
                orig_r = orig_results[0]
                params_k = v1_r['flash_params'] / 1000
                reduction = v1_r['param_reduction']
                print(f"{rank} & {params_k:.0f} & {reduction:.1f} & {v1_r['rel_error']:.4f} & {orig_r['rel_error']:.4f} & {v1_r['flash_time']:.2f} & {orig_r['flash_time']:.2f} \\\\")

def run_detailed_ffn_study():
    """Run detailed study focusing on FFN rank vs performance tradeoffs"""
    print("\n🔍 DETAILED FFN RANK STUDY")
    print("=" * 80)
    
    config = FFNBenchmarkConfig()
    
    # Focus on specific configuration
    batch_size, seq_len = 8, 512
    x = torch.randn(batch_size, seq_len, config.d_model, device=config.device, dtype=config.dtype)
    
    # Create dense baseline
    dense_ffn = DenseFFN(config.d_model, config.d_ff).to(config.device).to(config.dtype)
    dense_time, dense_memory, dense_output = benchmark_ffn(dense_ffn, x)
    dense_params = count_ffn_parameters(config.d_model, config.d_ff)
    
    print(f"Configuration: B={batch_size}, M={seq_len}, d_model={config.d_model}, d_ff={config.d_ff}")
    print(f"Dense baseline: {dense_time:.2f}ms, {dense_memory:.1f}MB, {dense_params/1000:.0f}K params")
    print()
    
    # Test both versions across ranks
    for version in ['v1', 'original']:
        print(f"\n📈 FlashSVD-{version} Performance Analysis:")
        print("Rank | Time(ms) | Memory(MB) | Rel Error | Params(K) | Reduction% | Speedup")
        print("-" * 80)
        
        best_results = []
        
        for rank in config.ranks:
            if rank > min(config.d_model, config.d_ff):
                continue
                
            try:
                flash_ffn = FlashSVDFFN(config.d_model, config.d_ff, rank, version=version).to(config.device).to(config.dtype)
                transplant_svd_weights(flash_ffn, dense_ffn)
                
                flash_time, flash_memory, flash_output = benchmark_ffn(flash_ffn, x)
                rel_error = compute_ffn_accuracy(flash_output, dense_output)
                
                flash_params = count_ffn_parameters(config.d_model, config.d_ff, rank)
                param_reduction = (1 - flash_params / dense_params) * 100
                speedup = dense_time / flash_time
                
                print(f"{rank:4d} | {flash_time:8.2f} | {flash_memory:10.1f} | {rel_error:9.6f} | {flash_params/1000:9.0f} | {param_reduction:10.1f} | {speedup:7.2f}x")
                
                best_results.append({
                    'rank': rank,
                    'time': flash_time,
                    'error': rel_error,
                    'reduction': param_reduction,
                    'speedup': speedup
                })
                
            except Exception as e:
                print(f"{rank:4d} | ERROR: {e}")
                continue
        
        # Find optimal ranks for this version
        if best_results:
            best_speed = max(best_results, key=lambda x: x['speedup'])
            best_params = max(best_results, key=lambda x: x['reduction'])
            best_accuracy = min(best_results, key=lambda x: x['error'])
            
            print(f"\n🏆 FlashSVD-{version} OPTIMAL RANKS:")
            print(f"Best speedup: rank {best_speed['rank']} ({best_speed['speedup']:.2f}x speedup)")
            print(f"Best param reduction: rank {best_params['rank']} ({best_params['reduction']:.1f}% reduction)")
            print(f"Best accuracy: rank {best_accuracy['rank']} (rel error {best_accuracy['error']:.6f})")
            
            # Find balanced rank
            balanced_ranks = [d for d in best_results if d['error'] < 0.01 and d['speedup'] > 1.0]
            if balanced_ranks:
                balanced = max(balanced_ranks, key=lambda x: x['speedup'] * (1 - x['error']))
                print(f"Balanced choice: rank {balanced['rank']} ({balanced['speedup']:.2f}x speedup, {balanced['error']:.6f} error)")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. This benchmark requires GPU.")
        exit(1)
    
    config = FFNBenchmarkConfig()
    
    print(f"🚀 Starting FlashSVD FFN Benchmarking")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Run comprehensive comparison
    results = run_ffn_rank_comparison(config)
    
    # Run block size analysis
    run_block_size_analysis(config)
    
    # Print summary tables
    print_ffn_summary_tables(results, config)
    
    # Run detailed study
    run_detailed_ffn_study()
    
    print("\n✅ FFN Benchmarking complete!") 