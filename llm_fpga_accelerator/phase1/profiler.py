# Fixed Profiler Implementation

import torch
import numpy as np
import pandas as pd
import time
import psutil
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import threading
import queue
from contextlib import contextmanager

# PyTorch profiling
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule, tensorboard_trace_handler

# Memory profiling
from memory_profiler import profile as memory_profile
import tracemalloc
import gc

# Import our RAG pipeline
from rag_pipeline import CompactRAG, create_sample_queries

class RAGProfiler:
    """Comprehensive profiler for RAG pipeline components."""
    
    def __init__(self, rag_pipeline: CompactRAG):
        self.rag = rag_pipeline
        self.profiling_results = {
            'embedding_profiles': [],
            'retrieval_profiles': [],
            'generation_profiles': [],
            'memory_traces': [],
            'system_metrics': []
        }
        
        # Setup profiling directories
        self.profile_dir = Path("profiling")
        self.trace_dir = self.profile_dir / "traces"
        self.plot_dir = self.profile_dir / "plots"
        
        for dir_path in [self.profile_dir, self.trace_dir, self.plot_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

    @contextmanager
    def memory_tracer(self, trace_name: str):
        """Context manager for memory tracing."""
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.profiling_results['memory_traces'].append({
            'trace_name': trace_name,
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'memory_delta_mb': end_memory - start_memory,
            'peak_traced_mb': peak / 1024 / 1024,
            'current_traced_mb': current / 1024 / 1024
        })

    def profile_embedding_generation(self, queries: List[str], batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Any]:
        """Profile embedding generation with different batch sizes."""
        print("Profiling embedding generation...")
        
        results = {
            'batch_profiles': [],
            'torch_profiles': [],
            'operation_breakdown': {}
        }
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Prepare batch
            query_batch = queries[:batch_size]
            
            # Profile with PyTorch profiler
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("embedding_generation"):
                    with self.memory_tracer(f"embedding_batch_{batch_size}"):
                        start_time = time.perf_counter()
                        embeddings = self.rag.embedder.encode(
                            query_batch, 
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        end_time = time.perf_counter()
            
            # Store batch results
            batch_time = end_time - start_time
            results['batch_profiles'].append({
                'batch_size': batch_size,
                'total_time': batch_time,
                'time_per_query': batch_time / batch_size,
                'queries_per_second': batch_size / batch_time,
                'embedding_shape': embeddings.shape,
                'memory_mb': embeddings.nbytes / 1024 / 1024
            })
            
            # Export trace
            trace_path = self.trace_dir / f"embedding_batch_{batch_size}.json"
            prof.export_chrome_trace(str(trace_path))
            
            # Analyze torch profiler results
            torch_profile = self._analyze_torch_profile(prof, f"embedding_batch_{batch_size}")
            results['torch_profiles'].append(torch_profile)
        
        self.profiling_results['embedding_profiles'] = results
        return results

    def profile_faiss_retrieval(self, queries: List[str], k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Profile FAISS retrieval with different k values."""
        print("Profiling FAISS retrieval...")
        
        results = {
            'k_profiles': [],
            'operation_breakdown': {}
        }
        
        for k in k_values:
            print(f"  Testing k={k}")
            
            times = []
            distances_list = []
            indices_list = []
            
            for query in queries[:10]:  # Test on first 10 queries
                with self.memory_tracer(f"retrieval_k_{k}"):
                    start_time = time.perf_counter()
                    
                    # Generate query embedding
                    query_emb = self.rag.embedder.encode([query], convert_to_numpy=True)
                    
                    # FAISS search
                    distances, indices = self.rag.faiss_index.search(
                        query_emb.astype('float32'), k
                    )
                    
                    end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                distances_list.append(distances)
                indices_list.append(indices)
            
            # Aggregate results
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results['k_profiles'].append({
                'k': k,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': np.min(times),
                'max_time': np.max(times),
                'searches_per_second': 1.0 / avg_time,
                'avg_distances': np.mean([d.flatten() for d in distances_list])
            })
        
        self.profiling_results['retrieval_profiles'] = results
        return results

    def profile_attention_mechanism(self, input_lengths: List[int] = [64, 128, 256, 512]) -> Dict[str, Any]:
        """Profile attention mechanism computation patterns."""
        print("Profiling attention mechanism...")
        
        results = {
            'attention_profiles': [],
            'matrix_ops': []
        }
        
        # Simulate attention computation patterns
        embed_dim = self.rag.vector_dim
        num_heads = 8
        head_dim = embed_dim // num_heads
        
        for seq_len in input_lengths:
            print(f"  Testing sequence length: {seq_len}")
            
            # Create dummy input tensors
            query = torch.randn(1, seq_len, embed_dim)
            key = torch.randn(1, seq_len, embed_dim)
            value = torch.randn(1, seq_len, embed_dim)
            
            # Reshape for multi-head attention
            batch_size = 1
            q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Profile attention computation
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True
            ) as prof:
                with record_function(f"attention_seq_{seq_len}"):
                    start_time = time.perf_counter()
                    
                    # Q @ K^T
                    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                    
                    # Softmax
                    attn_weights = torch.softmax(scores, dim=-1)
                    
                    # Attention @ V
                    attn_output = torch.matmul(attn_weights, v)
                    
                    # Concatenate heads
                    attn_output = attn_output.transpose(1, 2).contiguous().view(
                        batch_size, seq_len, embed_dim
                    )
                    
                    end_time = time.perf_counter()
            
            attention_time = end_time - start_time
            
            # Calculate FLOPS for attention
            # QK^T: batch_size * num_heads * seq_len^2 * head_dim
            # Softmax: approximately 3 * batch_size * num_heads * seq_len^2
            # AttnV: batch_size * num_heads * seq_len^2 * head_dim
            flops_qkt = batch_size * num_heads * seq_len * seq_len * head_dim
            flops_softmax = 3 * batch_size * num_heads * seq_len * seq_len
            flops_attnv = batch_size * num_heads * seq_len * seq_len * head_dim
            total_flops = flops_qkt + flops_softmax + flops_attnv
            
            results['attention_profiles'].append({
                'seq_len': seq_len,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'attention_time': attention_time,
                'flops': total_flops,
                'gflops_per_sec': total_flops / attention_time / 1e9,
                'memory_mb': (q.nbytes + k.nbytes + v.nbytes + 
                             scores.nbytes + attn_weights.nbytes + 
                             attn_output.nbytes) / 1024 / 1024
            })
            
            # Export trace
            trace_path = self.trace_dir / f"attention_seq_{seq_len}.json"
            prof.export_chrome_trace(str(trace_path))
            
            # Store matrix operation details
            results['matrix_ops'].append({
                'operation': 'QK_transpose',
                'seq_len': seq_len,
                'matrix_shape': (num_heads * seq_len, head_dim),
                'flops': flops_qkt
            })
            
            results['matrix_ops'].append({
                'operation': 'attention_values',
                'seq_len': seq_len,
                'matrix_shape': (num_heads * seq_len, seq_len),
                'flops': flops_attnv
            })
        
        return results

    def _analyze_torch_profile(self, prof, profile_name: str) -> Dict[str, Any]:
        """Analyze PyTorch profiler results."""
        key_averages = prof.key_averages()
        
        # Get top operations by CPU time
        top_ops = sorted(
            [item for item in key_averages],
            key=lambda x: x.cpu_time_total,
            reverse=True
        )[:10]
        
        analysis = {
            'profile_name': profile_name,
            'total_time': sum(item.cpu_time_total for item in key_averages),
            'top_operations': []
        }
        
        for op in top_ops:
            analysis['top_operations'].append({
                'name': op.key,
                'cpu_time_total': op.cpu_time_total,
                'cpu_time_avg': op.cpu_time,
                'count': op.count,
                'input_shapes': str(op.input_shapes) if hasattr(op, 'input_shapes') else 'N/A'
            })
        
        return analysis

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        print("Generating performance report...")
        
        report = {
            'summary': {},
            'hotspots': [],
            'recommendations': []
        }
        
        # Analyze embedding performance
        if self.profiling_results['embedding_profiles']:
            embedding_data = self.profiling_results['embedding_profiles']['batch_profiles']
            avg_embedding_time = np.mean([p['time_per_query'] for p in embedding_data])
            
            report['summary']['embedding'] = {
                'avg_time_per_query_ms': avg_embedding_time * 1000,
                'best_throughput_qps': max(p['queries_per_second'] for p in embedding_data),
                'memory_usage_mb': max(p['memory_mb'] for p in embedding_data)
            }
            
            if avg_embedding_time > 0.05:  # >50ms per query
                report['hotspots'].append({
                    'component': 'embedding_generation',
                    'severity': 'high',
                    'time_ms': avg_embedding_time * 1000,
                    'description': 'Embedding generation is a major bottleneck'
                })
        
        # Analyze retrieval performance
        if self.profiling_results['retrieval_profiles']:
            retrieval_data = self.profiling_results['retrieval_profiles']['k_profiles']
            avg_retrieval_time = np.mean([p['avg_time'] for p in retrieval_data])
            
            report['summary']['retrieval'] = {
                'avg_time_ms': avg_retrieval_time * 1000,
                'best_throughput_sps': max(p['searches_per_second'] for p in retrieval_data)
            }
            
            if avg_retrieval_time > 0.01:  # >10ms
                report['hotspots'].append({
                    'component': 'faiss_retrieval',
                    'severity': 'medium',
                    'time_ms': avg_retrieval_time * 1000,
                    'description': 'FAISS retrieval could benefit from optimization'
                })
        
        # Generate recommendations
        hotspot_components = [h['component'] for h in report['hotspots']]
        
        if 'embedding_generation' in hotspot_components:
            report['recommendations'].append({
                'component': 'embedding_generation',
                'recommendation': 'Implement dedicated matrix multiplication accelerator for transformer layers',
                'expected_speedup': '5-10x',
                'hardware_target': 'Systolic array for GEMM operations'
            })
        
        if 'faiss_retrieval' in hotspot_components:
            report['recommendations'].append({
                'component': 'faiss_retrieval',
                'recommendation': 'Implement vector dot-product accelerator for similarity search',
                'expected_speedup': '3-5x',
                'hardware_target': 'Parallel MAC units for vector operations'
            })
        
        return report

    def plot_profiling_results(self):
        """Generate visualization plots for profiling results."""
        print("Generating profiling plots...")
        
        # Set matplotlib style safely
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass  # Use default style if seaborn not available
        
        # Plot embedding performance vs batch size
        if self.profiling_results['embedding_profiles']:
            self._plot_embedding_performance()
        
        # Plot retrieval performance vs k
        if self.profiling_results['retrieval_profiles']:
            self._plot_retrieval_performance()
        
        # Plot memory usage
        if self.profiling_results['memory_traces']:
            self._plot_memory_usage()

    def _plot_embedding_performance(self):
        """Plot embedding generation performance."""
        data = self.profiling_results['embedding_profiles']['batch_profiles']
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Embedding Generation Performance Analysis', fontsize=16)
        
        # Batch size vs time per query
        axes[0, 0].plot(df['batch_size'], df['time_per_query'] * 1000, 'bo-')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Time per Query (ms)')
        axes[0, 0].set_title('Latency vs Batch Size')
        axes[0, 0].grid(True)
        
        # Batch size vs throughput
        axes[0, 1].plot(df['batch_size'], df['queries_per_second'], 'ro-')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Queries per Second')
        axes[0, 1].set_title('Throughput vs Batch Size')
        axes[0, 1].grid(True)
        
        # Memory usage
        batch_size_str = [str(bs) for bs in df['batch_size']]
        axes[1, 0].bar(batch_size_str, df['memory_mb'])
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Batch Size')
        
        # Efficiency (queries per second per MB)
        efficiency = df['queries_per_second'] / df['memory_mb']
        axes[1, 1].plot(df['batch_size'], efficiency, 'go-')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('QPS per MB')
        axes[1, 1].set_title('Memory Efficiency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'embedding_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_retrieval_performance(self):
        """Plot retrieval performance."""
        data = self.profiling_results['retrieval_profiles']['k_profiles']
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('FAISS Retrieval Performance Analysis', fontsize=16)
        
        # K vs average time
        axes[0].errorbar(df['k'], df['avg_time'] * 1000, 
                        yerr=df['std_time'] * 1000, 
                        fmt='bo-', capsize=5)
        axes[0].set_xlabel('Number of Retrieved Documents (k)')
        axes[0].set_ylabel('Average Time (ms)')
        axes[0].set_title('Retrieval Latency vs k')
        axes[0].grid(True)
        
        # K vs throughput
        axes[1].plot(df['k'], df['searches_per_second'], 'ro-')
        axes[1].set_xlabel('Number of Retrieved Documents (k)')
        axes[1].set_ylabel('Searches per Second')
        axes[1].set_title('Retrieval Throughput vs k')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'retrieval_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_memory_usage(self):
        """Plot memory usage traces."""
        df = pd.DataFrame(self.profiling_results['memory_traces'])
        
        plt.figure(figsize=(12, 8))
        
        # Memory delta by operation
        plt.subplot(2, 1, 1)
        x_positions = range(len(df))
        plt.bar(x_positions, df['memory_delta_mb'])
        plt.xticks(x_positions, df['trace_name'], rotation=45, ha='right')
        plt.ylabel('Memory Delta (MB)')
        plt.title('Memory Usage by Operation')
        plt.grid(True, alpha=0.3)
        
        # Peak memory usage
        plt.subplot(2, 1, 2)
        plt.bar(x_positions, df['peak_traced_mb'])
        plt.xticks(x_positions, df['trace_name'], rotation=45, ha='right')
        plt.ylabel('Peak Memory (MB)')
        plt.title('Peak Memory Usage by Operation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_comprehensive_profiling(self, n_queries: int = 20) -> Dict[str, Any]:
        """Run complete profiling suite."""
        print(f"Running comprehensive profiling with {n_queries} queries...")
        
        # Generate test queries
        queries = create_sample_queries(n_queries)
        
        # Profile embedding generation
        embedding_results = self.profile_embedding_generation(queries)
        
        # Profile FAISS retrieval
        retrieval_results = self.profile_faiss_retrieval(queries)
        
        # Profile attention mechanisms
        attention_results = self.profile_attention_mechanism()
        self.profiling_results['attention_profiles'] = attention_results
        
        # Generate comprehensive report
        report = self.generate_performance_report()
        
        # Create visualizations
        self.plot_profiling_results()
        
        # Save all results
        self.save_profiling_results()
        
        return report

    def save_profiling_results(self):
        """Save all profiling results to files."""
        # Save raw profiling data
        with open(self.profile_dir / 'raw_profiling_data.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._convert_numpy_for_json(self.profiling_results)
            json.dump(json_data, f, indent=2, default=str)
        
        # Save performance report
        report = self.generate_performance_report()
        with open(self.profile_dir / 'performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Profiling results saved to {self.profile_dir}")

    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

# Usage script
if __name__ == "__main__":
    from rag_pipeline import CompactRAG
    
    print("Initializing RAG pipeline for profiling...")
    
    # Initialize RAG pipeline
    rag = CompactRAG(max_docs=100)  # Smaller for testing
    rag.load_documents(dataset_name="wikipedia_sample")
    rag.build_index()
    
    # Initialize profiler
    profiler = RAGProfiler(rag)
    
    # Run comprehensive profiling
    report = profiler.run_comprehensive_profiling(n_queries=10)
    
    # Print summary
    print("\n" + "="*50)
    print("PROFILING SUMMARY")
    print("="*50)
    
    for component, metrics in report['summary'].items():
        print(f"\n{component.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\nHOTSPOTS IDENTIFIED:")
    for hotspot in report['hotspots']:
        print(f"  - {hotspot['component']}: {hotspot['time_ms']:.2f}ms ({hotspot['severity']} priority)")
    
    print("\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  - {rec['component']}: {rec['recommendation']}")
        print(f"    Expected speedup: {rec['expected_speedup']}")
        print(f"    Hardware target: {rec['hardware_target']}")