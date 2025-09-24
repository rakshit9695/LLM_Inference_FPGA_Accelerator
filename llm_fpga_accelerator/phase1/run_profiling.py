#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))

try:
    from rag_pipeline import CompactRAG
    from profiler import RAGProfiler
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure rag_pipeline_fixed.py and profiler_fixed.py are in the same directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive RAG pipeline profiling')
    parser.add_argument('--max_docs', type=int, default=500, help='Maximum number of documents')
    parser.add_argument('--n_queries', type=int, default=10, help='Number of queries for profiling')
    parser.add_argument('--embedding_model', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model name')
    parser.add_argument('--profile_attention', action='store_true',
                       help='Include attention mechanism profiling')
    parser.add_argument('--quick', action='store_true',
                       help='Quick profiling mode (fewer tests)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.max_docs = min(args.max_docs, 100)
        args.n_queries = min(args.n_queries, 5)
    
    print("="*60)
    print("RAG PIPELINE PROFILING SUITE")
    print("="*60)
    print(f"Documents: {args.max_docs}")
    print(f"Test queries: {args.n_queries}")
    print(f"Embedding model: {args.embedding_model}")
    print("="*60)
    
    try:
        # Initialize RAG pipeline
        print("\n1. Initializing RAG pipeline...")
        rag = CompactRAG(
            embedding_model_name=args.embedding_model,
            max_docs=args.max_docs
        )
        
        print("2. Loading documents and building index...")
        rag.load_documents(dataset_name="wikipedia_sample")
        rag.build_index()
        
        # Initialize profiler
        print("3. Setting up profiler...")
        profiler = RAGProfiler(rag)
        
        # Run profiling
        print("4. Running comprehensive profiling...")
        report = profiler.run_comprehensive_profiling(n_queries=args.n_queries)
        
        # Display results
        print("\n" + "="*60)
        print("PROFILING RESULTS")
        print("="*60)
        
        print("\nPERFORMANCE SUMMARY:")
        for component, metrics in report['summary'].items():
            print(f"\n{component.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
        
        print(f"\nHOTSPOTS IDENTIFIED ({len(report['hotspots'])}):")
        for i, hotspot in enumerate(report['hotspots'], 1):
            print(f"  {i}. {hotspot['component']} - {hotspot['time_ms']:.2f}ms")
            print(f"     Severity: {hotspot['severity']}")
            print(f"     Description: {hotspot['description']}")
        
        print(f"\nOPTIMIZATION RECOMMENDATIONS ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. Component: {rec['component']}")
            print(f"     Recommendation: {rec['recommendation']}")
            print(f"     Expected speedup: {rec['expected_speedup']}")
            print(f"     Hardware target: {rec['hardware_target']}")
        
        # Identify primary acceleration target
        if report['hotspots']:
            primary_target = max(report['hotspots'], key=lambda x: x['time_ms'])
            print(f"\nPRIMARY ACCELERATION TARGET: {primary_target['component']}")
            print(f"Time savings potential: {primary_target['time_ms']:.2f}ms per query")
        
        print(f"\nDetailed results saved to: ./profiling/")
        print("- Raw data: raw_profiling_data.json")
        print("- Report: performance_report.json") 
        print("- Plots: plots/")
        print("- Traces: traces/ (load in Chrome: chrome://tracing)")
        
        print("\n✅ Profiling completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Profiling interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        print("\nThis might be due to:")
        print("1. Missing dependencies (torch, transformers, sentence-transformers, faiss-cpu)")
        print("2. Insufficient memory")
        print("3. Model download issues")
        print("\nTo install dependencies:")
        print("pip install torch transformers sentence-transformers faiss-cpu matplotlib seaborn psutil memory-profiler")
        sys.exit(1)

if __name__ == "__main__":
    main()