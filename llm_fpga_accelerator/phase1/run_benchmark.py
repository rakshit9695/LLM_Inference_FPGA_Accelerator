import argparse
import json
import time
import sys
from pathlib import Path

# Import the fixed RAG pipeline
try:
    from rag_pipeline import CompactRAG, create_sample_queries
except ImportError:
    print("Error: Could not import rag_pipeline_fixed.py")
    print("Make sure rag_pipeline_fixed.py is in the same directory as this script.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run RAG pipeline benchmark')
    parser.add_argument('--max_docs', type=int, default=1000, help='Maximum number of documents')
    parser.add_argument('--n_queries', type=int, default=50, help='Number of queries to benchmark')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--embedding_model', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model name')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Initializing RAG pipeline with {args.max_docs} documents...")
    
    try:
        # Initialize RAG
        rag = CompactRAG(
            embedding_model_name=args.embedding_model,
            max_docs=args.max_docs
        )
        
        # Load documents and build index
        print("Loading documents...")
        rag.load_documents(dataset_name="wikipedia_sample")
        
        print("Building FAISS index...")
        rag.build_index()
        
        # Generate queries
        print(f"Generating {args.n_queries} test queries...")
        queries = create_sample_queries(n_queries=args.n_queries)
        
        # Run benchmark
        print("Running benchmark...")
        start_time = time.time()
        results_df = rag.benchmark(queries, output_path=output_dir / "benchmark_results.csv")
        total_time = time.time() - start_time
        
        # Get summary
        summary = rag.get_performance_summary()
        summary['total_benchmark_time'] = total_time
        
        # Save summary
        with open(output_dir / "performance_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nBenchmark Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Queries processed: {len(queries)}")
        print(f"Average query time: {summary['avg_total_time']:.4f} seconds")
        print(f"Queries per second: {summary['queries_per_second']:.2f}")
        
        # Print detailed breakdown
        print(f"\nDetailed Performance:")
        print(f"Average retrieval time: {summary['avg_retrieval_time']*1000:.2f} ms")
        print(f"Average generation time: {summary['avg_generation_time']*1000:.2f} ms")
        print(f"Total queries processed: {summary['total_queries']}")
        
        print(f"\nResults saved to {output_dir}")
        
        # Show sample results
        print("\nSample Query Results:")
        print("-" * 50)
        for i, row in results_df.head(3).iterrows():
            if row['response'] != 'ERROR':
                print(f"Query: {row['query'][:80]}...")
                print(f"Response: {row['response'][:100]}...")
                print(f"Time: {row['total_time']*1000:.2f} ms")
                print("-" * 50)
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        print("\nThis might be due to:")
        print("1. Missing dependencies (transformers, sentence-transformers, faiss-cpu)")
        print("2. Network issues downloading models")
        print("3. Insufficient memory")
        print("\nTo install dependencies:")
        print("pip install torch transformers sentence-transformers faiss-cpu numpy pandas")
        sys.exit(1)

if __name__ == "__main__":
    main()