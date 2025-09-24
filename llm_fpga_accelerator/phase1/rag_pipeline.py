# Fixed RAG Pipeline Implementation

import os
import time
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

# ML Libraries
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompactRAG:
    """Compact RAG pipeline optimized for profiling and hardware acceleration research."""
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generation_model_name: str = "microsoft/DialoGPT-small",
                 max_docs: int = 1000,
                 vector_dim: int = None):
        
        self.max_docs = max_docs
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        
        # Initialize models
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.vector_dim = vector_dim or self.embedder.get_sentence_embedding_dimension()
        
        logger.info(f"Loading generation model: {generation_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.generator = AutoModelForCausalLM.from_pretrained(generation_model_name)
        
        # Initialize document store and index
        self.documents = []
        self.document_embeddings = None
        self.faiss_index = None
        
        # Profiling metrics
        self.metrics = {
            'embedding_times': [],
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'memory_usage': []
        }

    def load_documents(self, documents: List[str] = None, dataset_name: str = None):
        """Load documents from list or dataset."""
        if documents:
            self.documents = documents[:self.max_docs]
        elif dataset_name:
            # Load from Hugging Face datasets
            if dataset_name == "wikipedia_sample":
                # Create sample Wikipedia-like documents
                sample_docs = [
                    "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans.",
                    "Machine learning is a method of data analysis that automates analytical model building.",
                    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
                    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.",
                    "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images.",
                    "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
                    "Supervised learning is a machine learning paradigm where algorithms learn from labeled training data.",
                    "Unsupervised learning finds hidden patterns or intrinsic structures in input data without labeled examples.",
                    "Reinforcement learning is an area where an agent learns to behave in an environment by performing actions.",
                    "Transformers are deep learning models that use attention mechanisms to process sequential data.",
                    "BERT (Bidirectional Encoder Representations from Transformers) revolutionized natural language understanding.",
                    "GPT (Generative Pre-trained Transformer) models excel at text generation and completion tasks.",
                    "Convolutional neural networks (CNNs) are particularly effective for image recognition and processing.",
                    "Recurrent neural networks (RNNs) can process sequences of variable length through hidden state memory.",
                    "Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem in traditional RNNs.",
                    "Support vector machines (SVMs) are supervised learning models used for classification and regression.",
                    "Random forests combine multiple decision trees to improve predictive accuracy and control overfitting.",
                    "Gradient boosting builds models sequentially, with each new model correcting errors from previous ones.",
                    "K-means clustering partitions data into k clusters based on feature similarity.",
                    "Principal component analysis (PCA) reduces data dimensionality while preserving important information.",
                    "Cross-validation is a technique for assessing how model results generalize to independent datasets.",
                    "Overfitting occurs when a model learns training data too specifically and fails to generalize.",
                    "Regularization techniques prevent overfitting by adding penalties to model complexity.",
                    "Feature engineering involves selecting and transforming variables to improve model performance.",
                    "Data preprocessing includes cleaning, transforming, and preparing data for machine learning algorithms."
                ]
                # Repeat and shuffle to reach max_docs
                multiplier = max(1, (self.max_docs // len(sample_docs)) + 1)
                extended_docs = sample_docs * multiplier
                np.random.seed(42)  # For reproducible shuffling
                np.random.shuffle(extended_docs)
                self.documents = extended_docs[:self.max_docs]
            else:
                # Load actual dataset (example with SQuAD)
                try:
                    dataset = load_dataset(dataset_name, split='train')
                    contexts = [item['context'] for item in dataset]
                    self.documents = contexts[:self.max_docs]
                except Exception as e:
                    logger.error(f"Failed to load dataset {dataset_name}: {e}")
                    logger.info("Falling back to wikipedia_sample...")
                    self.load_documents(dataset_name="wikipedia_sample")
                    return
        else:
            raise ValueError("Either documents or dataset_name must be provided")
        
        logger.info(f"Loaded {len(self.documents)} documents")

    def build_index(self):
        """Build FAISS index for document retrieval."""
        logger.info("Building document embeddings...")
        start_time = time.perf_counter()
        
        # Generate embeddings in batches for memory efficiency
        batch_size = 64
        embeddings_list = []
        
        for i in range(0, len(self.documents), batch_size):
            batch_docs = self.documents[i:i+batch_size]
            batch_embeddings = self.embedder.encode(batch_docs, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
        
        self.document_embeddings = np.vstack(embeddings_list)
        embedding_time = time.perf_counter() - start_time
        
        logger.info(f"Generated embeddings in {embedding_time:.2f}s")
        logger.info(f"Embedding shape: {self.document_embeddings.shape}")
        
        # Build FAISS index
        start_time = time.perf_counter()
        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
        self.faiss_index.add(self.document_embeddings.astype('float32'))
        index_time = time.perf_counter() - start_time
        
        logger.info(f"Built FAISS index in {index_time:.2f}s")

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[float], float]:
        """Retrieve relevant documents for a query."""
        # Generate query embedding
        start_time = time.perf_counter()
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        embedding_time = time.perf_counter() - start_time
        
        # Search FAISS index
        search_start = time.perf_counter()
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), k
        )
        search_time = time.perf_counter() - search_start
        
        # FIXED: Handle 2D indices array from FAISS correctly
        # indices is shape (1, k) for a single query, so we take indices[0]
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        retrieved_scores = distances[0].tolist()
        
        total_retrieval_time = time.perf_counter() - start_time
        
        return retrieved_docs, retrieved_scores, total_retrieval_time

    def generate(self, query: str, context_docs: List[str], max_length: int = 128) -> Tuple[str, float]:
        """Generate response using retrieved context."""
        start_time = time.perf_counter()
        
        # Prepare input with context
        context = " ".join(context_docs[:3])  # Use top 3 docs
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        generation_time = time.perf_counter() - start_time
        return response, generation_time

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Complete RAG query pipeline."""
        start_time = time.perf_counter()
        
        # Step 1: Retrieve relevant documents
        retrieved_docs, scores, retrieval_time = self.retrieve(question, k)
        
        # Step 2: Generate response
        response, generation_time = self.generate(question, retrieved_docs)
        
        total_time = time.perf_counter() - start_time
        
        # Store metrics
        self.metrics['retrieval_times'].append(retrieval_time)
        self.metrics['generation_times'].append(generation_time)
        self.metrics['total_times'].append(total_time)
        
        return {
            'question': question,
            'response': response,
            'retrieved_docs': retrieved_docs,
            'scores': scores,
            'timing': {
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': total_time
            }
        }

    def benchmark(self, queries: List[str], output_path: str = None) -> pd.DataFrame:
        """Run benchmark on multiple queries."""
        results = []
        
        logger.info(f"Running benchmark on {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                result = self.query(query)
                results.append({
                    'query_id': i,
                    'query': query,
                    'response': result['response'],
                    'retrieval_time': result['timing']['retrieval_time'],
                    'generation_time': result['timing']['generation_time'],
                    'total_time': result['timing']['total_time'],
                    'num_retrieved_docs': len(result['retrieved_docs'])
                })
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                # Add a placeholder result to maintain indexing
                results.append({
                    'query_id': i,
                    'query': query,
                    'response': 'ERROR',
                    'retrieval_time': 0.0,
                    'generation_time': 0.0,
                    'total_time': 0.0,
                    'num_retrieved_docs': 0
                })
                continue
        
        df = pd.DataFrame(results)
        
        if output_path:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Benchmark results saved to {output_path}")
        
        return df

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary."""
        if not self.metrics['total_times']:
            return {}
        
        return {
            'avg_retrieval_time': np.mean(self.metrics['retrieval_times']),
            'avg_generation_time': np.mean(self.metrics['generation_times']),
            'avg_total_time': np.mean(self.metrics['total_times']),
            'std_total_time': np.std(self.metrics['total_times']),
            'queries_per_second': len(self.metrics['total_times']) / sum(self.metrics['total_times']),
            'total_queries': len(self.metrics['total_times'])
        }

def create_sample_queries(n_queries: int = 50, seed: int = 42) -> List[str]:
    """Create sample queries for benchmarking."""
    np.random.seed(seed)
    
    base_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is deep learning?",
        "Explain natural language processing",
        "What is computer vision?",
        "How do neural networks function?",
        "What are the applications of AI?",
        "What is supervised learning?",
        "How does reinforcement learning work?",
        "What is the difference between AI and ML?",
        "Explain transformer models",
        "What is BERT used for?",
        "How does GPT work?",
        "What are convolutional neural networks?",
        "Explain recurrent neural networks",
        "What is LSTM?",
        "How do support vector machines work?",
        "What are random forests?",
        "Explain gradient boosting",
        "What is k-means clustering?"
    ]
    
    # Generate variations
    queries = []
    for _ in range(n_queries):
        base_query = np.random.choice(base_queries)
        # Add slight variations
        variations = [
            f"Can you explain {base_query.lower()}?",
            f"Tell me about {base_query.lower()}",
            f"What do you know about {base_query.lower()}?",
            base_query,
            f"Please describe {base_query.lower()}",
            f"How would you define {base_query.lower()}?"
        ]
        queries.append(np.random.choice(variations))
    
    return queries

# Example usage and benchmark script
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = CompactRAG(max_docs=1000)
    
    # Load sample documents
    rag.load_documents(dataset_name="wikipedia_sample")
    
    # Build index
    rag.build_index()
    
    # Create sample queries
    queries = create_sample_queries(n_queries=20)
    
    # Run benchmark
    results_df = rag.benchmark(queries, output_path="data/benchmark_results.csv")
    
    # Print performance summary
    summary = rag.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nResults shape: {results_df.shape}")
    print(f"Average total time: {results_df['total_time'].mean():.4f} seconds")