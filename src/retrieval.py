import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from model.schema import DenseRetrievalResults, SparseRetrievalResults, QueryRetrievalResult, RetrievalChunk
from query_translate import query_translate
from typing import List, Tuple


load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="vector_store",
    metadata={"hnsw:space": "cosine"}  
)

# Initialize BM25 index once at module load (for sparse retrieval)
print("Loading all chunks for BM25 index...")
all_chunks_data = collection.get(include=["metadatas"])
bm25_corpus = [metadata["text"] for metadata in all_chunks_data["metadatas"]]
tokenized_corpus = [doc.lower().split() for doc in bm25_corpus]
bm25_index = BM25Okapi(tokenized_corpus)
print(f"BM25 index ready with {len(bm25_corpus)} chunks")


def hybrid_retrieval(query:str)->Tuple[DenseRetrievalResults,SparseRetrievalResults]:
    final_queries = query_translate(query)
    all_queries = [final_queries.original_query] + final_queries.variations
    print(f"Total queries: {len(all_queries)}")
    dense_results = dense_retrieval(all_queries)
    sparse_results = sparse_retrieval(all_queries)
    return dense_results, sparse_results


# DENSE RETRIEVAL

def dense_retrieval(all_queries: List[str]) -> DenseRetrievalResults:
    
    # Step 2: Retrieve for each query
    results = []
    
    for q in all_queries:
        print(f"Retrieving for: {q}")
        
        # Embed the query
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=q
        ).data[0].embedding
        
        # Search ChromaDB (which has summary embeddings)
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas", "distances"]
        )
        
        # Convert to RetrievalChunk objects
        chunks = []
        for i, metadata in enumerate(search_results["metadatas"][0]):
            # For cosine distance: similarity = 1 - distance (distance is 0-2, similarity is 0-1)
            similarity_score = 1 - search_results["distances"][0][i]
            
            chunk = RetrievalChunk(
                chunk_id=metadata["chunk_id"],
                text=metadata["text"],
                similarity_score=round(similarity_score, 4),
                page_start=metadata["page_start"],
                page_end=metadata["page_end"]
            )
            chunks.append(chunk)
        
        # Create QueryRetrievalResult for this query
        query_result = QueryRetrievalResult(
            question=q,
            chunks=chunks
        )
        results.append(query_result)
    
    print(f"Dense retrieval complete. Retrieved {len(results)} query results with {len(results) * 5} total chunks")
    
    # Return DenseRetrievalResults
    return DenseRetrievalResults(results=results)


# SPARSE RETRIEVAL (BM25)

def sparse_retrieval(all_queries: List[str]) -> SparseRetrievalResults:
    """Uses pre-built global BM25 index for fast keyword search."""
    
    # Step 1: Retrieve for each query (using global bm25_index)
    results = []
    
    for q in all_queries:
        print(f"BM25 retrieving for: {q}")
        
        # Tokenize query
        tokenized_query = q.lower().split()
        
        # Get BM25 scores for all documents
        scores = bm25_index.get_scores(tokenized_query)
        
        # Get top 5 indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        
        # Convert to RetrievalChunk objects
        chunks = []
        for idx in top_k_indices:
            metadata = all_chunks_data["metadatas"][idx]
            bm25_score = scores[idx]
            
            chunk = RetrievalChunk(
                chunk_id=metadata["chunk_id"],
                text=metadata["text"],
                similarity_score=round(bm25_score, 4),  # BM25 score (not 0-1 range)
                page_start=metadata["page_start"],
                page_end=metadata["page_end"]
            )
            chunks.append(chunk)
        
        # Create QueryRetrievalResult for this query
        query_result = QueryRetrievalResult(
            question=q,
            chunks=chunks
        )
        results.append(query_result)
    
    print(f"Sparse retrieval complete. Retrieved {len(results)} query results with {len(results) * 5} total chunks")
    
    # Return SparseRetrievalResults
    return SparseRetrievalResults(results=results)


