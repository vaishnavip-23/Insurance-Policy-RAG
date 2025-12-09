import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from model.schema import DenseRetrievalResults, SparseRetrievalResults, QueryRetrievalResult, RetrievalChunk, RankedChunk, FinalRankedResults
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
metadatas = all_chunks_data.get("metadatas", []) if all_chunks_data else []

if len(metadatas) == 0:
    print("WARNING: No chunks found in ChromaDB. Run `python src/main.py` to build the vector store.")
    bm25_corpus = []
    tokenized_corpus = []
    bm25_index = None
else:
    bm25_corpus = [metadata["text"] for metadata in metadatas]
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
    
    if bm25_index is None:
        raise RuntimeError("BM25 index is empty. Run `python src/main.py` to build the vector store before querying.")
    
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


# MERGE AND RERANK USING RRF

def merge_and_rerank(dense_results: DenseRetrievalResults, sparse_results: SparseRetrievalResults, top_k: int = 10) -> FinalRankedResults:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
    Deduplicates by chunk_id and reranks by combined RRF score.
    
    Args:
        dense_results: Results from dense retrieval (20 chunks)
        sparse_results: Results from sparse retrieval (20 chunks)
        top_k: Number of top chunks to return (default: 10)
        
    Returns:
        FinalRankedResults with deduplicated and reranked chunks
    """
    print(f"\nMerging and reranking results...")
    
    # Step 1: Flatten all results with rank information
    all_results = []
    
    # From dense retrieval
    for query_result in dense_results.results:
        for rank, chunk in enumerate(query_result.chunks):
            all_results.append({
                "chunk_id": chunk.chunk_id,
                "chunk": chunk,
                "rank": rank,
                "source": "dense"
            })
    
    # From sparse retrieval
    for query_result in sparse_results.results:
        for rank, chunk in enumerate(query_result.chunks):
            all_results.append({
                "chunk_id": chunk.chunk_id,
                "chunk": chunk,
                "rank": rank,
                "source": "sparse"
            })
    
    total_before_dedup = len(all_results)
    print(f"Total chunks before deduplication: {total_before_dedup}")
    
    # Step 2: Calculate RRF scores (deduplicates by chunk_id)
    rrf_scores = {}
    k = 60  # Standard RRF constant
    
    for result in all_results:
        chunk_id = result["chunk_id"]
        rank = result["rank"]
        source = result["source"]
        rrf_score = 1.0 / (k + rank)
        
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "chunk": result["chunk"],
                "rrf_score": 0,
                "appearances": 0,
                "sources": set()
            }
        
        rrf_scores[chunk_id]["rrf_score"] += rrf_score
        rrf_scores[chunk_id]["appearances"] += 1
        rrf_scores[chunk_id]["sources"].add(source)
    
    total_after_dedup = len(rrf_scores)
    print(f"Total unique chunks after deduplication: {total_after_dedup}")
    
    # Step 3: Sort by RRF score
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True
    )
    
    # Step 4: Convert to RankedChunk objects
    ranked_chunks = []
    for chunk_id, data in sorted_chunks[:top_k]:
        chunk = data["chunk"]
        
        # Get chunk_summary from ChromaDB metadata
        chunk_metadata = collection.get(ids=[str(chunk.chunk_id)], include=["metadatas"])
        chunk_summary = chunk_metadata["metadatas"][0]["chunk_summary"] if chunk_metadata["metadatas"] else ""
        
        # Convert sources set to sorted list
        sources_list = sorted(list(data["sources"]))
        
        ranked_chunk = RankedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            chunk_summary=chunk_summary,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            rrf_score=round(data["rrf_score"], 6),
            appearances=data["appearances"],
            sources=sources_list
        )
        ranked_chunks.append(ranked_chunk)
    
    print(f"Returning top {len(ranked_chunks)} chunks")
    
    # Return FinalRankedResults
    return FinalRankedResults(
        chunks=ranked_chunks,
        total_before_dedup=total_before_dedup,
        total_after_dedup=total_after_dedup
    )


