from pydantic import BaseModel, Field
from typing import List, Dict

class Chunk(BaseModel):
    chunk_id: int = Field(..., description="Sequential chunk number")
    text: str = Field(..., description="Chunk content")
    start_offset: int = Field(..., description="Start char offset in full document")
    end_offset: int = Field(..., description="End char offset in full document")
    page_start: int = Field(..., description="First page covered by this chunk")
    page_end: int = Field(..., description="Last page covered by this chunk")
    chunk_summary: str = Field(default="", description="2-sentence summary of chunk content")

class InputQuery(BaseModel):
    query: str = Field(..., description="User query")

class QueryVariations(BaseModel):
    variations: List[str] = Field(..., min_length=3, max_length=3, description="3 query variations")

class FinalQueries(BaseModel):
    original_query: str = Field(..., description="Original user query")
    variations: List[str] = Field(..., min_length=3, max_length=3, description="3 query variations")

class RetrievalChunk(BaseModel): #Individual chunk result from dense/sparse
    chunk_id: int = Field(..., description="Chunk ID")
    text: str = Field(..., description="Full chunk text")
    similarity_score: float = Field(..., description="Similarity/BM25 score")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")

class QueryRetrievalResult(BaseModel): #Result for ONE question using RetrievalChunk
    question: str = Field(..., description="Query question text")
    chunks: List[RetrievalChunk] = Field(..., min_length=5, max_length=5, description="Top 5 retrieved chunks for this query")

class DenseRetrievalResults(BaseModel): #Results for ALL 4 questions (1 original + 3 variations)
    results: List[QueryRetrievalResult] = Field(..., min_length=4, max_length=4, description="Results for all 4 queries (1 original + 3 variations)")

class SparseRetrievalResults(BaseModel): #Results for ALL 4 questions using BM25
    results: List[QueryRetrievalResult] = Field(..., min_length=4, max_length=4, description="Results for all 4 queries (1 original + 3 variations)")

class RankedChunk(BaseModel): #Final ranked chunk after RRF
    chunk_id: int = Field(..., description="Chunk ID")
    text: str = Field(..., description="Full chunk text")
    chunk_summary: str = Field(..., description="Chunk summary")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score")
    appearances: int = Field(..., description="Number of times chunk appeared in results")
    sources: List[str] = Field(..., description="Retrieval sources: dense, sparse, or both")

class FinalRankedResults(BaseModel): #Final results after RRF merge and deduplication
    chunks: List[RankedChunk] = Field(..., description="Top ranked chunks after RRF")
    total_before_dedup: int = Field(..., description="Total chunks before deduplication")
    total_after_dedup: int = Field(..., description="Total unique chunks after deduplication")

class Citation(BaseModel): #Individual citation reference
    chunk_id: int = Field(..., description="Chunk ID used for this citation")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")

class Answer(BaseModel): #Final answer with citations
    answer: str = Field(..., description="Comprehensive answer to the user's query based on retrieved chunks")
    citations: List[Citation] = Field(..., description="Citations used in the answer")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")