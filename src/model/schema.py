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

class RetrievalChunk(BaseModel): #Individual chunk result
    chunk_id: int = Field(..., description="Chunk ID")
    text: str = Field(..., description="Full chunk text")
    similarity_score: float = Field(..., description="Similarity score from vector search")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")

class QueryRetrievalResult(BaseModel): #Result for ONE question using RetrievalChunk
    question: str = Field(..., description="Query question text")
    chunks: List[RetrievalChunk] = Field(..., min_length=5, max_length=5, description="Top 5 retrieved chunks for this query")

class DenseRetrievalResults(BaseModel): #Results for ALL 4 questions (1 original + 3 variations)
    results: List[QueryRetrievalResult] = Field(..., min_length=4, max_length=4, description="Results for all 4 queries (1 original + 3 variations)")