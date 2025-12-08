from pydantic import BaseModel, Field
from typing import List

class Chunk(BaseModel):
    chunk_id: int = Field(..., description="Sequential chunk number")
    text: str = Field(..., description="Chunk content")
    start_offset: int = Field(..., description="Start char offset in full document")
    end_offset: int = Field(..., description="End char offset in full document")
    page_start: int = Field(..., description="First page covered by this chunk")
    page_end: int = Field(..., description="Last page covered by this chunk")
    chunk_summary: str = Field(default="", description="2-sentence summary of chunk content")

class Query(BaseModel):
    query: str = Field(..., description="User query")

class QueryVariations(BaseModel):
    variations: List[str] = Field(..., min_length=3, max_length=3, description="3 query variations")

