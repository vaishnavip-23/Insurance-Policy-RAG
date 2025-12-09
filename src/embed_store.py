from typing import List
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from model.schema import Chunk

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB (local, persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="vector_store",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)


def embed_and_store(chunks: List[Chunk]):
    # Extract summaries for embedding (dense retrieval)
    summaries = [chunk.chunk_summary for chunk in chunks]
    
    # Create embeddings using OpenAI on summaries
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=summaries
    )
    embeddings = [item.embedding for item in response.data]
    
    # Prepare metadata and IDs
    ids = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(str(chunk.chunk_id)) #chroma requires ids to be strings
        metadatas.append({
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "text": chunk.text,
            "chunk_summary": chunk.chunk_summary
        })
    
    # Store in ChromaDB (embeddings are from summaries)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas  # Metadata has everything: full text, summary, citations
    )
    
    print(f"Stored {len(chunks)} chunks in vector_store (embedded summaries)")
