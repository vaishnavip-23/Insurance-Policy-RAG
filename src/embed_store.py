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
collection = chroma_client.get_or_create_collection(name="vector_store")


def embed_and_store(chunks: List[Chunk]):
    # Extract texts from chunks
    texts = [chunk.text for chunk in chunks]
    
    # Create embeddings using OpenAI
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    
    # Prepare metadata and IDs
    ids = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(str(chunk.chunk_id))
        metadatas.append({
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "text": chunk.text,
            "chunk_summary": chunk.chunk_summary
        })
    
    # Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Stored {len(chunks)} chunks in vector_store")
