from typing import List
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from model.schema import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize OpenAI async client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def chunking_markdown(markdown_text, page_map) -> List[Chunk]:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Use create_documents with add_start_index=True to get offsets
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n"],
        add_start_index=True
    )
    
    # create_documents returns Document objects with metadata['start_index']
    doc_objs = splitter.create_documents([markdown_text])
    
    # Extract texts for summary generation
    texts = [doc.page_content for doc in doc_objs]
    
    # Generate summaries for all texts in parallel
    summaries = await generate_summaries(texts)
    
    chunks = []
    
    for chunk_id, (doc, summary) in enumerate(zip(doc_objs, summaries), start=1):
        chunk_text = doc.page_content
        start_offset = doc.metadata.get("start_index", 0)
        end_offset = start_offset + len(chunk_text) - 1
        
        # Find page_start: which page contains start_offset
        page_start = None
        for page_idx, page_info in page_map.items():
            if page_info["start_offset"] <= start_offset <= page_info["end_offset"]:
                page_start = page_info["page"]
                break
        
        # Find page_end: which page contains end_offset
        page_end = None
        for page_idx, page_info in page_map.items():
            if page_info["start_offset"] <= end_offset <= page_info["end_offset"]:
                page_end = page_info["page"]
                break
        
        # Fallback if not found
        if page_start is None:
            page_start = page_map["0"]["page"]
        
        if page_end is None:
            # Find the last page_map entry
            last_key = "0"
            for page_idx in page_map.keys():
                if int(page_idx) > int(last_key):
                    last_key = page_idx
            page_end = page_map[last_key]["page"]
        
        # Create Chunk object with summary
        chunk = Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_offset=start_offset,
            end_offset=end_offset,
            page_start=page_start,
            page_end=page_end,
            chunk_summary=summary
        )
        chunks.append(chunk)
    
    return chunks


async def summarize_single_text(text: str) -> str:
    """Generate summary for a single text using OpenAI Responses API."""
    prompt = "Write 2 sentences summarizing the main information in this text that would help answer user questions."
    input_text = f"{prompt}\n\nText: {text}"
    
    response = await openai_client.responses.create(
        model="gpt-5-mini",
        input=input_text
    )
    
    return response.output_text.strip()


async def generate_summaries(texts: List[str]) -> List[str]:
    """Generate summaries for all texts in parallel using async processing."""
    print(f"Generating summaries for {len(texts)} chunks...")
    
    # Create async tasks for all texts
    tasks = []
    for text in texts:
        task = summarize_single_text(text)
        tasks.append(task)
    
    # Run all tasks in parallel
    summaries = await asyncio.gather(*tasks)
    
    print(f"Summaries generated for {len(summaries)} chunks")
    return summaries