from typing import List
from model.schema import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunking_markdown(markdown_text, page_map) -> List[Chunk]:
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
    
    chunks = []
    
    for chunk_id, doc in enumerate(doc_objs, start=1):
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
        
        # Create Chunk object
        chunk = Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_offset=start_offset,
            end_offset=end_offset,
            page_start=page_start,
            page_end=page_end
        )
        chunks.append(chunk)
    
    return chunks