from r2.r2_client import download_parsed_files
from chunking import chunking_markdown
import asyncio
from embed_store import embed_and_store

doc_id = "hdfc_ergo_arogya_2024"
markdown_text, page_map = download_parsed_files(doc_id)

chunks = asyncio.run(chunking_markdown(markdown_text, page_map))

embed_and_store(chunks)
