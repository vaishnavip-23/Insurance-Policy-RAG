from r2.r2_client import download_parsed_files
from chunking import chunking_markdown
doc_id = "hdfc_ergo_arogya_2024"
markdown_text, page_map = download_parsed_files(doc_id)

chunks=chunking_markdown(markdown_text,page_map)
# print(f"Total chunks: {len(chunks)}")
print(chunks[150])