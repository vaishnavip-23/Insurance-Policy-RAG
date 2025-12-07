from r2_client import upload_pdf, parse_pdf, upload_parsed_files


doc_id = "hdfc_ergo_arogya_2024"

# 1) upload raw PDF
print(f"Uploading PDF for {doc_id}...")
pdf_key = upload_pdf("../for rag.pdf", doc_id)
print(f"✓ Uploaded to: {pdf_key}")

# 2) parse from R2 → markdown + page_map in memory
print(f"\nParsing PDF {doc_id}...")
markdown_text, page_map = parse_pdf(doc_id)
print(f"✓ Parsed {len(page_map)} pages")

# 3) store parsed artifacts back in R2
print(f"\nUploading parsed files for {doc_id}...")
markdown_key, page_map_key = upload_parsed_files(doc_id, markdown_text, page_map)
print(f"✓ Markdown uploaded to: {markdown_key}")
print(f"✓ Page map uploaded to: {page_map_key}")

print(f"\n✓ Complete! All files processed for {doc_id}")

