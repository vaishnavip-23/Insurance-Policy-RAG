from typing import List
from model.schema import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunking_markdown(markdown_text,page_map)->List[Chunk]:

    chunks=[]

    return chunks