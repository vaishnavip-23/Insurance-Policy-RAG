from typing import List
from model.schema import Chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunking_markdown(markdown_text,page_map)->List[Chunk]:

    chunks=[]

    return chunks