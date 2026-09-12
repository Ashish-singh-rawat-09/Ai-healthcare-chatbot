# ---------------- FIX FOR WINDOWS ----------------
import sys
import types
import os
from typing import List
from langchain_core.embeddings import Embeddings
sys.modules['pwd'] = types.ModuleType('pwd')

# ---------------- IMPORTS ----------------
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding


# ---------------- LOAD DATA ----------------
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# ---------------- SPLIT TEXT ----------------
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# ---------------- FASTEMBED EMBEDDINGS (NO HF API / NO OOM) ----------------
class LocalFastEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = list(self.model.embed([text]))[0]
        return [float(x) for x in embedding]


def download_hugging_face_embeddings():
    return LocalFastEmbeddings()