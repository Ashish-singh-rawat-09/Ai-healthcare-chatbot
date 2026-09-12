# ---------------- FIX FOR WINDOWS ----------------
import sys
import types
import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
sys.modules['pwd'] = types.ModuleType('pwd')

# ---------------- IMPORTS ----------------
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


# ---------------- DIRECT HF EMBEDDINGS (NO LIBRARY CONFLICT) ----------------
class CloudHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        res = response.json()
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
        return res


def download_hugging_face_embeddings():
    return CloudHuggingFaceEmbeddings()