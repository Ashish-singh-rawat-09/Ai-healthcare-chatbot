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


# ---------------- DIRECT HF EMBEDDINGS (FEATURE-EXTRACTION FIX) ----------------
class CloudHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Explicit feature-extraction endpoint
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        } if self.token else {"Content-Type": "application/json"}

    def _call_api(self, text_or_texts):
        # HuggingFace expects parameters or raw list for feature-extraction
        payload = {
            "inputs": text_or_texts,
            "options": {"wait_for_model": True, "use_cache": True}
        }
        resp = requests.post(self.api_url, headers=self.headers, json=payload)
        res = resp.json()
        
        # If model returned error, log and raise
        if isinstance(res, dict) and "error" in res:
            # Fallback URL format if router needs direct pipeline
            direct_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            resp = requests.post(direct_url, headers=self.headers, json=payload)
            res = resp.json()
            
        return res

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self._call_api(texts)
        if isinstance(res, list):
            # Res shape is either [num_texts, 384] or [num_texts, seq_len, 384]
            results = []
            for item in res:
                while isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                    item = item[0]
                results.append([float(x) for x in item])
            return results
        raise ValueError(f"HF embed_documents error: {res}")

    def embed_query(self, text: str) -> List[float]:
        res = self._call_api([text])
        if isinstance(res, list) and len(res) > 0:
            vec = res[0]
            while isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
                vec = vec[0]
            return [float(x) for x in vec]
        raise ValueError(f"HF embed_query error: {res}")


def download_hugging_face_embeddings():
    return CloudHuggingFaceEmbeddings()