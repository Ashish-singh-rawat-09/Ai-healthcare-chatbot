# ---------------- FIX FOR WINDOWS (pwd issue) ----------------
import sys
import types
import os
sys.modules['pwd'] = types.ModuleType('pwd')

# ---------------- IMPORTS ----------------
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings


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


# ---------------- EMBEDDINGS ----------------
def download_hugging_face_embeddings():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )
    return embeddings