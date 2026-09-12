import os
import traceback
from flask import Flask, render_template, request
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# ---------------- ENV ----------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API keys missing. Check .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("PINECONE OK")
print("GROQ OK")

rag_chain = None


# ---------------- RAG SETUP ----------------
def get_rag_chain():
    global rag_chain

    if rag_chain is not None:
        return rag_chain

    try:
        print("Initializing RAG chain...")

        # Lazy imports
        from src.helper import download_hugging_face_embeddings
        from src.prompt import system_prompt

        from langchain_pinecone import PineconeVectorStore
        from langchain_groq import ChatGroq
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        print("Imports successful")

        # ---------------- EMBEDDINGS ----------------
        embeddings = download_hugging_face_embeddings()
        print("Embeddings loaded")

        # ---------------- VECTOR STORE ----------------
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        print("Pinecone vector store connected")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("Retriever ready")

       # ---------------- LLM (AUTO-DETECT ACTIVE MODEL) ----------------
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Groq se live available models ki list fetch karna
        available_models = [m.id for m in groq_client.models.list().data if not m.id.startswith("whisper")]
        print("Available Groq models:", available_models)

        # Active chat model select karna (Llama preference)
        selected_model = next((m for m in available_models if "llama" in m.lower() and "preview" not in m), available_models[0])
        print(f"Using active model: {selected_model}")

        llm = ChatGroq(
            model=selected_model,
            temperature=0,
            groq_api_key=GROQ_API_KEY
        )
        print("LLM ready")

        # ---------------- PROMPT ----------------
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Context:\n{context}\n\nQuestion:\n{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        print("RAG chain ready")
        return rag_chain

    except Exception as e:
        print("Error during initialization:", e)
        traceback.print_exc()
        raise


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()

        if not msg:
            return "Please enter a message."

        print("USER:", msg)

        chain = get_rag_chain()
        result = chain.invoke({"input": msg})

        print("RAW RESULT:", result)

        if isinstance(result, dict):
            if "answer" in result:
                return result["answer"]
            if "output_text" in result:
                return result["output_text"]

        return str(result)

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return "Backend error aaya hai, terminal check karo"


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)