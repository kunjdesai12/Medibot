# app.py
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# 🔹 Import helper + LangChain components
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate  # 🔸 Use simple PromptTemplate
from src.prompt import system_prompt

# ----------------------------------------
# Flask App Setup
# ----------------------------------------
app = Flask(__name__)
load_dotenv()

# ----------------------------------------
# API Keys
# ----------------------------------------
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ----------------------------------------
# Embeddings & Pinecone setup
# ----------------------------------------
embeddings = download_embeddings()
index_name = "medical-chatbot"

# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------------------
# Ollama LLM setup
# ----------------------------------------
# Make sure Ollama is running: `ollama serve`
# And model pulled: `ollama pull llama3.2`
llm = OllamaLLM(model="llama3.2")

# ----------------------------------------
# Prompt Template (Updated for RAG)
# ----------------------------------------
rag_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are MediBot, a professional AI-powered medical assistant.\n\n"
        "Use the following medical context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer clearly and concisely:"
    )
)

# ----------------------------------------
# Create RAG Chain
# ----------------------------------------
combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ----------------------------------------
# Flask Routes
# ----------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f" User: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn’t find a reliable medical explanation.")
        print(f" MediBot: {answer}")
        return str(answer)

    except Exception as e:
        print(" Error:", e)
        return "An error occurred while generating the response."

# ----------------------------------------
# Run Flask App
# ----------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
