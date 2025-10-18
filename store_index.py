# ✅ Imports (fixed and modern)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# ✅ Import your helper functions (from helper.py)
from src.helper import load_pdf_files, text_split, download_embeddings

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Step 1: Load PDF documents
extracted_data = load_pdf_files(data='data/')

# ✅ Step 2: Split into chunks
text_chunks = text_split(extracted_data)

# ✅ Step 3: Create Embeddings
embeddings = download_embeddings()

# ✅ Step 4: Initialize Pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"  # your Pinecone index name

# ✅ Step 5: Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# ✅ Step 6: Connect to index
index = pc.Index(index_name)

# ✅ Step 7: Upload documents to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Pinecone index created and documents uploaded successfully!")
