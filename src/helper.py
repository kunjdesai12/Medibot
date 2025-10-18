# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings import HugginffaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List
from langchain.schema import Document


#  Extract Data From the PDF files

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


# split the documents into chunks

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
       
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


#  Embedding models

from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    """Download and return the HuggingFace embeddings model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={"device": "cpu"}  # Use "cuda" if you have a compatible GPU)
    )
    return embeddings