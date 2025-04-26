import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"

# Load all PDFs from data/
pdf_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

if not documents:
    raise ValueError("No documents found! Check your data folder.")

# Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
vectorstore = FAISS.from_documents(texts, embedding=embeddings)

# Save index
os.makedirs(INDEX_FOLDER, exist_ok=True)
vectorstore.save_local(INDEX_FOLDER)

print(f"✅ FAISS index successfully rebuilt and saved to '{INDEX_FOLDER}/' folder with {len(texts)} chunks.")

