import os
import pickle
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Paths
DATA_FOLDER = "data"
OUTPUT_FOLDER = "faiss_index"

# Prepare documents
documents = []
for file_name in os.listdir(DATA_FOLDER):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file_name))
        documents.extend(loader.load())

# Split if needed
texts = [doc for doc in documents]

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(texts, embedding=embeddings)

# Save
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
faiss.write_index(vectorstore.index, os.path.join(OUTPUT_FOLDER, "index.faiss"))

with open(os.path.join(OUTPUT_FOLDER, "index.pkl"), "wb") as f:
    pickle.dump({
        "docstore": vectorstore.docstore,
        "index_to_docstore_id": vectorstore.index_to_docstore_id,
    }, f)

print("✅ FAISS index successfully rebuilt and saved to 'faiss_index/' folder with docstore and mappings.")

