import os
import pickle
import faiss
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # If needed

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download index.faiss
    index_file = hf_hub_download(
        repo_id=repo_id,
        filename="index.faiss",
        repo_type="dataset",
        token=HUGGINGFACE_TOKEN
    )

    # Download index.pkl
    pkl_file = hf_hub_download(
        repo_id=repo_id,
        filename="index.pkl",
        repo_type="dataset",
        token=HUGGINGFACE_TOKEN
    )

    # Rebuild FAISS index and docstore
    embedding_dimension = 384  # Fixed for sentence-transformers/all-MiniLM-L6-v2
    faiss_index = faiss.read_index(index_file)

    with open(pkl_file, "rb") as f:
        docstore = pickle.load(f)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS(embedding_function=embeddings, index=faiss_index, docstore=docstore)
    return vectorstore

def get_qa_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )
    vectorstore = load_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})
    return response["result"] 


