from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    token = os.getenv("HUGGINGFACE_TOKEN")

    # Download FAISS index files from Hugging Face Dataset
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset", token=token)
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset", token=token)

    folder = os.path.dirname(index_file)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_qa_chain():
    vectorstore = load_vectorstore()
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa

def ask_question(question: str) -> str:
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})
    return response["result"] if isinstance(response, dict) and "result" in response else response
