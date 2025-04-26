import os
import faiss
import pickle
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Download FAISS files from Hugging Face dataset
def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download index files
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    # Load index and docstore
    faiss_index = faiss.read_index(index_file)
    with open(pkl_file, "rb") as f:
        docstore = pickle.load(f)

    # Embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vectorstore manually
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore["docstore"],
        index_to_docstore_id=docstore["index_to_docstore_id"],
    )

    return vectorstore

def get_qa_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    vectorstore = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # ✅ Important for citation
    )
    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})

    answer_text = response["result"]

    sources = []
    for doc in response["source_documents"]:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        sources.append(f"{source} (page {page})")

    if sources:
        answer_text += "\n\nSources:\n" + "\n".join(sources)

    return answer_text

