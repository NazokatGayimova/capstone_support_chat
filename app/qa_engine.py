from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from huggingface_hub import hf_hub_download
import os

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    # Download from Hugging Face Dataset
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss")
    index_pkl = hf_hub_download(repo_id=repo_id, filename="index.pkl")

    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        folder=os.path.dirname(index_file),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

def get_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def ask_question(question: str):
    qa_chain = get_qa_chain()
    result = qa_chain({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    source_info = ""

    for doc in sources:
        metadata = doc.metadata
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        source_info += f"\n📚 {source} (page {page})"

    return f"{answer}\n{source_info}" if sources else answer

