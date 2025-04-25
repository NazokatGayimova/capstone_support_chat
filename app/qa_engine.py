import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from huggingface_hub import hf_hub_download
import pickle

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Download FAISS files
    index_file = hf_hub_download(
        repo_id=repo_id,
        filename="index.faiss",
        repo_type="dataset",
        token=token
    )

    index_pkl = hf_hub_download(
        repo_id=repo_id,
        filename="index.pkl",
        repo_type="dataset",
        token=token
    )

    embeddings = OpenAIEmbeddings()

    # Load FAISS vector store
    folder = os.path.dirname(index_file)
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def ask_question(question):
    qa_chain = get_qa_chain()
    result = qa_chain(question)
    return result
