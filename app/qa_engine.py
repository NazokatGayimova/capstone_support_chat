import os
from huggingface_hub import hf_hub_download
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"

    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # load token from environment
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", token=token)
    index_pkl = hf_hub_download(repo_id=repo_id, filename="index.pkl", token=token)

    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        folder=os.path.dirname(index_file),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

def get_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        streaming=True
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def ask_question(question: str):
    qa_chain = get_qa_chain()
    result = qa_chain({"query": question})
    answer = result["result"]
    return answer
