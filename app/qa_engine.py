import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download

def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"
    token = os.getenv("HF_TOKEN")  # optional, if needed for private repo

    # Download FAISS files
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset", token=token)
    index_pkl = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset", token=token)

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Correct unpacking here:
    docstore, index_to_docstore_id = pickle.load(open(index_pkl, "rb"))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return vectorstore
    

def get_qa_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def ask_question(question):
    qa_chain = get_qa_chain()

    response = qa_chain.invoke({"query": question})

    answer = response['result']
    return answer

