from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download
import pickle
import faiss

# 🔥 Load FAISS vectorstore from Hugging Face dataset
def load_vectorstore():
    repo_id = "Nazokatgmva/volkswagen-support-data"  # Your uploaded dataset

    # Download files
    index_file = hf_hub_download(repo_id=repo_id, filename="index.faiss", repo_type="dataset")
    pkl_file = hf_hub_download(repo_id=repo_id, filename="index.pkl", repo_type="dataset")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    faiss_index = faiss.read_index(index_file)

    # Load docstore and index mapping
    with open(pkl_file, "rb") as f:
        docstore = pickle.load(f)

    # Create FAISS vectorstore object
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        index_to_docstore_id=docstore['index_to_docstore_id'],
        docstore=docstore['docstore'],
    )

    return vectorstore

# 🚀 Get QA chain ready
def get_qa_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

# 💬 Ask a question
def ask_question(question):
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})
    return response["result"]

