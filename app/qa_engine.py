import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load .env values
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load all PDF documents from /data
def load_documents():
    folder_path = "data"
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(pdf_path)
            loaded_pages = loader.load()
            for page in loaded_pages:
                page.metadata["source"] = f"{filename} (page {page.metadata.get('page', 0)+1})"
            docs.extend(loaded_pages)
    return docs

# Create vectorstore from documents
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Save vectorstore locally
def save_vectorstore(vectorstore, folder="faiss_index"):
    vectorstore.save_local(folder)

# Load vectorstore (enable safe pickle load)
def load_vectorstore(folder="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

# Set up LangChain Q&A chain
def get_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Ask a question and get a response with source citation
def ask_question(question: str):
    qa_chain = get_qa_chain()
    result = qa_chain(question)
    answer = result["result"]
    sources = result["source_documents"]

    citations = "\n\n📚 Sources:\n"
    for doc in sources:
        citations += f"- {doc.metadata['source']}\n"

    return answer + citations
