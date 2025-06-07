# backend.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

# Initialize LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-8b-instant"
    )
    return llm

# Load or reuse vector database
def load_or_create_vector_db():
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        loader = PyPDFLoader("static/mental_health_Document.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        return vector_db
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# Setup QA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """
    You are a compassionate mental health expert. Given a question about mental health, generate an answer using this context: {context}
    User: {question}
    Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# You no longer need the app.run() here