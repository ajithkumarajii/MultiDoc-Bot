import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks for embedding
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create and save vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# Load saved vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create QA chain using Gemini model
def create_qa_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question. If the answer isn't in the context, say "I don't know".

        Context: {context}
        Question: {question}
        Answer:
        """
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user questions
def answer_question(question):
    db = load_vector_store()
    docs = db.similarity_search(question)
    chain = create_qa_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("💬 Answer:", result["output_text"])

# Streamlit UI
def main():
    st.set_page_config("Simple PDF Chatbot", page_icon="📄")
    st.title("📚 Simple Multi-PDF Chatbot 🤖")

    question = st.text_input("Ask a question about your PDF(s):")

    if question:
        answer_question(question)

    with st.sidebar:
        st.header("Upload PDFs")
        pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Reading and processing PDFs..."):
                text = get_pdf_text(pdfs)
                chunks = get_text_chunks(text)
                create_vector_store(chunks)
                st.success("PDFs processed and indexed!")

if __name__ == "__main__":
    main()
