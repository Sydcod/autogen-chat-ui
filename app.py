import streamlit as st
import os
import json
import autogen
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


# Assuming the environment variable for OpenAI API key is already set up externally
# os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY_HERE'

def load_and_process_pdf(file_path):
    loaders = [PyPDFLoader(file_path)]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(docs)

    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings()
    )
    vectorstore.add_documents(docs)
    return vectorstore

def setup_qa(vectorstore):
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0),
        vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    return qa

st.title('Financial Statement Analysis Tool')

uploaded_file = st.file_uploader("Upload your financial statement PDF", type="pdf")
if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file to process
    file_path = "/tmp/uploaded_financial_statement.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load and process the PDF
    vectorstore = load_and_process_pdf(file_path)
    qa_chain = setup_qa(vectorstore)

    # Input for user questions
    user_question = st.text_input("Type your question about the financial statement here:")
    if user_question:
        # Generate response using the conversational retrieval chain
        response = qa_chain({"question": user_question})
        st.write("Answer:", response['answer'])
else:
    st.write("Please upload a PDF file to proceed.")

          

