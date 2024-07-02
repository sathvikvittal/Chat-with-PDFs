from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name='gemini-1.5-flash')
chat_session = model.start_chat()



def get_response(prompt):
    response = chat_session.send_message(prompt)
    return response.text



def pdfread(docs):
    text = ""
    for pdf in docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    
    return text


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=250, length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectordb(chunks):
    embed = HuggingFaceEmbeddings(model_kwargs = {'device': 'cuda'})
    vectordb = Chroma.from_texts(texts=chunks,embedding=embed)
    return vectordb
    

def get_chain(vectordb):
    mem = ConversationBufferMemory(memory_key="history",return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_session,
        memory=mem,
        vectorstore=vectordb.as_retriever(),
    )
    return chain


def main():
    st.set_page_config(page_title="Multiple PDFs chatting",page_icon=":books:")
    st.title("Multiple PDFs chatting")
    inp = st.text_input("Ask question : ")
    if st.button("Ask"):
        st.write(get_response(inp))
    with st.sidebar:
        st.title("Your PDFs : ")
        docs = st.file_uploader("Upload PDF",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Loading"):
                text = pdfread(docs)
                # st.write(data)
                chunks = get_chunks(text)
                # st.write(chunks)
                vectordb = get_vectordb(chunks)



                
if __name__ == "__main__":
    main()

