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
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.ollama import Ollama
from template import ai_chat, human_chat, css


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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
    embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = Chroma.from_texts(texts=chunks,embedding=embed)
    return vectordb
    

def get_chain(vectordb):
    model = Ollama()
    mem = ConversationBufferMemory(memory_key="history",return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=mem,
        retriever=vectordb.as_retriever(),
    )
    return chain


def get_response(inp):
    response = st.session_state.chain({"question":inp})
    st.write(response)


def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs chatting",page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "chain" not in st.session_state:
        st.session_state.chain = None

    st.title("Multiple PDFs chatting")
    inp = st.text_input("Ask question : ")
    if inp:
        get_response(inp)

    if st.button("Ask"):
        st.write(get_response(inp))

    st.write(human_chat.replace("{{MSG}}","Hey"),unsafe_allow_html=True)
    st.write(ai_chat.replace("{{MSG}}","Hey"),unsafe_allow_html=True)


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
                st.session_state.chain = get_chain(vectordb)
        



                
if __name__ == "__main__":
    main()

