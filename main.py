from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GooglePalm
from template import ai_chat, human_chat, css
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings


# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# SYSTEM_PROMPT='''
# You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# '''
load_dotenv()
my_api = os.getenv("GOOGLE_API_KEY")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def pdfread(docs):
    text = ""
    for pdf in docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    
    return text


def get_chunks(text):
    splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=250, length_function=len,separator="\n"
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectordb(chunks):
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_texts(texts=chunks,embedding=embed)
    return vectordb
    

def get_chain(vectordb):
    model = GoogleGenerativeAI(model="gemini-pro")
    # mem = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=model,
    #     memory=mem,
    #     retriever=vectordb.as_retriever(),
    # )
    prompt = hub.pull("rlm/rag-prompt")

    chain = (
    {"context": vectordb.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )
    return chain


def get_response(inp):
    response = st.session_state.chain.invoke(inp)
    st.write(human_chat.replace("{{MSG}}",inp),unsafe_allow_html=True)
    st.write(ai_chat.replace("{{MSG}}",response),unsafe_allow_html=True)

def main():
    
    st.set_page_config(page_title="Multiple PDFs chatting",page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "chain" not in st.session_state:
        st.session_state.chain = None

    st.title("Multiple PDFs chatting")
    inp = st.text_input("Ask question : ")
    if inp:
        get_response(inp)
        

    # if st.button("Ask"):
    #     st.write(get_response(inp))

    


    with st.sidebar:
        st.title("Your PDFs : ")
        docs = st.file_uploader("Upload PDF",accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Loading"):
                text = pdfread(docs)
                # st.write(text)
                chunks = get_chunks(text)
                # st.write(chunks)
                vectordb = get_vectordb(chunks)
                st.session_state.chain = get_chain(vectordb)
        



                
if __name__ == "__main__":
    main()

