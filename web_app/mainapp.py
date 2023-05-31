import streamlit as st
import os
import fitz
import time
from langchain.llms import LlamaCpp, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyMuPDFLoader #todo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



MODEL_PATH = r"D:\gpt4all\models\guanaco-7B.ggmlv3.q4_1.bin"
MODEL_N_CTX=2000
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
PERSIST_DIRECTORY= "vectorstore_db"
TARGET_SOURCE_CHUNKS = 2

@st.cache_resource
def load_llm(iModelPath):
    # llm = LlamaCpp(model_path=iModelPath, n_ctx=MODEL_N_CTX, callbacks=[StreamingStdOutCallbackHandler()])
    llm = OpenAI(temperature=0.2)
    return llm


def generate_vectorstore(iChunks, iFileName):
    print(f"Generate the Embeddings...")
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    embeddings = OpenAIEmbeddings()
    metadatas = [{'filename': iFileName} for x in iChunks]
    db = Chroma.from_texts(iChunks, embeddings, persist_directory=PERSIST_DIRECTORY, metadatas=metadatas) #, client_settings=CHROMA_SETTINGS)
    db.persist()    
    db = None

def generate_docs(iText):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(iText)
    print(f"Number of Chunks: {len(chunks)}")
    return chunks

def get_retriever():
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings) #, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
    return retriever
    
def get_qa_chain(iLLM):
    retriever = get_retriever()
    qaChain = RetrievalQA.from_chain_type(llm=iLLM, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qaChain

def get_conversational_chain(iLLM):
    retriever = get_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history",  input_key='question', output_key='answer', return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(iLLM, retriever=retriever, memory=memory, return_source_documents=True, verbose=True)
    return qa



def are_embeddings_present(iFileName):
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings) #, client_settings=CHROMA_SETTINGS)
    collection = db.get()
    isEmbeddingPresent = False
    for metadata in collection['metadatas']:
        if metadata['filename'] == iFileName:
            isEmbeddingPresent = True
            break
    
    print(f"Existing Embeddings Present?: {isEmbeddingPresent}")
    return isEmbeddingPresent


st.title("Chat with PDF")
uploaded_file = st.file_uploader("upload file")

if uploaded_file:
    with fitz.open(stream= uploaded_file.read(), filetype='pdf') as doc:
        allText = ""
        for page in doc:
            allText += page.get_text("text")

    with st.expander("Text Extracted from PDF"):
        st.info(allText)

    with st.spinner("Embedding the Docs"):
        fileName = uploaded_file.name
        if not are_embeddings_present(fileName):
            # Generate Embeddings and Store it 
            print(f"Generating New Embeddings")
            chunks = generate_docs(allText)
            generate_vectorstore(chunks, fileName)

    with st.spinner("Loading the LLM"):
        llm = load_llm(MODEL_PATH)
        # qaChain = get_qa_chain(llm)
        qaChain = get_conversational_chain(llm)
        # qaChain.combine_documents_chain.verbose = True
        # qaChain.combine_documents_chain.llm_chain.verbose = True 
        # qaChain.combine_documents_chain.llm_chain.llm.verbose = True

    st.subheader("AI is ready now you can ask question related to pdf")
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    query = st.text_input("Write you query here")
    if query:
        with st.spinner("AI is thinking..."):
            # response = qaChain(query)
            # answer = response['result']
            # message(answer)

            # st.header("Sources Used")
            # for doc in response['source_documents']:
            #     st.markdown(f"### {doc.metadata['filename']}")
            #     st.info(doc.page_content)
            response = qaChain({"question": query})

            st.session_state["past"].append(query)
            st.session_state["generated"].append(response['answer'])

            if st.session_state["generated"]:

             for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

            for doc in response['source_documents']:
                st.markdown(f"### {doc.metadata['filename']}")
                st.info(doc.page_content)

            