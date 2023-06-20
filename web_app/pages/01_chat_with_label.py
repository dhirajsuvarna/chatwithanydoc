import streamlit as st
import os
import fitz
import time
from dotenv import load_dotenv

from langchain.llms import LlamaCpp, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyMuPDFLoader #todo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

prompt_template = """
    You are a Product Label Analyzer in Medical Domain.
    You need to check if the text contained in the Product Label meets the Regulatory Standards.
    Following is the text of the Product Label - 

    {product_label_text}

    Answer the question based on the above product label - 

    {question}
"""

llm_model_path = os.environ.get('LLM_MODEL_PATH')
embedding_model = os.environ.get('EMBEDDINGS_MODEL_NAME')
persist_dir = os.environ.get('PERSIST_DIRECTORY')
model_n_ctx = os.environ.get('MODEL_N_CTX')
retrieval_doc_num = os.environ.get('RETRIEVAL_DOC_NUM')

@st.cache_resource
def load_llm(iModelPath):
    llm = LlamaCpp(model_path=iModelPath, n_ctx=model_n_ctx, callbacks=[StreamingStdOutCallbackHandler()])
    # llm = OpenAI(temperature=0.2)
    return llm


def generate_vectorstore(iChunks, iEmbeddings, iFileName):
    print(f"Generate the Embeddings...")
    metadatas = [{'sources': iFileName} for x in iChunks]
    db = Chroma.from_texts(iChunks, iEmbeddings, persist_directory=persist_dir, metadatas=metadatas) #, client_settings=CHROMA_SETTINGS)
    db.persist()    
    db = None


def are_embeddings_present(iFileName, iEmbeddings):
    print(f"dhiraj: {iFileName}")
    db = Chroma(persist_directory=persist_dir, embedding_function=iEmbeddings) #, client_settings=CHROMA_SETTINGS)
    collection = db.get()
    isEmbeddingPresent = False
    print(f"dhiraj: {collection['metadatas']}")
    for metadata in collection['metadatas']:
        if metadata['sources'] == iFileName:
            isEmbeddingPresent = True
            break
    
    print(f"Existing Embeddings Present?: {isEmbeddingPresent}")
    return isEmbeddingPresent

def generate_docs(iText):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(iText)
    print(f"Number of Chunks: {len(chunks)}")
    return chunks

def get_retriever():
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings) #, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": retrieval_doc_num})
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


def get_chain(iLLM):
    string_prompt = PromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=iLLM, prompt=string_prompt, verbose=True)
    return chain




st.title("Chat with PDF")
uploaded_file = st.file_uploader("upload file")

if uploaded_file:
    with fitz.open(stream= uploaded_file.read(), filetype='pdf') as doc:
        allText = ""
        if len(doc) > 0:
            allText = doc[0].get_text("text")

    with st.expander("Text Extracted from PDF"):
        st.info(allText)

    # with st.spinner("Embedding the Docs"):
    #     fileName = uploaded_file.name
    #     if not are_embeddings_present(fileName):
    #         # Generate Embeddings and Store it 
    #         print(f"Generating New Embeddings")
    #         chunks = generate_docs(allText)
    #         generate_vectorstore(chunks, fileName)

    with st.spinner("Loading the LLM"):
        llm = load_llm(llm_model_path)
        chain = get_chain(llm)
        # qaChain = get_qa_chain(llm)
        # qaChain = get_conversational_chain(llm)
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
            # response = qaChain({"question": query})
            response = chain.run({'product_label_text': allText, 'question': query})

            st.session_state["past"].append(query)
            # st.session_state["generated"].append(response['answer'])
            st.session_state["generated"].append(response)

            if st.session_state["generated"]:

             for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

            # for doc in response['source_documents']:
            #     st.markdown(f"### {doc.metadata['filename']}")
            #     st.info(doc.page_content)

            