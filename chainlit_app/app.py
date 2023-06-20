import chainlit as cl 
import fitz
from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

embedding_model = os.environ.get('EMBEDDINGS_MODEL_NAME')
persist_dir = os.environ.get('PERSIST_DIRECTORY')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llm_model_path = os.environ.get('LLM_MODEL_PATH')


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

def load_llm(iModelPath):
    llm = LlamaCpp(model_path=iModelPath, n_ctx=model_n_ctx, callbacks=[StreamingStdOutCallbackHandler()])
    # llm = OpenAI(temperature=0.2)
    return llm

def get_conversational_chain(iLLM, iEmbeddings):
    retriever = get_retriever(iEmbeddings)
    memory = ConversationBufferMemory(memory_key="chat_history",  input_key='question', output_key='answer', return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(iLLM, retriever=retriever, memory=memory, return_source_documents=True, verbose=True)
    return qa

def get_retriever(iEmbeddings):
    db = Chroma(persist_directory=persist_dir, embedding_function=iEmbeddings) #, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return retriever


@cl.langchain_factory
def main():
    pdfFile = None 

    while pdfFile == None:
        pdfFile = cl.AskFileMessage(content= "Please upload a label file", 
                                    accept=["application/pdf"] ).send()
    
        allText = ""
        with fitz.open(stream=pdfFile.content, filetype='pdf') as doc:
            for page in doc:
                allText += page.get_text("text")
        elements = [
            cl.Text(name="Text Extracted", text=allText, display="inline")
        ]

        cl.Message(content="Extracted text from pdf", elements=elements).send()


    # Split the text 
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = textSplitter.split_text(allText)
    print(f"Number of chunks: {len(chunks)}")


    # Embbed the text 
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if not are_embeddings_present(pdfFile.name, embeddings):
        generate_vectorstore(chunks, embeddings, pdfFile.name)

    # Save the metadata and texts in the user session
    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))]
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", chunks)

    # Generate a qachain
    llm = load_llm(llm_model_path)
    chain = get_conversational_chain(llm, embeddings)
    # Let the user know that the system is ready
    cl.Message(content=f"{pdfFile.name} uploaded, you can now ask questions!").send()

    return chain

@cl.langchain_postprocess
def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(text=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    cl.Message(content=answer, elements=source_elements).send()