import os
import streamlit as st
import re
from io import StringIO
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from pypdf import PdfReader
import tempfile
import time

import openpyxl
from openpyxl.styles import Font
from openpyxl.worksheet.hyperlink import Hyperlink
from datetime import datetime


# sidebar contents
# with st.sidebar:
#         st.title('DOC-QA DEMO ')
#         st.markdown('''
#         ## About        
#         This app is an LLM-powered Doc-QA Demo built using:
#         - [Streamlit](https://streamlit.io/)
#         - [LangChain](https://python.langchain.com/)
#         - [HuggingFace](https://huggingface.co/declare-lab/flan-alpaca-large)
#         ''')
#         st.write ('Made this app for testing Document Question Answering with Custom URL Data')


custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context : {context}
Question : {question}

Only returns the helpful and reasonable answer below and nothing else.
Helpful answer:
"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context',
                                                                              'question'])
    return prompt

@st.cache_resource 
def load_llm():
    n_gpu_layers = 32  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        #model_path="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_path="th-ggml-model-q4_0.bin",
        callback_manager=callback_manager,
        verbose=True,n_ctx = 4096, temperature = 0.1, max_tokens = 4096,
        n_batch=n_batch
    )
    return llm

# @st.cache_resource 
# def load_llm():
#     llm = CTransformers(model = "/home/sira/sira_project/meta-Llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
#                         model_type = "llama",
#                         max_new_tokens = 512,
#                         temperature = 0.5)
#     return llm


def check_duplicate(source_list):
    res = []
    for i in source_list:
        if i not in res:
            res.append(i)
    return res

def convert_to_website_format(urls):
    convert_urls = []
    for url in urls:
        # Remove any '.html' at the end of the URL
        url = re.sub(r'\.html$', '', url)
        # Check if the URL starts with 'www.' or 'http://'
        if not re.match(r'(www\.|http://)', url):
            url = 'https://' + url
        if '/index' in url:
            url = url.split('/index')[0]
        match = re.match(r'^([^ ]+)', url)
        if match:
            url = match.group(1)
        convert_urls.append(url)
    return convert_urls

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    convert_urls = convert_to_website_format(matchs)
    res_urls = check_duplicate(source_list=convert_urls)
    res_urls = filter_similar_url(res_urls)
    return res_urls


def filter_search(db_similarity):
    filter_list = []
    top_score = db_similarity[0][1]
    for index, score in enumerate(db_similarity) :
        if score[1] - top_score <= 0.05:
              filter_list.append(score)
    return filter_list  


@st.cache_resource 
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device': 'cpu'})
    return embeddings

@st.cache_resource
def create_vector(_pages,_embeddings):
    db = FAISS.from_documents(_pages, _embeddings)
    return db

@st.cache_data
def load_docs(pdf_file):
    st.info("`Reading doc ...`")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(pdf_file.read())
        
    loader = PyPDFLoader(temp_path)
    docs = loader.load_and_split()
    return docs

@st.cache_resource
def split_texts(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap = overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()
    return splits


def load_docs(docs_path):
    loader = DirectoryLoader(docs_path, glob="**/*.pdf")
    documents = loader.load()
    return documents

def split_docs(documents,chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs


def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING ")
    st.subheader("Retrival QA")
    #DB_FAISS_PATH = "./vectorstores_clean_doc_gte-base_no_overlap/db_faiss"
    #DB_FAISS_PATH = "/home/sira/sira_project/meta-Llama2/vectorstores_clean_doc_gte-base/db_faiss"
    # uploaded_file = st.file_uploader("Choose a file", type = ["pdf"])
    # if uploaded_file is not None:
    #     splits = load_docs(uploaded_file)
    #     #splits = split_texts(loaded_text, chunk_size=1000,overlap=0)
    documents = load_docs('paper')
    sp_docs = split_docs(documents)
    #loader = PyPDFLoader("Transformer_paper.pdf")
    #splits = loader.load_and_split()
    embeddings = load_embeddings()
    db = create_vector(sp_docs,embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt(custom_prompt_template)
    memory = ConversationBufferMemory(memory_key="chat_history", 
                                    return_messages=True, 
                                    input_key="query", 
                                    output_key="result")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {'k':3}), 
        memory = memory,
        chain_type_kwargs = {"prompt":qa_prompt}) 

    query = st.text_input("ASK ABOUT THE DOCS:")        
    if query:
        start = time.time()
        response = qa_chain({'query': query})
        st.write(response["result"])
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")



if __name__ == '__main__':
        main()