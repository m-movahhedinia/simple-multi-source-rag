# -*- coding: utf-8 -*-
"""
Created on March 24, 2023

@auther: mansour
"""

from hashlib import sha256
from pathlib import Path

import pinecone
from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import WikipediaAPIWrapper  # GoogleSerperAPIWrapper, WolframAlphaAPIWrapper
from langchain.vectorstores import Pinecone
from streamlit import (title as ui_title, text_area as ui_text_area, text_input, write as ui_write, expander, info,
                       file_uploader, form_submit_button, sidebar, form)

# Prerequisites
# Huggingface token
huggingface_hub_token = ""
pinecone_api_key = ""
pinecone_env = ""

# To keep the document chunks and ids
document_chunks = []
chunk_ids = []

# To split the documents.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# To create the vectors.
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/llm-embedder", model_kwargs={'device': 'cpu'},
                                      encode_kwargs={'normalize_embeddings': True})

# To store the embeddings
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
pinecone_index = "test"
if pinecone_index not in pinecone.list_indexes():
    pinecone.create_index(name=pinecone_index, metric="cosine", dimension=768)

vs_db = Pinecone(index=pinecone.Index(pinecone_index), embedding=embeddings, text_key='text')

# UI
# To upload pdfs
with sidebar.form(key="File upload", clear_on_submit=True):
    files = file_uploader("Upload your document", accept_multiple_files=True, type=["pdf"],
                          help="You can choose one pdf.")
    submit = form_submit_button("Upload")

    if files and submit:
        save_loc = Path("pdfs")
        save_loc.mkdir(parents=True, exist_ok=True)
        for file in files:
            loc = save_loc.joinpath(file.name)
            with open(loc, "wb") as output:
                output.write(file.getvalue())
            document_loader = PyMuPDFLoader(loc.as_posix())
            document_chunks.extend(document_loader.load_and_split(text_splitter=splitter))

ui_title("RAG Tests")
with form(key="Question and instructions"):
    question = text_input(label="Question")
    instructions = ui_text_area(label="Instructions")
    submit = form_submit_button("Ask")

# Processing PDFs
# To write documents to the vector store
# Controlled IDs help in avoiding duplicates.
if document_chunks:
    for chunk in document_chunks:
        hashed_id = chunk.page_content
        hash_hex = sha256(hashed_id.encode('utf-8')).hexdigest()
        chunk_ids.append(hash_hex)

    vs_db.add_documents(document_chunks, ids=chunk_ids)
    document_chunks.clear()

# Prompt Templates
instructions_template = PromptTemplate(
    input_variables=["wikipedia_summary", "question"],
    template=""" Answer the question based only on the following context:
    {wikipedia_summary}
    Question: {question}
    
    """ + f"Then {instructions}" if instructions else ""
)

prompts = PromptTemplate(
    input_variables=["wikipedia_excerpt"],
    template="""Summarize the text bellow:
    {wikipedia_excerpt}
    """
)

# Model, Memory, and chains
wiki = WikipediaAPIWrapper()
llm = HuggingFaceHub(huggingfacehub_api_token=huggingface_hub_token, repo_id="google/flan-t5-xxl",
                     model_kwargs={"temperature": 0.5, "max_length": 64})

retrieval_memory = ConversationBufferMemory(input_key="query", memory_key='content_history')
document_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                 retriever=vs_db.as_retriever(search_kwargs={"k": 5}),
                                                 return_source_documents=False, memory=retrieval_memory,
                                                 output_key="result"
                                                 )

summarization_memory = ConversationBufferMemory(input_key="wikipedia_excerpt", memory_key='content_history')
summary_chain = LLMChain(llm=llm, prompt=prompts, verbose=True, output_key='wikipedia_summary',
                         memory=summarization_memory)

discourse_memory = ConversationBufferMemory(input_key="wikipedia_summary", memory_key='content_history')
discourse_chain = LLMChain(llm=llm, prompt=instructions_template, verbose=True, output_key='response',
                           memory=discourse_memory)

responser = SequentialChain(chains=[document_retriever, summary_chain, discourse_chain],  # verbose=True,
                            input_variables=["wikipedia_excerpt", "question", "query"],
                            output_variables=["result", "wikipedia_summary", "response"])

# Answer the question using Wikipedia and info from PDF if there is one
if question:
    wiki_research = wiki.run(question)
    response = responser({"wikipedia_excerpt": wiki_research, "question": question, "query": question})
    ui_write(response["response"])

    with expander('Wikipedia extraction'):
        info(response["wikipedia_excerpt"])

    with expander('Wikipedia summary'):
        info(response["wikipedia_summary"])

    with expander('Document retrieval'):
        info(response["result"])

    with expander('Chat History'):
        info(discourse_memory.buffer)
