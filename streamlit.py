import os
import openai
import shutil
import warnings
import pandas as pd
import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader, ListIndex, VectorStoreIndex, TreeIndex,
    KeywordTableIndex, SimpleKeywordTableIndex, DocumentSummaryIndex,
    KnowledgeGraphIndex, StorageContext, load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.memory import ChatMemoryBuffer

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Suppressing warnings
warnings.filterwarnings('ignore')

# Streamlit UI
st.title("Document Q&A System")
st.write("Upload a document and ask questions about its content.")

# Upload documents
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    DOCS_DIR = "uploaded_docs"
    INDEX_DIR = "index_dir"
    PERSIST_DIR = "persist_dir"
    
    # Ensure the directories exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Load documents
    pdf_files = [os.path.join(DOCS_DIR, filename) for filename in os.listdir(DOCS_DIR) if filename.endswith('.pdf')]
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    
    # Create index
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo", presence_penalty=-2, top_p=1)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, llm=llm)
    
    # Save the index to storage
    index.storage_context.persist(persist_dir=INDEX_DIR)
    
    # Load index from storage if it exists
    if os.path.exists(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    
    # Define query engine
    query_engine = index.as_query_engine()
    
    # Query input
    query = st.text_input("Ask a question about the document")
    
    if query:
        response = query_engine.query(query)
        st.write("Response:", response)

    # Chat engine setup
    memory = ChatMemoryBuffer.from_defaults(token_limit=80000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources, answer the query.\n"
            "If the query is unrelated to the context, just answer: I don't know.\n"
            "Always start your answer with 'Dear Student'.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
    )
    chat_engine.reset()

    st.write("Chat with the document")
    user_input = st.text_input("Your question for chat engine")
    
    if user_input:
        chat_response = chat_engine.chat_repl(user_input)
        st.write("Chat Response:", chat_response)

else:
    st.write("Please upload at least one PDF document.")