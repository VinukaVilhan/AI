import streamlit as st
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from functions.get_embeddings import get_embedding_function
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "./db"

PROMPT_TEMPLATE = """ 
Answer the question based only on the following context: {context}
----------------------------------------------------------------------
Answer the question based on above context: {question}
"""

def load_docs():
    doc_loader = PyPDFDirectoryLoader("./PDF")
    return doc_loader.load()

def split_documents(documents):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_split.split_documents(documents)

def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Add or update docs in DB
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    
    return chunks

def query_rag(query_text: str):
    embedding_func = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = Ollama(model="llama3.1:latest")
    response_text = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return formatted_response

def main():
    st.title("PDF DOC Chatter")

    # Document embedding section
    if st.button("Load and Embed Documents"):
        with st.spinner("Loading and processing documents..."):
            documents = load_docs()
            chunks = split_documents(documents)
            add_to_chroma(chunks)
            st.write(f"Documents loaded and embedded: {len(documents)}")

    # Querying section
    query_text = st.text_input("Enter your query: ")
    if st.button("Submit Query"):
        with st.spinner("Processing query..."):
            response = query_rag(query_text)
            st.write("Response:", response)

if __name__ == "__main__":
    main()
