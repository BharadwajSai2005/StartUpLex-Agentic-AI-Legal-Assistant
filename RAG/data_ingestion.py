import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    db_dir = os.path.join(base_dir, "legal_faiss_db")

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            print(f"Loading {file}...")
            loader = PyMuPDFLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("No PDF documents found in data folder.")
        return

    print(f"Loaded {len(documents)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks. Generating embeddings and saving to FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create new FAISS db or load existing to add
    if os.path.exists(db_dir):
        vectorstore = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
    vectorstore.save_local(db_dir)
    print("Ingestion complete! FAISS database updated.")

if __name__ == "__main__":
    ingest_data()
