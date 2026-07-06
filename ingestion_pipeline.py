import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class EmbeddingFunction:
    """Wrapper class for SentenceTransformer to work with Chroma"""
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        """Embed a single query"""
        return self.model.encode([text]).tolist()[0]

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load() # gives a list of langchain documents

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    # for i, doc in enumerate(documents[:2]): #Show first 2 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f" Sorce: {doc.metadata['source']}")
    #     print(f" Content preview: {doc.page_content[:100]}...")
    #     print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks"""
    print(f"\nSplitting documents into chunks...")
    
    # Debug: show document sizes
    for i, doc in enumerate(documents):
        doc_size = len(doc.page_content)
        print(f"Document {i+1} ({doc.metadata['source']}): {doc_size} characters")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Debug: show chunks per document
    print(f"\nTotal chunks created: {len(chunks)}")
    from collections import Counter
    chunk_sources = Counter([chunk.metadata['source'] for chunk in chunks])
    for source, count in chunk_sources.items():
        print(f"  {source}: {count} chunks")

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = EmbeddingFunction(embedding_model)
        
    #Create CromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():   
    print("Main function")

   # 1. Loading the files
    documents = load_documents(docs_path="docs")

   # 2. Splitting the documents into chunks
    chunks = split_documents(documents)

   # 3. Creating embeddings and storing in vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()