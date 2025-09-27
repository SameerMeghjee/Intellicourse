import os
import chromadb
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_and_process_pdfs(self, pdf_folder: str) -> List[Document]:
        """Load and process all PDF files in the specified folder."""
        documents = []
        
        if not os.path.exists(pdf_folder):
            raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found. Please create the folder and add PDF files.")
            
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{pdf_folder}'. Please add course catalog PDF files.")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing: {pdf_file}")
            
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata['source_file'] = pdf_file
                    page.metadata['department'] = self._extract_department(pdf_file)
                
                documents.extend(pages)
                print(f"Loaded {len(pages)} pages from {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        return documents
    
    def _extract_department(self, filename: str) -> str:
        """Extract department name from filename."""
        if 'CS' in filename.upper():
            return 'Computer Science'
        elif 'MATH' in filename.upper():
            return 'Mathematics'
        elif 'BIO' in filename.upper():
            return 'Biology'
        elif 'BUS' in filename.upper():
            return 'Business'
        else:
            return 'Unknown'
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and populate vector store."""
        # Create chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        
        print(f"Created vector store with {len(chunks)} chunks")
        return vectorstore
    
    def load_existing_vector_store(self) -> Chroma:
        """Load existing vector store."""
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        return vectorstore

def initialize_vector_store(pdf_folder: str = "pdfs", force_rebuild: bool = False):
    """Initialize or load the vector store."""
    processor = DocumentProcessor()
    
    # Check if vector store already exists
    if os.path.exists(CHROMA_PERSIST_DIR) and not force_rebuild:
        print("Loading existing vector store...")
        try:
            vectorstore = processor.load_existing_vector_store()
            print("Successfully loaded existing vector store")
            return vectorstore
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Rebuilding vector store...")
    
    # Create new vector store
    print("Creating new vector store...")
    documents = processor.load_and_process_pdfs(pdf_folder)
    
    if not documents:
        raise ValueError("No documents were loaded. Please check your PDF files and ensure the 'pdfs' folder contains valid course catalog PDFs.")
    
    vectorstore = processor.create_vector_store(documents)
    print("Vector store created successfully!")
    return vectorstore

if __name__ == "__main__":
    # Test the document processor
    print("Initializing vector store...")
    vectorstore = initialize_vector_store(force_rebuild=True)
    
    # Test retrieval
    query = "What are the prerequisites for machine learning?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nTest query: {query}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.page_content[:200]}...")
        print(f"   Source: {result.metadata.get('source_file', 'Unknown')}")
        print(f"   Department: {result.metadata.get('department', 'Unknown')}")
