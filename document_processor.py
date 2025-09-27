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
            print(f"Warning: PDF folder '{pdf_folder}' not found. Creating sample documents.")
            return self._create_sample_documents()
            
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in '{pdf_folder}'. Creating sample documents.")
            return self._create_sample_documents()
        
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
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample course documents for demonstration."""
        sample_courses = [
            {
                "content": """Computer Science Department - Fall 2025

CS 101 - Introduction to Computer Science
Prerequisites: None
Credits: 3
Description: Introduction to fundamental concepts of computer science including programming, algorithms, and data structures. Students will learn Python programming language and basic problem-solving techniques.

CS 201 - Data Structures and Algorithms
Prerequisites: CS 101
Credits: 4
Description: In-depth study of data structures including arrays, linked lists, stacks, queues, trees, and graphs. Analysis of algorithms and their time/space complexity.

CS 301 - Advanced Machine Learning
Prerequisites: CS 201, MATH 201 (Statistics)
Credits: 3
Description: Advanced topics in machine learning including deep learning, neural networks, and natural language processing. Students will work with Python, TensorFlow, and scikit-learn.""",
                "metadata": {"source_file": "CS_Catalog_Fall_2025.pdf", "department": "Computer Science"}
            },
            {
                "content": """Mathematics Department - Fall 2025

MATH 101 - Calculus I
Prerequisites: High School Algebra
Credits: 4
Description: Introduction to differential calculus including limits, derivatives, and applications. Foundation course for STEM majors.

MATH 201 - Statistics
Prerequisites: MATH 101
Credits: 3
Description: Introduction to statistical concepts, probability theory, hypothesis testing, and data analysis. Includes practical applications using Python and R.

MATH 301 - Linear Algebra
Prerequisites: MATH 101
Credits: 3
Description: Vector spaces, matrices, eigenvalues, eigenvectors, and linear transformations. Essential for machine learning and computer graphics.""",
                "metadata": {"source_file": "MATH_Catalog_Fall_2025.pdf", "department": "Mathematics"}
            },
            {
                "content": """Biology Department - Fall 2025

BIO 101 - General Biology
Prerequisites: None
Credits: 4
Description: Introduction to biological principles including cell structure, genetics, evolution, and ecology. Laboratory component included.

BIO 201 - Molecular Biology
Prerequisites: BIO 101, CHEM 101
Credits: 4
Description: Study of biological processes at the molecular level including DNA replication, transcription, and translation.

BIO 401 - Bioinformatics
Prerequisites: BIO 201, CS 101
Credits: 3
Description: Application of computer science techniques to biological data analysis. Combines biology with programming and data visualization using Python and R.""",
                "metadata": {"source_file": "BIO_Catalog_Fall_2025.pdf", "department": "Biology"}
            }
        ]
        
        documents = []
        for course_data in sample_courses:
            doc = Document(
                page_content=course_data["content"],
                metadata=course_data["metadata"]
            )
            documents.append(doc)
            
        print(f"Created {len(documents)} sample documents")
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
        raise ValueError("No documents were loaded. Please check your PDF files.")
    
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
        print()