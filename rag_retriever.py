from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.schema.runnable import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from document_processor import initialize_vector_store
from config import GOOGLE_API_KEY, LLM_MODEL

class CourseRetriever:
    def __init__(self, vectorstore: Chroma = None):
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Initialize vector store if not provided
        if vectorstore is None:
            self.vectorstore = initialize_vector_store()
        else:
            self.vectorstore = vectorstore
            
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create the RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping students with university course information. 
            Use the following pieces of retrieved context to answer the question about courses, prerequisites, 
            schedules, or academic programs. Be specific and accurate.

            If the question cannot be answered based on the context provided, say so clearly.
            Always cite which course or document you're referencing when possible.

            Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Create the retrieval chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a readable context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown source')
            department = doc.metadata.get('department', 'Unknown department')
            content = doc.page_content.strip()
            formatted.append(f"Document {i} ({department} - {source}):\n{content}")
        return "\n\n".join(formatted)
    
    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve relevant context without generating an answer."""
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """Get answer using RAG pipeline."""
        try:
            # Get retrieved documents for context
            retrieved_docs = self.retriever.invoke(query)
            retrieved_context = [doc.page_content for doc in retrieved_docs]
            
            # Generate answer
            answer = self.chain.invoke(query)
            
            return {
                "answer": answer,
                "source_tool": "course_retriever",
                "retrieved_context": retrieved_context
            }
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error while searching the course catalog: {str(e)}",
                "source_tool": "course_retriever",
                "retrieved_context": []
            }

# Create a simple runnable for the retriever
class CourseRetrieverRunnable(Runnable):
    def __init__(self):
        self.retriever = CourseRetriever()
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        return self.retriever.get_answer(query)

if __name__ == "__main__":
    # Test the RAG retriever
    print("Initializing Course Retriever...")
    retriever = CourseRetriever()
    
    test_queries = [
        "What are the prerequisites for the advanced machine learning course?",
        "Which courses cover Python programming?",
        "Tell me about bioinformatics courses",
        "What math courses are available?",
        "What is CS 201 about?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = retriever.get_answer(query)
        print(f"Answer: {result['answer']}")
        print(f"Context retrieved: {len(result['retrieved_context'])} chunks")
        print()