#!/usr/bin/env python3
"""
Setup script for IntelliCourse project.
Run this script to initialize the vector database and test the system.
"""

import os
import sys
from document_processor import initialize_vector_store

def setup_project():
    """Initialize the project by setting up the vector database."""
    print("🚀 Setting up IntelliCourse project...")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file based on .env.template and add your API keys.")
        print("Required API keys:")
        print("- GOOGLE_API_KEY (from https://makersuite.google.com/app/apikey)")
        print("- TAVILY_API_KEY (from https://tavily.com/)")
        return False
    
    print("✅ Found .env file")
    
    try:
        # Import config to validate environment variables
        import config
        print("✅ Environment variables validated")
    except Exception as e:
        print(f"❌ Error with environment variables: {e}")
        return False
    
    # Create pdfs directory if it doesn't exist
    if not os.path.exists('pdfs'):
        os.makedirs('pdfs')
        print("📁 Created 'pdfs' directory")
        print("Note: You can place your PDF course catalogs in this directory")
    
    # Initialize vector store
    print("🔧 Initializing vector database...")
    try:
        vectorstore = initialize_vector_store(pdf_folder='pdfs', force_rebuild=True)
        print("✅ Vector database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        return False
    
    # Test the system
    print("\n🧪 Testing the system...")
    try:
        from rag_retriever import CourseRetriever
        from web_search_tool import WebSearchTool
        from agent_graph import IntelliCourseAgent
        
        print("Testing course retriever...")
        retriever = CourseRetriever(vectorstore)
        test_result = retriever.get_answer("What courses are available in computer science?")
        print(f"✅ Course retriever working: {len(test_result['answer'])} chars response")
        
        print("Testing web search...")
        web_search = WebSearchTool()
        web_result = web_search.get_answer("What is artificial intelligence?")
        print(f"✅ Web search working: {len(web_result['answer'])} chars response")
        
        print("Testing agent...")
        agent = IntelliCourseAgent()
        agent_result = agent.query("Tell me about machine learning courses")
        print(f"✅ Agent working: Route taken - {agent_result.get('route_taken', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print("1. Run the API server: python main.py")
    print("2. Test the API at: http://localhost:8000")
    print("3. View API docs at: http://localhost:8000/docs")
    print("4. Use the /chat endpoint to ask questions")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
