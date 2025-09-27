from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import QueryRequest, QueryResponse
from agent_graph import IntelliCourseAgent
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the agent
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent
    logger.info("Initializing IntelliCourse Agent...")
    try:
        agent = IntelliCourseAgent()
        logger.info("Agent initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="IntelliCourse API",
    description="AI-Powered University Course Advisor",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to IntelliCourse API",
        "description": "AI-Powered University Course Advisor",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Main chat endpoint for course queries",
            "/health": "GET - Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global agent
    return {
        "status": "healthy" if agent is not None else "unhealthy",
        "agent_initialized": agent is not None
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Main chat endpoint for processing course-related queries.
    
    - **query**: The user's question about courses or general academic topics
    
    Returns an AI-generated response with source information.
    """
    global agent
    
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Agent not initialized.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process the query through the agent
        result = agent.query(request.query)
        
        # Create response
        response = QueryResponse(
            answer=result["answer"],
            source_tool=result["source_tool"],
            retrieved_context=result.get("retrieved_context", [])
        )
        
        logger.info(f"Query processed successfully. Source: {result['source_tool']}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your query: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get basic statistics about the system."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Get some basic stats from the vector store
        collection = agent.course_retriever.vectorstore._collection
        doc_count = collection.count()
        
        return {
            "total_documents": doc_count,
            "agent_status": "active",
            "available_tools": ["course_retriever", "web_search"]
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_documents": "unknown",
            "agent_status": "active",
            "available_tools": ["course_retriever", "web_search"],
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )