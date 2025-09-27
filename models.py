from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the prerequisites for the advanced machine learning course?"
            }
        }

class QueryResponse(BaseModel):
    answer: str
    source_tool: str  # "course_retriever" or "web_search"
    retrieved_context: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The prerequisites for the advanced machine learning course are...",
                "source_tool": "course_retriever",
                "retrieved_context": ["Course catalog excerpt 1", "Course catalog excerpt 2"]
            }
        }