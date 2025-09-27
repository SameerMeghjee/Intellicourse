from typing import Dict, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from rag_retriever import CourseRetriever
from web_search_tool import WebSearchTool
from config import GOOGLE_API_KEY, LLM_MODEL

# Define the state structure
class AgentState(TypedDict):
    query: str
    route: str
    answer: str
    source_tool: str
    retrieved_context: list
    error: str

class IntelliCourseAgent:
    def __init__(self):
        # Initialize tools
        self.course_retriever = CourseRetriever()
        self.web_search_tool = WebSearchTool()
        
        # Initialize LLM for routing
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Create routing prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query router for an AI university course advisor. 
            Your job is to classify user queries into one of two categories:

            1. "course_related" - Questions about specific courses, prerequisites, course content, 
               schedules, academic programs, degree requirements, or anything related to the 
               university's course catalog.

            2. "general_knowledge" - Questions about career advice, job market trends, general 
               educational topics, study tips, or any topic not specifically about the university's courses.

            Examples:
            - "What are the prerequisites for CS 301?" → course_related
            - "Tell me about the machine learning course" → course_related  
            - "Which courses cover Python programming?" → course_related
            - "What is the job market like for data scientists?" → general_knowledge
            - "How to prepare for technical interviews?" → general_knowledge
            - "Best programming languages to learn?" → general_knowledge

            Respond with only: "course_related" or "general_knowledge" """),
            ("human", "{query}")
        ])
        
        # Create the routing chain
        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow."""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("course_retrieval", self._course_retrieval_node)
        workflow.add_node("web_search", self._web_search_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "course_related": "course_retrieval",
                "general_knowledge": "web_search"
            }
        )
        
        # Add edges to END
        workflow.add_edge("course_retrieval", END)
        workflow.add_edge("web_search", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node to classify the query."""
        try:
            route = self.routing_chain.invoke({"query": state["query"]}).strip().lower()
            
            # Ensure valid route
            if route not in ["course_related", "general_knowledge"]:
                # Default to course_related if classification is unclear
                route = "course_related"
            
            state["route"] = route
            return state
            
        except Exception as e:
            state["route"] = "course_related"  # Default fallback
            state["error"] = f"Router error: {str(e)}"
            return state
    
    def _course_retrieval_node(self, state: AgentState) -> AgentState:
        """Course retrieval node using RAG."""
        try:
            result = self.course_retriever.get_answer(state["query"])
            state["answer"] = result["answer"]
            state["source_tool"] = result["source_tool"]
            state["retrieved_context"] = result["retrieved_context"]
            return state
            
        except Exception as e:
            state["answer"] = f"Sorry, I encountered an error while searching the course catalog: {str(e)}"
            state["source_tool"] = "course_retriever"
            state["retrieved_context"] = []
            state["error"] = str(e)
            return state
    
    def _web_search_node(self, state: AgentState) -> AgentState:
        """Web search node for general queries."""
        try:
            result = self.web_search_tool.get_answer(state["query"])
            state["answer"] = result["answer"]
            state["source_tool"] = result["source_tool"]
            state["retrieved_context"] = result["retrieved_context"]
            return state
            
        except Exception as e:
            state["answer"] = f"Sorry, I encountered an error during web search: {str(e)}"
            state["source_tool"] = "web_search"
            state["retrieved_context"] = []
            state["error"] = str(e)
            return state
    
    def _route_decision(self, state: AgentState) -> Literal["course_related", "general_knowledge"]:
        """Determine which node to route to based on classification."""
        return state["route"]
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the agent graph."""
        try:
            # Initialize state
            initial_state = AgentState(
                query=user_query,
                route="",
                answer="",
                source_tool="",
                retrieved_context=[],
                error=""
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Return formatted response
            return {
                "answer": final_state["answer"],
                "source_tool": final_state["source_tool"],
                "retrieved_context": final_state.get("retrieved_context", []),
                "route_taken": final_state.get("route", "unknown"),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an unexpected error: {str(e)}",
                "source_tool": "error",
                "retrieved_context": [],
                "route_taken": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    # Test the agent
    print("Initializing IntelliCourse Agent...")
    agent = IntelliCourseAgent()
    
    test_queries = [
        "What are the prerequisites for the advanced machine learning course?",
        "What is the job market like for data scientists?",
        "Which courses cover Python programming?",
        "Best programming languages for beginners?",
        "Tell me about bioinformatics courses",
        "How to prepare for technical interviews?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = agent.query(query)
        
        print(f"Route taken: {result['route_taken']}")
        print(f"Source tool: {result['source_tool']}")
        print(f"Answer: {result['answer']}")
        print(f"Context pieces: {len(result['retrieved_context'])}")
        if result['error']:
            print(f"Error: {result['error']}")
        print()