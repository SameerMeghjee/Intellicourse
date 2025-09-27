from typing import Dict, Any, List
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from config import TAVILY_API_KEY, GOOGLE_API_KEY, LLM_MODEL

class WebSearchTool:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Create prompt for web search results
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping students with general knowledge questions 
            related to education, careers, and academic topics. Use the following web search results 
            to provide a comprehensive and helpful answer.

            Be informative, accurate, and cite the sources when appropriate.
            If the search results don't contain relevant information, say so clearly.

            Web Search Results: {search_results}"""),
            ("human", "{question}")
        ])
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform web search and return formatted results."""
        try:
            # Perform search using Tavily
            search_response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            
            # Format search results
            search_results = []
            if 'results' in search_response:
                for result in search_response['results']:
                    search_results.append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': result.get('url', '')
                    })
            
            return {
                "success": True,
                "results": search_results,
                "query": query
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "query": query
            }
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """Get answer using web search."""
        try:
            # Perform web search
            search_result = self.search_web(query)
            
            if not search_result["success"]:
                return {
                    "answer": f"Sorry, I couldn't perform a web search: {search_result.get('error', 'Unknown error')}",
                    "source_tool": "web_search",
                    "retrieved_context": []
                }
            
            if not search_result["results"]:
                return {
                    "answer": "I couldn't find relevant information for your query. Please try rephrasing your question.",
                    "source_tool": "web_search",
                    "retrieved_context": []
                }
            
            # Format search results for the LLM
            formatted_results = self._format_search_results(search_result["results"])
            
            # Generate answer using LLM
            answer = self.prompt.format(
                search_results=formatted_results,
                question=query
            )
            
            response = self.llm.invoke(answer)
            
            # Extract content from search results for context
            retrieved_context = [result['content'] for result in search_result["results"] if result['content']]
            
            return {
                "answer": response.content,
                "source_tool": "web_search",
                "retrieved_context": retrieved_context
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error during web search: {str(e)}",
                "source_tool": "web_search",
                "retrieved_context": []
            }
    
    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for LLM consumption."""
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            
            formatted.append(f"Result {i}:\nTitle: {title}\nContent: {content}\nSource: {url}\n")
        
        return "\n".join(formatted)

# Create a simple runnable for the web search tool
class WebSearchRunnable(Runnable):
    def __init__(self):
        self.web_search_tool = WebSearchTool()
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        return self.web_search_tool.get_answer(query)

if __name__ == "__main__":
    # Test the web search tool
    print("Testing Web Search Tool...")
    web_search = WebSearchTool()
    
    test_queries = [
        "What is the job market like for data scientists in 2024?",
        "Best programming languages to learn for machine learning",
        "Career opportunities in bioinformatics",
        "Latest trends in computer science education"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = web_search.get_answer(query)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Context retrieved: {len(result['retrieved_context'])} sources")
        print()