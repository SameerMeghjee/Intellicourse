# This script to test the API endpoints after starting the server.
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat_endpoint(query: str):
    """Test the chat endpoint with a specific query."""
    print(f"\nTesting chat endpoint with query: '{query}'")
    try:
        payload = {"query": query}
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Answer: {data['answer']}")
            print(f"Source Tool: {data['source_tool']}")
            print(f"Context Pieces: {len(data.get('retrieved_context', []))}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint."""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Stats: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_curl_examples():
    """Print curl examples for testing."""
    print("\n" + "="*60)
    print("CURL EXAMPLES FOR TESTING")
    print("="*60)
    
    print("\n1. Health check:")
    print("curl -X GET http://localhost:8000/health")
    
    print("\n2. Chat endpoint (course-related query):")
    print('curl -X POST http://localhost:8000/chat \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "What are the prerequisites for machine learning?"}\'')
    
    print("\n3. Chat endpoint (general knowledge query):")
    print('curl -X POST http://localhost:8000/chat \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "What is the job market for data scientists?"}\'')
    
    print("\n4. Stats endpoint:")
    print("curl -X GET http://localhost:8000/stats")

def run_quick_test():
    """Run quick tests of the API."""
    print("🧪 Quick API Test...")
    print("=" * 30)
    
    # Check if server is running
    try:
        response = requests.get(API_BASE_URL, timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API server is not running!")
        print("Please start the server first: python main.py")
        return False
    
    # Just 4 quick tests
    test_queries = [
        "What is CS 201?",
        "Which courses cover Python?",
        "Job market for data scientists?",
        "Prerequisites for machine learning?"
    ]
    
    results = []
    
    # Test health
    print("1. Health check...")
    health_ok = test_health_endpoint()
    results.append(health_ok)
    
    # Test queries
    for i, query in enumerate(test_queries, 2):
        print(f"\n{i}. Testing: {query}")
        success = test_chat_endpoint(query)
        results.append(success)
    
    passed = sum(results)
    total = len(results)
    print(f"\n✅ {passed}/{total} tests passed")
    
    return passed == total

def interactive_chat():
    """Interactive chat with the IntelliCourse API."""
    print("💬 IntelliCourse Interactive Chat")
    print("=" * 30)
    print("Ask questions about courses or general topics!")
    print("Type 'quit' or 'exit' to stop\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not healthy!")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server!")
        print("Make sure to run: python main.py")
        return
    
    print("✅ Connected to IntelliCourse API\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Send request to API
            print("🤔 Thinking...")
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"query": user_input},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n🤖 IntelliCourse: {data['answer']}")
                print(f"📊 Source: {data['source_tool']}")
                print("-" * 50)
            else:
                print(f"❌ Error: {response.text}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run quick test
        success = run_quick_test()
        exit(0 if success else 1)
    else:
        # Run interactive chat
        interactive_chat()