# IntelliCourse - AI-Powered University Course Advisor

An intelligent REST API-powered assistant that helps students navigate university course catalogs using advanced AI technologies including Retrieval-Augmented Generation (RAG) and multi-tool agents.

## 🚀 Features

- **Intelligent Query Routing**: Automatically determines whether to search course catalogs or the web
- **RAG-Powered Course Search**: Retrieves relevant information from university course catalogs
- **Web Search Integration**: Answers general knowledge questions about education and careers
- **FastAPI REST API**: Clean, well-documented API endpoints
- **Multi-Tool Agent**: Uses LangGraph for intelligent decision-making

## 🛠️ Technology Stack

- **API Framework**: FastAPI
- **LLM Orchestration**: LangChain & LangGraph
- **Language Model**: Google Gemini-2.5-Pro
- **Embeddings**: Hugging Face `all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB
- **Web Search**: Tavily Search API
- **Document Processing**: PyPDF for course catalog processing

## 📋 Prerequisites

- Python 3.8 or higher
- Google API key (for Gemini Pro)
- Tavily API key (for web search)

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone www.github.com/SameerMeghjee/IntelliCourse
cd intellicourse
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create file named `.env` and add your API keys:

Edit the `.env` file:
```env
# Google Gemini API Key (get from https://makersuite.google.com/app/apikey)
GOOGLE_API_KEY=your_google_api_key_here

# Tavily Search API Key (get from https://tavily.com/)
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Initialize the System
Run the setup script to initialize the vector database:

```bash
python setup.py
```

This script will:
- Validate your environment variables
- Create necessary directories
- Initialize the vector database with sample course data
- Test all system components

### 5. Add Course Catalogs (Optional)
Place your university's PDF course catalogs in the `pdfs/` directory. The system will automatically process them.

## 🚦 Running the Application

### Start the API Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

### View API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 📡 API Usage

### Main Chat Endpoint

**POST** `/chat`

Send queries about courses or general academic topics.

#### Request Format
```json
{
  "query": "What are the prerequisites for the advanced machine learning course?"
}
```

#### Response Format
```json
{
  "answer": "The prerequisites for the advanced machine learning course (CS 301) are CS 201 (Data Structures and Algorithms) and MATH 201 (Statistics)...",
  "source_tool": "course_retriever",
  "retrieved_context": [
    "Course catalog excerpt 1",
    "Course catalog excerpt 2"
  ]
}
```

#### Example Requests

**Course-related query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the prerequisites for machine learning?"}'
```

**General knowledge query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the job market for data scientists?"}'
```

**Python requests example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "Which courses cover Python programming?"}
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Source: {data['source_tool']}")
```

### Other Endpoints

#### Health Check
**GET** `/health` - Check if the service is running

#### Statistics
**GET** `/stats` - Get system statistics

#### Root
**GET** `/` - API information and available endpoints

## 🧪 Testing

### Run Comprehensive Tests
```bash
python test_client.py
```

This will test all endpoints and provide a detailed report.

### Manual Testing
You can also test individual components:

```bash
# Test document processing
python document_processor.py

# Test RAG retriever
python rag_retriever.py

# Test web search
python web_search_tool.py

# Test agent graph
python agent_graph.py
```

## 🎯 How It Works

### 1. Query Routing
When you send a query, the system first determines whether it's:
- **Course-related**: Questions about specific courses, prerequisites, schedules
- **General knowledge**: Career advice, job market, study tips

### 2. Tool Selection
Based on the classification:
- **Course queries** → RAG retriever searches the course catalog
- **General queries** → Web search provides current information

### 3. Response Generation
The system:
1. Retrieves relevant information from the chosen source
2. Uses the LLM to generate a comprehensive, helpful answer
3. Returns the response with source attribution

## 📁 Project Structure

```
intellicourse/
├── main.py                # FastAPI application
├── models.py              # Pydantic models for API
├── config.py              # Configuration and environment variables
├── document_processor.py  # PDF processing and vector store setup
├── rag_retriever.py       # RAG pipeline for course queries
├── web_search_tool.py     # Web search tool for general queries
├── agent_graph.py         # LangGraph agent orchestration
├── setup.py               # Project initialization script
├── test_client.py         # API testing script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables template
├── pdfs/                  # Directory for course catalog PDFs
└── chroma_db/             # ChromaDB vector database (created automatically)
```

## 🔍 Sample Queries

### Course-Related Queries
- "What are the prerequisites for CS 301?"
- "Which courses cover Python programming?"
- "Tell me about bioinformatics courses"
- "What math courses are required for computer science?"
- "Show me courses that combine biology and computer science"

### General Knowledge Queries
- "What is the job market like for data scientists?"
- "Best programming languages to learn for machine learning"
- "How to prepare for technical interviews?"
- "Career opportunities in artificial intelligence"
