# **Demo Video**

https://github.com/user-attachments/assets/a3c78ab9-91b2-4116-b583-703bcd020fcc




# Bengali RAG Chatbot

A FastAPI-based chatbot that answers questions in Bengali using Retrieval-Augmented Generation (RAG).

## Features

- **Bengali Language Support**: Handles Bengali questions and provides Bengali answers
- **Multiple Categories**: Supports education, health, travel, technology, and sports topics
- **Intelligent Routing**: Automatically detects question category using LLM
- **RAG Architecture**: Uses FAISS vector store for semantic search
- **REST API**: Simple HTTP endpoints for easy integration

## Installation

1. **Activate virtual environment**:
   ```bash
   env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Edit `.env` and add your GitHub token:
     ```
     GITHUB_TOKEN=your_github_token_here
     ```
   - Get your token from: https://github.com/settings/tokens

## Running the Server

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns server health status.

### 3. Chat Endpoint
```
POST /chat
```

**Request Body**:
```json
{
  "question": "শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?"
}
```

**Response**:
```json
{
  "question": "শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?",
  "category": "education",
  "answer": "শিক্ষা মানুষের জ্ঞান, দক্ষতা ও মূল্যবোধ গঠনে গুরুত্বপূর্ণ ভূমিকা রাখে..."
}
```

## Testing the API

### Using cURL

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?\"}"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/chat"
payload = {
    "question": "শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?"
}

response = requests.post(url, json=payload)
print(response.json())
```

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Supported Categories

- **education** (শিক্ষা): Questions about education, learning, schools
- **health** (স্বাস্থ্য): Questions about health, wellness, medical topics
- **travel** (ভ্রমণ): Questions about tourism, travel planning
- **technology** (প্রযুক্তি): Questions about technology, internet, AI
- **sports** (খেলাধুলা): Questions about sports, exercise, fitness

## Example Questions

1. **Education**: "প্রাথমিক শিক্ষা কেন গুরুত্বপূর্ণ?"
2. **Health**: "সুস্থ থাকার জন্য কী করা উচিত?"
3. **Travel**: "ভ্রমণের আগে কী পরিকল্পনা করা দরকার?"
4. **Technology**: "কৃত্রিম বুদ্ধিমত্তা কী?"
5. **Sports**: "নিয়মিত খেলাধুলার উপকারিতা কী?"

## Architecture

1. **Embedding Model**: l3cube-pune/bengali-sentence-similarity-sbert
2. **Vector Store**: FAISS for efficient similarity search
3. **LLM**: OpenAI GPT-4.1-nano (via GitHub Models)
4. **Framework**: FastAPI with async support
5. **Category Detection**: LLM-based intelligent routing

## Environment Variables

Create a `.env` file in the project root with your GitHub token:

```env
GITHUB_TOKEN=your_github_token_here
```

Get your GitHub token from: https://github.com/settings/tokens

**Important**: Never commit the `.env` file to version control. Use `.env.example` as a template.

## Notes

- First request may be slow due to model loading
- The embedding model downloads on first run (~500MB)
- Supports CORS for web browser requests
