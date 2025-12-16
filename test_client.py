"""
Test client for Bengali RAG Chatbot API
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_chat(question):
    """Test chat endpoint"""
    print(f"Question: {question}")
    print("-" * 80)
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Category: {result['category']}")
        print(f"Answer: {result['answer']}")
    else:
        print(f"Error: {response.text}")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    print("Bengali RAG Chatbot API - Test Client")
    print("=" * 80 + "\n")
    
    # Test health endpoint
    try:
        test_health()
    except Exception as e:
        print(f"Health check failed: {e}\n")
    
    # Test questions for different categories
    test_questions = [
        "শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?",
        "সুস্থ থাকার জন্য কী করা উচিত?",
        "ভ্রমণের আগে কী পরিকল্পনা করা দরকার?",
        "কৃত্রিম বুদ্ধিমত্তা কী?",
        "নিয়মিত খেলাধুলার উপকারিতা কী?",
    ]
    
    for question in test_questions:
        try:
            test_chat(question)
        except Exception as e:
            print(f"Error testing question: {e}\n")
