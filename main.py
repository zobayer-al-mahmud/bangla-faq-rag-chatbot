import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Bengali RAG Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    question: str

# Response model
class ChatResponse(BaseModel):
    question: str
    category: str
    answer: str

# --- Initialize Components ---
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")

# --- Data Chunks with Metadata ---



education_chunks = [
    ("শিক্ষা মানুষের জ্ঞান, দক্ষতা ও মূল্যবোধ গঠনে গুরুত্বপূর্ণ ভূমিকা রাখে এবং একটি দেশের সামাজিক ও অর্থনৈতিক উন্নয়নের ভিত্তি তৈরি করে।", {"category": "education"}),
    ("প্রাথমিক শিক্ষা শিশুদের মৌলিক পড়া, লেখা ও গণিত শেখায় যা ভবিষ্যৎ শিক্ষার জন্য ভিত্তি হিসেবে কাজ করে।", {"category": "education"}),
    ("কারিগরি ও বৃত্তিমূলক শিক্ষা শিক্ষার্থীদের বাস্তব কাজের দক্ষতা অর্জনে সহায়তা করে এবং কর্মসংস্থানের সুযোগ বাড়ায়।", {"category": "education"}),
    ("অনলাইন শিক্ষা ইন্টারনেটের মাধ্যমে ঘরে বসেই শেখার সুযোগ তৈরি করেছে, যা সময় ও খরচ কমায়।", {"category": "education"}),
    ("উচ্চশিক্ষা গবেষণা ও নতুন জ্ঞান সৃষ্টির মাধ্যমে সমাজের দীর্ঘমেয়াদি উন্নয়নে অবদান রাখে।", {"category": "education"}),
    ("শিক্ষা মানুষের যুক্তিবোধ ও সমস্যা সমাধানের ক্ষমতা বৃদ্ধি করে।", {"category": "education"}),
    ("শিক্ষার মাধ্যমে মানুষ সামাজিক দায়িত্ব ও নৈতিকতা সম্পর্কে সচেতন হয়।", {"category": "education"}),
    ("শিক্ষিত জনগোষ্ঠী একটি দেশের গণতন্ত্র ও সুশাসন শক্তিশালী করে।", {"category": "education"}),
    ("আজীবন শিক্ষা মানুষকে পরিবর্তনশীল বিশ্বের সাথে মানিয়ে নিতে সাহায্য করে।", {"category": "education"}),
    ("শিক্ষায় সমান সুযোগ না থাকলে সামাজিক বৈষম্য বৃদ্ধি পায়।", {"category": "education"})
]

health_chunks = [
    ("সুস্থ জীবনযাপনের জন্য সুষম খাদ্য, নিয়মিত ব্যায়াম এবং পর্যাপ্ত বিশ্রাম অত্যন্ত গুরুত্বপূর্ণ।", {"category": "health"}),
    ("পর্যাপ্ত পানি পান শরীরের তাপমাত্রা নিয়ন্ত্রণ করে এবং হজম ও রক্ত সঞ্চালনে সাহায্য করে।", {"category": "health"}),
    ("মানসিক স্বাস্থ্য ভালো না থাকলে কাজের দক্ষতা কমে যায়, তাই মানসিক চাপ নিয়ন্ত্রণ করা জরুরি।", {"category": "health"}),
    ("নিয়মিত স্বাস্থ্য পরীক্ষা করলে রোগ প্রাথমিক অবস্থায় শনাক্ত করা সম্ভব হয়।", {"category": "health"}),
    ("ধূমপান ও অতিরিক্ত ফাস্টফুড গ্রহণ দীর্ঘমেয়াদে হৃদরোগ ও ডায়াবেটিসের ঝুঁকি বাড়ায়।", {"category": "health"}),
    ("নিয়মিত শারীরিক ব্যায়াম হৃদযন্ত্র সুস্থ রাখতে সহায়তা করে।", {"category": "health"}),
    ("পর্যাপ্ত ঘুম শরীরের রোগ প্রতিরোধ ক্ষমতা বৃদ্ধি করে।", {"category": "health"}),
    ("মানসিক চাপ দীর্ঘদিন অব্যাহত থাকলে শারীরিক রোগের ঝুঁকি বাড়ে।", {"category": "health"}),
    ("স্বাস্থ্যকর জীবনযাপন দীর্ঘায়ু ও কর্মক্ষমতা বাড়ায়।", {"category": "health"}),
    ("স্বাস্থ্য সচেতনতা মানুষকে রোগ প্রতিরোধে আরও দায়িত্বশীল করে তোলে।", {"category": "health"})
]

travel_chunks = [
    ("ভ্রমণ মানুষকে নতুন স্থান, সংস্কৃতি ও জীবনধারা সম্পর্কে জানার সুযোগ দেয়।", {"category": "travel"}),
    ("ভ্রমণের আগে বাজেট, যাতায়াত ও থাকার ব্যবস্থা পরিকল্পনা করলে অপ্রয়োজনীয় ঝামেলা কমে।", {"category": "travel"}),
    ("পর্যটন শিল্প একটি দেশের অর্থনীতিতে বৈদেশিক মুদ্রা অর্জনে গুরুত্বপূর্ণ ভূমিকা রাখে।", {"category": "travel"}),
    ("বিদেশ ভ্রমণের জন্য সাধারণত পাসপোর্ট, ভিসা এবং ভ্রমণ বীমা প্রয়োজন হয়।", {"category": "travel"}),
    ("ভ্রমণের সময় স্থানীয় নিয়ম ও সংস্কৃতিকে সম্মান করা দায়িত্বশীল পর্যটনের অংশ।", {"category": "travel"}),
    ("ভ্রমণ মানুষের মানসিক চাপ কমাতে গুরুত্বপূর্ণ ভূমিকা রাখে।", {"category": "travel"}),
    ("ভ্রমণের মাধ্যমে নতুন ভাষা ও সংস্কৃতি সম্পর্কে জ্ঞান অর্জন করা যায়।", {"category": "travel"}),
    ("পরিকল্পিত ভ্রমণ সময় ও অর্থ উভয়ই সাশ্রয় করে।", {"category": "travel"}),
    ("দায়িত্বশীল ভ্রমণ পরিবেশ সংরক্ষণে সহায়তা করে।", {"category": "travel"}),
    ("ভ্রমণের অভিজ্ঞতা ব্যক্তির আত্মবিশ্বাস ও সিদ্ধান্ত গ্রহণ ক্ষমতা বাড়ায়।", {"category": "travel"})
]

technology_chunks = [
    ("প্রযুক্তি মানুষের দৈনন্দিন কাজকে দ্রুত, সহজ ও স্বয়ংক্রিয় করেছে।", {"category": "technology"}),
    ("ইন্টারনেট তথ্য আদান-প্রদান এবং বৈশ্বিক যোগাযোগকে সহজ করেছে।", {"category": "technology"}),
    ("কৃত্রিম বুদ্ধিমত্তা (Artificial Intelligence বা AI) হলো এমন একটি প্রযুক্তি, যার মাধ্যমে কম্পিউটার বা মেশিন মানুষের মতো চিন্তা করা, শেখা, সিদ্ধান্ত নেওয়া এবং সমস্যা সমাধান করতে পারে।", {"category": "technology"}),
    ("ক্লাউড কম্পিউটিং ব্যবহারকারীদের অনলাইনে ডেটা সংরক্ষণ ও অ্যাক্সেস করার সুবিধা দেয়।", {"category": "technology"}),
    ("সাইবার নিরাপত্তা ডিজিটাল তথ্য ও ব্যক্তিগত ডেটা সুরক্ষার জন্য অত্যন্ত গুরুত্বপূর্ণ।", {"category": "technology"}),
    ("প্রযুক্তি শিক্ষা, স্বাস্থ্য ও ব্যবসা খাতে নতুন সম্ভাবনা সৃষ্টি করেছে।", {"category": "technology"}),
    ("ডিজিটাল প্রযুক্তি তথ্য সংরক্ষণ ও বিশ্লেষণকে সহজ করেছে।", {"category": "technology"}),
    ("স্বয়ংক্রিয় প্রযুক্তি মানুষের সময় ও শ্রম সাশ্রয় করে।", {"category": "technology"}),
    ("প্রযুক্তির অপব্যবহার ব্যক্তিগত গোপনীয়তার ঝুঁকি বাড়াতে পারে।", {"category": "technology"}),
    ("প্রযুক্তিগত দক্ষতা আধুনিক কর্মজীবনের জন্য অত্যন্ত গুরুত্বপূর্ণ।", {"category": "technology"})
]

sports_chunks = [
    ("খেলাধুলা শরীরের শক্তি, সহনশীলতা ও সমন্বয় ক্ষমতা বৃদ্ধি করে।", {"category": "sports"}),
    ("নিয়মিত খেলাধুলা মানসিক চাপ কমাতে এবং আত্মবিশ্বাস বাড়াতে সাহায্য করে।", {"category": "sports"}),
    ("দলগত খেলাধুলা নেতৃত্ব, শৃঙ্খলা ও দলগত কাজের মানসিকতা তৈরি করে।", {"category": "sports"}),
    ("ক্রিকেট ও ফুটবল বাংলাদেশে সবচেয়ে জনপ্রিয় খেলাগুলোর মধ্যে অন্যতম।", {"category": "sports"}),
    ("স্কুল ও কলেজ পর্যায়ে খেলাধুলা শিক্ষার্থীদের শারীরিক ও মানসিক বিকাশে সহায়ক।", {"category": "sports"}),
    ("খেলাধুলা শরীরের রোগ প্রতিরোধ ক্ষমতা বৃদ্ধি করতে সাহায্য করে।", {"category": "sports"}),
    ("নিয়মিত খেলাধুলা স্থূলতা ও হৃদরোগের ঝুঁকি কমায়।", {"category": "sports"}),
    ("খেলাধুলা শৃঙ্খলা ও আত্মনিয়ন্ত্রণ শেখায়।", {"category": "sports"}),
    ("খেলাধুলার মাধ্যমে শিশুদের সামাজিক দক্ষতা গড়ে ওঠে।", {"category": "sports"}),
    ("পেশাদার খেলাধুলা দেশের জন্য আন্তর্জাতিক সুনাম বয়ে আনতে পারে।", {"category": "sports"})
]


# Combine all chunks
all_chunks = education_chunks + health_chunks + travel_chunks + technology_chunks + sports_chunks

# Create documents
documents = [Document(page_content=text, metadata=meta) for text, meta in all_chunks]

print("Creating vector store...")
vector_store = FAISS.from_documents(documents, embedding_model)

# Setup OpenAI client with environment variable
token = os.getenv('GITHUB_TOKEN')
if not token:
    raise ValueError("GITHUB_TOKEN not found in environment variables. Please check your .env file.")

endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

print("Server initialization complete!")

# --- Helper Functions ---
def filter_by_metadata(query, category):
    """
    Filter vector store by category metadata first, then perform similarity search.
    """
    # Try to filter by category first
    filtered_docs = [doc for doc in documents if doc.metadata['category'] == category]
    
    if filtered_docs:
        temp_vector_store = FAISS.from_documents(filtered_docs, embedding_model)
        return temp_vector_store.similarity_search(query, k=3)
    
    # Fallback: search all documents if no category match
    return vector_store.similarity_search(query, k=3)

def detect_category_llm(question):
    """
    Use LLM to determine appropriate category
    """
    system_msg = """তুমি একটি শ্রেণিবিন্যাসকারী এজেন্ট। নিচের প্রশ্নটি পড়ে বলো এটি কোন ক্যাটাগরির মধ্যে পড়ে: education, health, travel, technology, sports। 

শুধুমাত্র এই পাঁচটির মধ্যে একটি শব্দ উত্তর দাও (অন্য কিছু লিখো না):
- education (শিক্ষা, জ্ঞান, স্কুল, কলেজ, পড়াশোনা সম্পর্কিত)
- health (স্বাস্থ্য, খাদ্য, ব্যায়াম সম্পর্কিত)
- travel (ভ্রমণ, পর্যটন সম্পর্কিত)
- technology (প্রযুক্তি, AI, ইন্টারনেট সম্পর্কিত)
- sports (খেলাধুলা সম্পর্কিত)"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question}
            ],
            model=model,
            temperature=0,
            top_p=1.0
        )
        detected = response.choices[0].message.content.strip().lower()
        
        # Ensure we only return valid categories
        valid_categories = ["education", "health", "travel", "technology", "sports"]
        return detected if detected in valid_categories else "education"
    except Exception as e:
        return "education"

def ask_faq_bot(user_question: str, category: str):
    """
    Main RAG function to answer questions - ONLY from provided chunks
    """
    docs = filter_by_metadata(user_question, category)
    
    if not docs:
        return "দুঃখিত, এই বিষয়ে আমার জ্ঞানভাণ্ডারে কোনো তথ্য নেই।"
    
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context.strip():
        return "দুঃখিত, এই বিষয়ে আমার জ্ঞানভাণ্ডারে কোনো তথ্য নেই।"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""তুমি একজন কঠোর নিয়ম মেনে চলা বাংলা সহকারী। তোমার একমাত্র কাজ হলো নিচের দেওয়া তথ্য থেকে উত্তর দেওয়া।

**গুরুত্বপূর্ণ নিয়ম:**
1. শুধুমাত্র নিচের প্রদত্ত তথ্য ব্যবহার করে উত্তর দাও
2. তোমার নিজের জ্ঞান বা সাধারণ তথ্য কখনোই ব্যবহার করবে না
3. প্রদত্ত তথ্যের বাইরে যেকোনো প্রশ্নের উত্তর দিও: "দুঃখিত, এই বিষয়ে আমার জ্ঞানভাণ্ডারে কোনো তথ্য নেই।"
4. যদি প্রশ্নের সরাসরি উত্তর প্রদত্ত তথ্যে না থাকে তাহলে অনুমান করো না
5. তোমার কাছে শুধু নিচের তথ্যই আছে, এর বাইরে কিছু নেই

**প্রদত্ত তথ্য:**
{context}

মনে রাখো: এই তথ্যের বাইরে তুমি কোনো উত্তর দিতে পারবে না।""",
                },
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
            temperature=0.1,
            top_p=0.7,
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"দুঃখিত, একটি ত্রুটি ঘটেছে: {str(e)}"

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bengali RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that accepts a Bengali question and returns an answer
    
    Example request:
    {
        "question": "শিক্ষা একটি দেশের উন্নয়নে কেন গুরুত্বপূর্ণ?"
    }
    """
    if not request.question or request.question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        category = detect_category_llm(request.question)
        answer = ask_faq_bot(request.question, category)
        
        return ChatResponse(
            question=request.question,
            category=category,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
