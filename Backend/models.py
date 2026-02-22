import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# -----------------------------
#  LLM
# -----------------------------

llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="qwen/qwen3-32b",
    temperature=0
)

llm_image = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="qwen/qwen3-32b",
        temperature=0,
        model_kwargs={
            "response_format": {"type": "json_object"}  
        },
)

llm = OllamaLLM(                            
    model="llama3",
    temperature=0.3
)

llm1 = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    GOOGLE_API_KEY= GOOGLE_API_KEY,
    temperature = 0.6
)