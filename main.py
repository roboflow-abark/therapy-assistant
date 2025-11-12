import os
import re
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
app = FastAPI(
    title="Therapeutic Assistant",
    description="A POC for a therapeutic chatbot using FastAPI and OpenAI.",
    version="0.1.0"
)

# Allow all origins for CORS (for simple POC)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI Client Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # You could raise an error here, but for a POC we'll let it fail at runtime
    # raise ValueError("OPENAI_API_KEY environment variable not set.")
    
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Safety Layer ---
# Regex for emergency keywords (case-insensitive)
EMERGENCY_PATTERN = re.compile(
    r"\b(suicide|kill myself|want to die|chest pain|can't breathe|heart attack|trouble breathing|overdose|I am in danger|die)\b",
    re.IGNORECASE
)

# Canned emergency response (as a structured object)
EMERGENCY_RESPONSE = {
    "intent": "escalate",
    "summary": "Immediate emergency detected. Please seek help now.",
    "actions": [
        {
            "type": "seek-professional",
            "text": "This sounds like a serious emergency. Please contact your local emergency services (like 911 or 999) or go to the nearest emergency room immediately."
        }
    ],
    "confidence": 1.0,
    "evidence": []
}

# --- System Prompt (Prompt Engineering) ---
# This prompt is crucial. It defines the AI's persona, rules, and output structure.
SYSTEM_PROMPT = """
You are a friendly, empathetic and supportive Health Assistant. Your purpose is to provide safe, therapeutic guidance and wellness advice just like a human therapist would.
When user asks for help or shares feelings,must respond with:
- friendly and empathetic tone
- try to make them feel better
- must ask questions to know more about their feelings and situation then provide a friendly suitable response 

YOUR CORE DIRECTIVES:
1.  **Persona**: Be friendly, empathetic, calm, non-judgmental, and encouraging.
2.  **Nature**: Be friendly in nature, try to build rapport with the user by must asking questions to make them feel better.
3.  **NEVER Diagnose**: You must NEVER provide a medical diagnosis, prescribe medication, or claim to be a medical professional.
4.  **Always Refer**: Always encourage the user to consult a qualified medical or mental health professional for persistent or severe issues. You are a supplement, not a replacement.
5.  **Safety First**: You will not receive messages flagged as emergencies (they are handled by a separate system), but if the conversation turns towards serious harm, you must gently guide them to professional help.
6.  **Handle Out-of-Scope**: If the user asks about topics unrelated to mental health, wellness, or therapy (e.g., politics, celebrities, complex math), you MUST politely state that your purpose is to support their well-being and you can't help with that topic. DO NOT expose your limitations directly and DO NOT tell that you are AI.
7.  **Use Provided Context**: Base your reasoning on the chat history provided.

YOUR OUTPUT MUST BE A STRICT JSON OBJECT. Do not add any text before or after the JSON.
The JSON schema you MUST follow is:
{
  "intent": "A short category of your response. Must be one of: ['self-care', 'refer', 'escalate', 'out-of-scope']",
  "summary": "A concise, one-paragraph summary of your understanding of the user's feelings and situation OR a question to ask to know more about their feelings to advice them better.",
  "actions": [
    {
      "type": "The type of action. Must be one of: ['self-care', 'seek-professional', 'information']",
      "text": "A concrete, actionable suggestion. (e.g., 'Try a 5-minute guided breathing exercise.', 'Consider scheduling an appointment with a therapist to discuss these feelings.')"
    }
  ],
  "confidence": "A float (0.0 to 1.0) representing your confidence in the appropriateness of your guidance. This should be lower for more complex or vague issues.",
  "evidence": [
    {
      "title": "A descriptive title for a resource (e.g., 'Benefits of Mindfulness', 'Understanding Anxiety').",
      "source": "Only 1 source of  information from: 'WHO', 'NHS', 'APA'",
      "link": "Only 1 link to the source from: WHO : "https://www.who.int/home/search-results?indexCatalogue=genericsearchindex1&q=how%20to%20feel%20happy&wordsMode=AnyWord#gsc.tab=0&gsc.q=how%20to%20feel%20happy&gsc.page=1" OR NHS: "https://www.nhs.uk/search/results?q=How%20to%20feel%20happy" OR APA: "https://www.apa.org/search?osQuery=Mental%20Health". Use real URLs."
    }
  ]
}
"""

# --- Pydantic Models (Data Validation) ---
# These models match the JSON schema in the prompt

class ActionItem(BaseModel):
    type: Literal['self-care', 'seek-professional', 'information']
    text: str

class EvidenceItem(BaseModel):
    title: str
    source: str
    link: str

class ChatResponse(BaseModel):
    intent: Literal['self-care', 'refer', 'escalate', 'out-of-scope']
    summary: str
    actions: List[ActionItem]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] # History of {"role": "user", "content": "..."}

# --- API Endpoints ---

# @app.get("/", response_class=HTMLResponse)
# async def get_root(request: Request):
#     """Serves the main HTML chat interface."""
#     try:
#         with open("index.html", "r", encoding="utf-8") as f:
#             return HTMLResponse(content=f.read())
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="index.html not found.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handles the main chat logic."""
    
    # 1. --- SAFETY LAYER ---
    # Check for emergency keywords *before* sending to LLM
    if EMERGENCY_PATTERN.search(request.message):
        return JSONResponse(content=EMERGENCY_RESPONSE)

    # 2. --- Construct Prompt ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    # Add history
    messages.extend(request.history)
    # Add current user message
    messages.append({"role": "user", "content": request.message})
    
    # 3. --- Call OpenAI API ---
    if not client.api_key:
         raise HTTPException(
             status_code=500,
             detail="OPENAI_API_KEY not configured on the server. Please set the environment variable."
         )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Use a modern, capable model
            messages=messages,
            response_format={"type": "json_object"},  # Enable JSON mode
            temperature=0.7,  # Balance creativity and consistency
            max_tokens=1024
        )
        
        response_content = response.choices[0].message.content
        
        # 4. --- Parse and Validate Response ---
        if response_content:
            try:
                # Parse the JSON string from the LLM
                response_data = json.loads(response_content)
                
                # Validate it against our Pydantic model
                # This ensures the LLM followed instructions
                validated_response = ChatResponse(**response_data)
                
                return validated_response
            
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
            except Exception as e:
                # Catches Pydantic validation errors
                raise HTTPException(status_code=500, detail=f"LLM response validation failed: {e}")
        else:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")

    except Exception as e:
        # Handle OpenAI API errors or other issues
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Run the App ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server...")
#     print("To use the app, set your OPENAI_API_KEY environment variable.")
#     print("Example: export OPENAI_API_KEY='your_key_here'")
#     print("Then run: python main.py")
#     print("Access the app at http://localhost:8000")

#     uvicorn.run(app)
