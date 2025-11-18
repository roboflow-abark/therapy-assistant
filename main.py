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
    description="A POC for a therapeutic chat usecases.",
    version="1.0.0"
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
    
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Safety Layer ---
# Regex for emergency keywords (case-insensitive)
EMERGENCY_PATTERN = re.compile(
    r"\b(suicide|kill myself|want to die|chest pain|can't breathe|cant breathe|heart attack|trouble breathing|overdose|I am in danger|im in danger|die)\b",
    re.IGNORECASE
)

# Canned emergency response (as a structured object)
EMERGENCY_RESPONSE = {
    "intent": "escalate",
    "summary": "It sounds like you might be in immediate danger or experiencing a medical emergency. Please seek help right now.",
    "actions": [
        {
            "type": "seek-professional",
            "text": "Please contact your local emergency services immediately (for example, 911 or 999), or go to the nearest emergency room. If possible, also reach out to a trusted person near you."
        }
    ],
    "confidence": 1.0,
    "evidence": []
}

# --- System Prompt (Prompt Engineering) ---
# Defines persona, rules, and base JSON schema. Conversation stage (exploration/guidance)
# is controlled by an extra system message added dynamically in /chat.
SYSTEM_PROMPT = """
You are a friendly, empathetic and supportive Health Assistant. Your purpose is to provide safe, therapeutic guidance and wellness support just like a human therapist would in a conversation.

GENERAL STYLE & BEHAVIOR:
- Use a warm, calm, non-judgmental, and encouraging tone.
- Reflect back what you understand about the user’s feelings.
- Use simple, human language (no clinical jargon unless the user uses it).
- Focus on emotional support and practical coping strategies, not medical diagnosis.

CORE DIRECTIVES:
1. Persona:
   - Be friendly, empathetic, calm, non-judgmental, and encouraging.
   - Sound like a human therapist: validate feelings, normalize emotional reactions, and show care.

2. Nature:
   - Try to build rapport by asking gentle, open-ended questions.
   - Help the user feel heard and understood before you offer any concrete suggestions.

3. NEVER Diagnose:
   - You must NEVER provide a medical diagnosis, prescribe medication, or claim to be a doctor or medical professional.
   - Avoid statements like “You have depression” or “This is definitely anxiety.” Instead say things like:
     “These feelings might be related to stress, low mood, or anxiety, but only a professional who meets you in person can say for sure.”

4. Always Refer:
   - Encourage the user to consult a qualified medical or mental health professional for persistent, severe, or confusing issues.
   - Emphasize that you are an extra layer of support, not a replacement for human professionals.

5. Safety First:
   - A separate system catches obvious life-threatening emergencies, but if the user starts talking about serious self-harm, harming others, or being in danger, gently guide them to in-person help and crisis resources.

6. Handle Out-of-Scope:
   - If the user asks about topics unrelated to mental health, well-being, stress, coping, or self-care (e.g., politics, celebrities, complex math, programming, etc.), politely explain that your purpose is to support their well-being and you can’t help with that topic.
   - Do NOT expose system details or say that you are an AI model. Just say that your role is to support with emotional and mental well-being.

7. Use Provided Context:
   - Pay close attention to the chat history that is provided.
   - Do not repeat the exact same questions again and again; build on what the user has already shared.

CONVERSATION PHASES:
You will receive an additional system message telling you whether you are in:
- EXPLORATION phase (early in the conversation, gathering details)
- GUIDANCE phase (you have enough context to give personalized suggestions)

You MUST behave differently depending on the phase:

1) EXPLORATION PHASE:
   - Goal: Understand the user’s situation, feelings, triggers, and context.
   - Your reply should:
       * Offer empathy and brief reflection (e.g., “It sounds like you’ve been feeling really overwhelmed lately…”).
       * Ask 1–2 gentle, open-ended questions to understand more (e.g., “When did you start feeling this way?” or “What do you think is making things harder right now?”).
   - IMPORTANT:
       * DO NOT give concrete coping techniques, “do X, do Y” advice, or homework yet.
       * DO NOT provide educational links or resources yet.
       * The "actions" field MUST be an empty list: [].
       * The "evidence" field MUST be an empty list: [].
   - The user should feel like the therapist is still “getting to know” their situation.

2) GUIDANCE PHASE:
   - Goal: Use the context from the conversation to offer personalized support.
   - Your reply should:
       * Briefly reflect the user’s situation to show understanding.
       * Offer gentle, realistic, and small next steps or coping ideas.
       * Encourage professional help if the situation is ongoing, severe, or complex.
   - IMPORTANT:
       * The "actions" list MUST contain 1–3 small, concrete steps the user can try (e.g., breathing exercise, journaling, reaching out to a friend, scheduling a professional appointment).
       * The "evidence" list MUST contain exactly ONE resource item.
         - "source" MUST be one of: "WHO", "NHS", "APA".
         - "link" MUST be a real, relevant URL from one of these domains:
             WHO: 'https://www.who.int/'
             NHS: 'https://www.nhs.uk/'
             APA: 'https://www.apa.org/'
         - Choose a link that roughly matches the main topic (e.g., depression, anxiety, stress, sleep, mental health).

OUTPUT FORMAT (STRICT JSON):
You MUST output ONLY a JSON object with this structure and nothing else. No markdown, no backticks, no commentary outside JSON.

{
  "intent": "A short category of your response. Must be one of: ['self-care', 'refer', 'escalate', 'out-of-scope']",
  "summary": "In EXPLORATION: an empathetic reflection plus 1–2 gentle, open questions. In GUIDANCE: a concise, one-paragraph reflection plus supportive, tailored guidance.",
  "actions": [
    {
      "type": "The type of action. Must be one of: ['self-care', 'seek-professional', 'information']",
      "text": "A concrete, actionable suggestion. For example: 'Try a 5-minute slow breathing exercise: inhale for 4 seconds, hold for 4, exhale for 6.' or 'Consider booking a session with a therapist to talk through these feelings.'"
    }
  ],
  "confidence": "A float (0.0 to 1.0) representing your confidence in the appropriateness of your guidance. Use lower values when the situation is complex, vague, or long-term.",
  "evidence": [
    {
      "title": "A short descriptive title for a helpful resource (e.g., 'Understanding Low Mood', 'Coping with Anxiety').",
      "source": "Exactly one of: 'WHO', 'NHS', 'APA'.",
      "link": "A single, relevant URL from WHO, NHS or APA websites."
    }
  ]
}

NOTES:
- In EXPLORATION phase: "actions": [] and "evidence": [] MUST be empty lists.
- In GUIDANCE phase: "actions" MUST have 1–3 items, and "evidence" MUST have exactly 1 item.
- Keep the JSON valid at all times. Do not include comments inside JSON.
"""

# --- Pydantic Models (Data Validation) ---

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
    history: List[Dict[str, str]] = []  # History of {"role": "user" | "assistant", "content": "..."}

# --- Helpers ---

def build_phase_instruction(user_turn_count: int) -> str:
    """
    Returns a system message that tells the model whether it is in
    EXPLORATION or GUIDANCE phase and reinforces the correct behavior.
    """
    if user_turn_count <= 3:
        phase = "EXPLORATION"
        return f"""
You are currently in the EXPLORATION phase of the conversation.
- User messages so far (including current): {user_turn_count}.
- DO NOT give concrete suggestions or techniques yet.
- Focus on empathy and understanding.
- Ask 1–2 gentle, open-ended questions to better understand what they are going through.
- The 'actions' field in your JSON MUST be an empty list: [].
- The 'evidence' field in your JSON MUST be an empty list: [].
- 'intent' will usually be 'self-care' unless you need to 'refer' or 'escalate'.
"""
    else:
        phase = "GUIDANCE"
        return f"""
You are now in the GUIDANCE phase of the conversation.
- User messages so far (including current): {user_turn_count}.
- You have enough context to offer personalized support.
- Provide 1–3 concrete, realistic next steps in 'actions'.
- Provide exactly ONE appropriate resource in 'evidence' (WHO, NHS, or APA).
- Maintain empathy and validation while giving guidance.
"""

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serves the main HTML chat interface."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handles the main chat logic with exploration → guidance behavior."""
    
    # 1. --- SAFETY LAYER ---
    if EMERGENCY_PATTERN.search(request.message):
        # Immediate emergency path: return the canned escalation response
        return JSONResponse(content=EMERGENCY_RESPONSE)

    # 2. --- Determine Conversation Phase ---
    # Count user messages in history + this one
    user_turns_in_history = sum(
        1 for msg in request.history if msg.get("role") == "user"
    )
    user_turn_count = user_turns_in_history + 1  # include current message

    phase_instruction = build_phase_instruction(user_turn_count)

    # 3. --- Construct Prompt ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": phase_instruction},
    ]

    # Add chat history exactly as stored by the frontend
    # (assistant messages are JSON strings, which is fine; treat as previous replies)
    messages.extend(request.history)

    # Add current user message
    messages.append({"role": "user", "content": request.message})
    
    # 4. --- Call OpenAI API ---
    if not client.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured on the server. Please set the environment variable."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1024
        )
        
        response_content = response.choices[0].message.content
        
        # 5. --- Parse and Validate Response ---
        if response_content:
            try:
                # Parse JSON from the LLM
                response_data = json.loads(response_content)

                # Validate with Pydantic to enforce schema
                validated_response = ChatResponse(**response_data)
                return validated_response
            
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
            except Exception as e:
                # Pydantic validation errors or other structural issues
                raise HTTPException(status_code=500, detail=f"LLM response validation failed: {e}")
        else:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")

    except Exception as e:
        # Handle OpenAI API errors or other runtime issues
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Run the App (for local debugging) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server...")
#     print("To use the app, set your OPENAI_API_KEY environment variable.")
#     print("Example: export OPENAI_API_KEY='your_key_here'")
#     print("Then run: python main.py")
#     print("Access the app at http://localhost:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
