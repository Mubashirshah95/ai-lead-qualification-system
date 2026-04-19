"""
AI Lead Qualification & Booking System
Built with FastAPI + Claude API + Vector Embeddings (RAG)

How it works:
1. User sends a message via the chat UI
2. FastAPI receives it and performs semantic search on the knowledge base
3. Only the most relevant chunks are injected into the LLM context
4. Claude uses that grounded context to respond accurately
5. Intent classifier determines if lead is qualified
6. If qualified, simulate a booking
7. All leads are saved to leads.csv
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import csv
import os
from datetime import datetime
from knowledge_base import load_knowledge_base, semantic_search
from intent_classifier import classify_intent

# --- Setup ---
app = FastAPI(title="AI Lead Qualification System")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Load the knowledge base and generate embeddings at startup
load_knowledge_base("knowledge_base.txt")

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Data model ---
class ChatMessage(BaseModel):
    session_id: str
    user_name: str
    user_email: str
    message: str


# --- Conversation history ---
conversation_store: dict[str, list] = {}


# --- Main chat endpoint ---
@app.post("/chat")
async def chat(payload: ChatMessage):
    """
    Receives a message, performs semantic search to find relevant knowledge,
    sends to Claude with grounded context, classifies intent, logs lead.
    """

    history = conversation_store.get(payload.session_id, [])
    history.append({"role": "user", "content": payload.message})

    # Semantic search — find most relevant chunks for this specific message
    # This is the key upgrade: instead of injecting everything, we find
    # only what's relevant to what the user just asked
    relevant_context = semantic_search(payload.message)

    system_prompt = f"""You are Lexi, a friendly and professional AI assistant for Hockley Mint, a luxury British jewellery manufacturer based in Birmingham's historic Jewellery Quarter.

You help customers design and commission bespoke jewellery pieces including engagement rings, wedding bands, necklaces, bracelets, earrings, and signet rings.

Your job is to:
1. Warmly greet new customers and understand what they are looking for
2. Have a natural conversation — ask ONE question at a time, never bombard them
3. Gradually learn their occasion, budget, metal preference, gemstone preference, and timeline across multiple messages
4. Use the knowledge base below to answer questions about our pieces, materials, and pricing
5. Only offer to book a design consultation AFTER you have gathered enough information AND the customer has shown clear interest in proceeding
6. Keep responses warm, elegant, and concise — 2-3 sentences max

CONVERSATION FLOW — follow this naturally across multiple messages:
- First learn what they are looking for (occasion/piece type)
- Then explore their style preferences (metal, gemstone, design)
- Then understand their timeline and budget
- Only then suggest booking a free design consultation
- Wait for them to agree before confirming a booking

IMPORTANT: Do NOT rush to offer a booking. Have a proper conversation first.
Only suggest booking after at least 4-5 meaningful exchanges.

RELEVANT KNOWLEDGE BASE (semantically retrieved for this conversation):
{relevant_context}

Always be helpful, warm, and never pushy. If you don't know something, say so honestly.
When a customer is qualified and ready to book, tell them you will arrange a free 30-minute design consultation."""

    # Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=history
    )

    ai_reply = response.content[0].text

    history.append({"role": "assistant", "content": ai_reply})
    conversation_store[payload.session_id] = history

    # Only classify intent after minimum 3 user messages
    user_messages = [m for m in history if m["role"] == "user"]
    message_count = len(user_messages)

    intent = "NURTURING"
    if message_count >= 3:
        full_conversation = " ".join([m["content"] for m in history])
        intent = classify_intent(full_conversation, client)

    log_lead(
        session_id=payload.session_id,
        name=payload.user_name,
        email=payload.user_email,
        last_message=payload.message,
        intent=intent,
        ai_reply=ai_reply
    )

    booking_confirmed = None
    if intent == "QUALIFIED" and message_count >= 3:
        booking_confirmed = simulate_booking(payload.user_name, payload.user_email)

    return {
        "reply": ai_reply,
        "intent": intent,
        "booking": booking_confirmed
    }


def simulate_booking(name: str, email: str) -> dict:
    slot = "Tuesday 15 April 2026 at 10:00 AM GMT"
    return {
        "confirmed": True,
        "name": name,
        "email": email,
        "slot": slot,
        "message": f"Design consultation booked for {name} on {slot}"
    }


def log_lead(session_id, name, email, last_message, intent, ai_reply):
    file_exists = os.path.isfile("leads.csv")
    with open("leads.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "session_id", "name", "email",
            "last_message", "intent", "ai_reply"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "name": name,
            "email": email,
            "last_message": last_message,
            "intent": intent,
            "ai_reply": ai_reply
        })


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/leads")
async def get_leads():
    if not os.path.isfile("leads.csv"):
        return {"leads": [], "message": "No leads captured yet"}
    leads = []
    with open("leads.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(row)
    return {"total": len(leads), "leads": leads}
