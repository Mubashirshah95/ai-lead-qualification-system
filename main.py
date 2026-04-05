"""
AI Lead Qualification & Booking System
Built with FastAPI + Claude API

How it works:
1. User sends a message via the chat UI
2. FastAPI receives it and sends it to Claude with a qualification prompt
3. Claude reads the knowledge base and decides if the lead is qualified
4. If qualified → simulate a booking. If not → continue nurturing.
5. All leads are saved to leads.csv
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import csv
import os
from datetime import datetime
from knowledge_base import load_knowledge_base
from intent_classifier import classify_intent

# --- Setup ---
app = FastAPI(title="AI Lead Qualification System")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Load the knowledge base once at startup
KNOWLEDGE_BASE = load_knowledge_base("knowledge_base.txt")

# Serve the frontend (index.html + static files)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Data model for incoming chat messages ---
class ChatMessage(BaseModel):
    session_id: str          # Unique ID per conversation
    user_name: str           # Lead's name
    user_email: str          # Lead's email
    message: str             # What they typed


# --- In-memory conversation history (resets on server restart) ---
# In a real system this would be a database
conversation_store: dict[str, list] = {}


# --- Main chat endpoint ---
@app.post("/chat")
async def chat(payload: ChatMessage):
    """
    Receives a message from the lead, sends it to Claude,
    classifies intent, logs the lead, and returns the AI reply.
    """

    # Build conversation history for this session
    history = conversation_store.get(payload.session_id, [])
    history.append({"role": "user", "content": payload.message})

    # System prompt — this is the AI's "personality" and instructions
    system_prompt = f"""You are Lexi, a friendly and professional AI assistant for a business consulting agency.

Your job is to:
1. Warmly greet new leads and understand what they need
2. Ask qualifying questions to understand their budget, timeline, and business size
3. Use the knowledge base below to answer questions about our services
4. When a lead seems ready (has budget, clear need, and urgency), offer to book a discovery call
5. Keep responses concise — 2-3 sentences max unless explaining a service

A lead is QUALIFIED when they have mentioned:
- Their business type or industry
- Some indication of budget or seriousness
- A specific problem they want to solve

KNOWLEDGE BASE (our services and FAQs):
{KNOWLEDGE_BASE}

Always be helpful, never pushy. If you don't know something, say so honestly."""

    # Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=history
    )

    ai_reply = response.content[0].text

    # Add AI reply to conversation history
    history.append({"role": "assistant", "content": ai_reply})
    conversation_store[payload.session_id] = history

    # Classify intent from the full conversation so far
    full_conversation = " ".join([m["content"] for m in history])
    intent = classify_intent(full_conversation, client)

    # Log the lead to CSV
    log_lead(
        session_id=payload.session_id,
        name=payload.user_name,
        email=payload.user_email,
        last_message=payload.message,
        intent=intent,
        ai_reply=ai_reply
    )

    # If qualified, simulate a booking
    booking_confirmed = None
    if intent == "QUALIFIED":
        booking_confirmed = simulate_booking(payload.user_name, payload.user_email)

    return {
        "reply": ai_reply,
        "intent": intent,
        "booking": booking_confirmed
    }


# --- Booking simulation ---
def simulate_booking(name: str, email: str) -> dict:
    """
    Simulates confirming a booking slot.
    In a real system this would connect to Calendly, GHL, etc.
    """
    slot = "Tuesday 15 April 2025 at 10:00 AM GMT"
    return {
        "confirmed": True,
        "name": name,
        "email": email,
        "slot": slot,
        "message": f"Discovery call booked for {name} on {slot}"
    }


# --- CSV logger ---
def log_lead(session_id, name, email, last_message, intent, ai_reply):
    """
    Appends lead data to leads.csv.
    Creates the file with headers if it doesn't exist.
    """
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


# --- Serve the frontend ---
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


# --- View all leads (bonus endpoint) ---
@app.get("/leads")
async def get_leads():
    """Returns all captured leads as JSON."""
    if not os.path.isfile("leads.csv"):
        return {"leads": [], "message": "No leads captured yet"}

    leads = []
    with open("leads.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(row)

    return {"total": len(leads), "leads": leads}
