"""
intent_classifier.py

Uses Claude to analyse the full conversation and classify
the lead's intent into one of three categories:

  QUALIFIED  — Ready to book. Has a clear need, budget signals, and urgency.
  NURTURING  — Interested but not ready yet. Needs more information.
  COLD       — Not a fit or just browsing with no real intent.

Why use Claude for classification instead of keywords?
Because "I can't really afford much" and "budget isn't an issue"
both contain the word "budget" — a language model understands
the difference. Keywords don't.

This is the NLP component of the project, which maps directly
to the KTP job description requirement for LLM and NLP experience.
"""

import anthropic


def classify_intent(conversation_text: str, client: anthropic.Anthropic) -> str:
    """
    Sends the full conversation to Claude with a classification prompt.
    Returns one of: QUALIFIED, NURTURING, COLD

    Args:
        conversation_text: The full conversation as a single string
        client: An initialised Anthropic client

    Returns:
        String label: "QUALIFIED", "NURTURING", or "COLD"
    """

    classification_prompt = f"""You are a lead qualification analyst. 
    
Read this sales conversation and classify the lead's intent.

CONVERSATION:
{conversation_text}

CLASSIFICATION RULES:
- QUALIFIED: Lead has shared their business type, indicated budget or seriousness, and has a specific problem to solve. They are ready for a discovery call.
- NURTURING: Lead is interested and engaging but hasn't confirmed budget, timeline, or specific need yet.  
- COLD: Lead is just browsing, has no real intent, or is clearly not a fit.

Respond with ONLY one word: QUALIFIED, NURTURING, or COLD.
No explanation. No punctuation. Just the single word."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,  # We only need one word back
        messages=[{"role": "user", "content": classification_prompt}]
    )

    # Extract and clean the response
    raw = response.content[0].text.strip().upper()

    # Validate — default to NURTURING if unexpected response
    valid_intents = {"QUALIFIED", "NURTURING", "COLD"}
    if raw not in valid_intents:
        print(f"[Intent Classifier] Unexpected response: '{raw}', defaulting to NURTURING")
        return "NURTURING"

    print(f"[Intent Classifier] Lead classified as: {raw}")
    return raw
