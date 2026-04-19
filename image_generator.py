"""
image_generator.py

Multimodal generative AI extension — generates a bespoke jewellery
visualisation using DALL-E 3 when a lead is qualified.

How it works:
1. Receives the full conversation context
2. Extracts jewellery attributes mentioned by the customer
   (metal type, gemstone, style, occasion)
3. Constructs a detailed prompt for DALL-E 3
4. Calls the OpenAI Images API to generate a photorealistic image
5. Returns the image URL for display in the booking confirmation card

Why DALL-E 3:
- Strong instruction following for specific product descriptions
- Consistent photorealistic quality for jewellery
- Simple REST API integration
- In production, a fine-tuned diffusion model on Hockley Mint's
  actual product catalogue would give better brand consistency

This demonstrates:
- Multimodal generative AI pipeline (text-to-image)
- OpenAI Images API integration
- Prompt engineering for image generation
- Integration of generative AI into a production workflow
"""

import os
from openai import OpenAI

# Initialise OpenAI client
# Uses OPENAI_API_KEY environment variable
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_jewellery_attributes(conversation_text: str, claude_client) -> str:
    """
    Uses Claude to extract key jewellery attributes from the conversation.
    Returns a structured description for the image prompt.

    Args:
        conversation_text: Full conversation history as string
        claude_client: Anthropic client instance

    Returns:
        Structured jewellery description string
    """
    import anthropic

    extraction_prompt = f"""You are a jewellery design assistant. Read this customer conversation and extract the jewellery attributes they mentioned.

CONVERSATION:
{conversation_text}

Extract and return a single sentence describing the piece they want. Include:
- Type of piece (ring, necklace, bracelet etc.)
- Metal (gold colour, silver, platinum)
- Gemstone if mentioned
- Style (classic, modern, minimal etc.)
- Occasion if relevant

Return ONLY the description sentence. Example:
"A classic rose gold engagement ring with a round brilliant diamond in a solitaire setting"

If details are missing, use elegant defaults appropriate for Hockley Mint's luxury brand."""

    response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": extraction_prompt}]
    )

    return response.content[0].text.strip()


def generate_jewellery_image(conversation_text: str, claude_client) -> dict:
    """
    Generates a photorealistic jewellery image using DALL-E 3.

    Args:
        conversation_text: Full conversation to extract attributes from
        claude_client: Anthropic client for attribute extraction

    Returns:
        Dict with image_url and description, or error info
    """

    try:
        # Step 1: Extract jewellery attributes from conversation
        jewellery_description = extract_jewellery_attributes(conversation_text, claude_client)
        print(f"[Image Generator] Extracted description: {jewellery_description}")

        # Step 2: Build DALL-E prompt
        # Structured prompt for consistent, professional jewellery photography
        dalle_prompt = f"""Professional luxury jewellery photography of {jewellery_description}.

Style: High-end product photography on white background, soft studio lighting, 
sharp focus showing fine craftsmanship detail, photorealistic, 8K quality.
Setting: Clean white surface, subtle shadow, no text or watermarks.
Brand aesthetic: British luxury jewellery, Hockley Mint quality."""

        print(f"[Image Generator] Calling DALL-E 3...")

        # Step 3: Call DALL-E 3 API
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        print(f"[Image Generator] Image generated successfully")

        return {
            "success": True,
            "image_url": image_url,
            "description": jewellery_description
        }

    except Exception as e:
        print(f"[Image Generator] Error: {e}")
        return {
            "success": False,
            "image_url": None,
            "description": None,
            "error": str(e)
        }
