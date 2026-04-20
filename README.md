# AI Lead Qualification & Booking System

An end-to-end AI system for intelligent lead qualification, booking automation, and personalised product visualisation — built with Python, FastAPI, PyTorch, and the Claude API.

## Overview

This system demonstrates a production-style AI pipeline that qualifies leads through natural conversation, retrieves relevant knowledge semantically, classifies intent using NLP, and generates personalised jewellery visualisations using DALL-E 3.

Built as a proof of concept for Hockley Mint Limited — a luxury British jewellery manufacturer — to demonstrate AI-driven customer engagement and personalised design previews.

## Architecture

The system is built across four layers:

- **Presentation layer** — Branded chat UI (HTML/CSS/JS)
- **API layer** — FastAPI backend with REST endpoints and session management
- **LLM processing layer** — Claude API for conversational AI and intent classification
- **Data layer** — Vector embeddings, semantic retrieval, CSV lead logging

## Key Features

- **Retrieval-Augmented Generation (RAG)** — sentence-transformers (PyTorch) generates 384-dimensional vector embeddings for the knowledge base; cosine similarity retrieval ensures only relevant chunks are injected into the LLM context, reducing token usage by approximately 90%
- **NLP Intent Classification** — structured LLM prompting classifies conversation state as QUALIFIED, NURTURING, or COLD to drive downstream automation
- **Multimodal Generation** — DALL-E 3 (OpenAI Images API) generates photorealistic jewellery visualisations from natural language conversation context when a lead is qualified
- **Multi-turn Conversation** — session-based conversation history with context persistence across turns
- **Lead Logging** — all conversations and intent scores logged to CSV for analysis

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| LLM | Claude API (Anthropic SDK) |
| Embeddings | sentence-transformers, PyTorch |
| Image Generation | DALL-E 3 (OpenAI Images API) |
| Vector Search | numpy, cosine similarity |
| Frontend | HTML, CSS, JavaScript |

## Setup

```bash
# Clone the repo
git clone https://github.com/Mubashirshah95/ai-lead-qualification-system

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here

# Run the server
python3 -m uvicorn main:app --reload
```

Visit `http://localhost:8000` to start the chat interface.

## How It Works

1. User enters name and email and begins a conversation with Lexi — Hockley Mint's AI consultant
2. Each message triggers semantic search across the knowledge base — only the most relevant chunks are retrieved
3. Claude responds using grounded context, asking qualifying questions naturally
4. After sufficient exchanges, the NLP classifier determines if the lead is QUALIFIED
5. On qualification — a consultation is booked and DALL-E 3 generates a personalised jewellery preview image based on the conversation

## Project Context

This project was built to demonstrate the Python engineering layer behind commercial AI systems — specifically for the Hockley Mint KTP Associate application (BCU / Innovate UK).

The architecture directly maps to the KTP project scope: conversational AI, retrieval pipelines, multimodal generation, and API-driven system integration.
