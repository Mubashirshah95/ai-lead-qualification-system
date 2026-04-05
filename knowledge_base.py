"""
knowledge_base.py

Loads the knowledge base text file that the AI uses to answer
questions about services, pricing, and FAQs.

In a real system this could:
- Load multiple PDFs
- Scrape a website
- Pull from a CRM
- Use vector embeddings for semantic search (more advanced)

For this project we keep it simple: one plain text file.
"""

import os


def load_knowledge_base(filepath: str) -> str:
    """
    Reads the knowledge base file and returns its contents as a string.
    If the file doesn't exist, returns a default placeholder.

    Args:
        filepath: Path to the knowledge base text file

    Returns:
        String containing the full knowledge base content
    """
    if not os.path.isfile(filepath):
        return """
        DEFAULT KNOWLEDGE BASE (replace with knowledge_base.txt):

        Services offered:
        - AI Automation Setup: We build AI chatbots, voice bots, and automation workflows
        - CRM Implementation: GoHighLevel setup, custom pipelines, and integrations
        - Lead Generation Systems: Automated outreach, follow-up sequences, and qualification bots

        Pricing:
        - Starter package: £997 one-time setup
        - Growth package: £1,997 one-time + £297/month maintenance
        - Enterprise: Custom quote

        Typical results:
        - 40% reduction in admin time
        - 3x increase in lead response speed
        - Clients typically see ROI within 60 days

        Discovery call: 30 minutes, free, no obligation
        """

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"[Knowledge Base] Loaded {len(content)} characters from {filepath}")
    return content
