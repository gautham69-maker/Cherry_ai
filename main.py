"""AI Agent — optimized for maximum cosine & Jaccard similarity scoring."""

import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are a precise answering agent being scored on exact text similarity against a reference answer.

CRITICAL RULES — follow every single one:
1. Answer DIRECTLY. No preamble like "Sure!", "Great question!", "Let me think...". Just the answer.
2. Do NOT add disclaimers, caveats, or "I think" / "I believe" / "It seems".
3. Do NOT add extra explanation beyond what's asked.
4. End with a period. One sentence when possible.
5. Use natural, clean English — not robotic, not verbose.

ANSWER PATTERNS — match these exactly:
- Math: "What is 10 + 15?" → "The sum is 25."
- Math: "What is 20 * 3?" → "The product is 60."
- Math: "What is 100 / 4?" → "The result is 25."
- Factual: "What is the capital of France?" → "The capital of France is Paris."
- Definition: "What is photosynthesis?" → "Photosynthesis is the process by which plants convert sunlight into energy."
- Yes/No: "Is the sky blue?" → "Yes, the sky is blue."
- List: "Name 3 colors" → "Three colors are red, blue, and green."

KEY PRINCIPLES:
- Be CONCISE — every extra word hurts your similarity score.
- Be PRECISE — match the natural phrasing a human would expect.
- Be COMPLETE — don't omit key information from the answer.
- Use PROVIDED CONTEXT from assets when available — the answer is usually IN the assets.
- If assets contain the answer, extract and restate it cleanly. Do not add information beyond what's in the assets.
- NEVER refuse to answer. Always give your best answer.
"""


class AgentRequest(BaseModel):
    query: str
    assets: list[str] = []


class AgentResponse(BaseModel):
    answer: str


async def fetch_asset(url: str) -> str:
    """Download text content from an asset URL."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text" in content_type or "json" in content_type or "csv" in content_type:
                return resp.text[:8000]
            elif "pdf" in content_type:
                return f"[PDF document from {url}]"
            return f"[Binary file: {content_type}]"
    except Exception as e:
        return f"[Failed to fetch {url}: {e}]"


async def ask_claude(query: str, asset_context: str = "") -> str:
    """Send query to Claude and return the answer."""

    if asset_context and asset_context.strip():
        user_content = (
            f"Use the following context to answer the question. "
            f"The answer should be based on this context.\n\n"
            f"CONTEXT:\n{asset_context}\n\n"
            f"QUESTION: {query}"
        )
    else:
        user_content = query

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    answer = "".join(
        block["text"] for block in data.get("content", []) if block.get("type") == "text"
    ).strip()

    # Clean up stray quotes or markdown
    answer = answer.strip('"').strip("'").strip("`")

    # Remove common LLM preambles that hurt similarity
    for prefix in [
        "Sure! ", "Sure, ", "Well, ", "Certainly! ",
        "Of course! ", "Here's the answer: ", "Answer: ",
    ]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]

    return answer


@app.post("/v1/answer")
async def answer(request: AgentRequest) -> AgentResponse:
    """Main endpoint — receives a query, returns an answer."""
    try:
        # Fetch assets if provided
        asset_context = ""
        if request.assets:
            parts = []
            for url in request.assets:
                content = await fetch_asset(url)
                parts.append(content)
            asset_context = "\n\n".join(parts)

        # Get answer from Claude
        result = await ask_claude(request.query, asset_context)
        return AgentResponse(answer=result)

    except Exception as e:
        return AgentResponse(answer=f"Error processing request: {str(e)}")


@app.get("/")
async def health():
    return {"status": "ok", "service": "ai-agent"}
