"""Cherry AI Agent — powered by Groq (free + ultra-fast)."""

import os
import httpx
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

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
- If assets contain the answer, extract and restate it cleanly.
- NEVER refuse to answer. Always give your best answer.
"""


# ── Keep-alive: ping self every 10 minutes so Render never sleeps ──
async def keep_alive():
    """Ping self every 10 min to prevent Render free tier from sleeping."""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        try:
            if RENDER_URL:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.get(f"{RENDER_URL}/")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start keep-alive background task
    task = asyncio.create_task(keep_alive())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


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


async def ask_llm(query: str, asset_context: str = "") -> str:
    """Send query to Groq and return the answer."""

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
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    answer = data["choices"][0]["message"]["content"].strip()

    # Clean up stray quotes or markdown
    answer = answer.strip('"').strip("'").strip("`")

    # Remove common LLM preambles that destroy similarity scores
    for prefix in [
        "Sure! ", "Sure, ", "Well, ", "Certainly! ",
        "Of course! ", "Here's the answer: ", "Answer: ",
        "Here is the answer: ", "The answer is: ",
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

        # Get answer from LLM
        result = await ask_llm(request.query, asset_context)
        return AgentResponse(answer=result)

    except Exception as e:
        return AgentResponse(answer=f"Error: {str(e)}")


@app.get("/")
async def health():
    return {"status": "ok", "service": "cherry-ai-agent"}
