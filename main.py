"""Cherry AI Agent — clean and optimized."""

import os
import re
import httpx
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer every question accurately, clearly, and concisely.

Rules:
- Give the answer directly. Do not start with "Sure", "Of course", "Great question", or any filler.
- Do not use markdown formatting like **bold**, *italic*, bullet points, or numbered lists.
- Write in plain natural sentences.
- If context or documents are provided, base your answer on them.
- Be thorough but not verbose. Include all important details without unnecessary repetition.
- Always end your answer with proper punctuation."""


async def keep_alive():
    while True:
        await asyncio.sleep(600)
        try:
            if RENDER_URL:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.get(f"{RENDER_URL}/")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            try:
                text = resp.text
                if text and len(text.strip()) > 0:
                    return text[:10000]
            except Exception:
                pass
            return ""
    except Exception:
        return ""


async def ask_llm(query: str, asset_context: str = "") -> str:
    if asset_context.strip():
        user_content = f"{asset_context}\n\nBased on the above, answer this question: {query}"
    else:
        user_content = query

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 1024,
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

    # Clean artifacts
    answer = answer.strip('"').strip("'").strip("`")
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
    answer = re.sub(r'\*(.*?)\*', r'\1', answer)

    for prefix in [
        "Sure! ", "Sure, ", "Well, ", "Certainly! ", "Of course! ",
        "Here's the answer: ", "Answer: ", "Here is the answer: ",
        "The answer is: ", "Response: ", "Output: ",
    ]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):]

    return answer.strip()


@app.post("/v1/answer")
async def answer(request: AgentRequest) -> AgentResponse:
    try:
        asset_context = ""
        if request.assets:
            tasks = [fetch_asset(url) for url in request.assets]
            results = await asyncio.gather(*tasks)
            asset_context = "\n\n".join(r for r in results if r)

        result = await ask_llm(request.query, asset_context)
        return AgentResponse(answer=result)

    except Exception as e:
        return AgentResponse(answer=f"Error: {str(e)}")


@app.get("/")
async def health():
    return {"status": "ok", "service": "cherry-ai-agent"}
