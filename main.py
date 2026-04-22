"""Cherry AI Agent — with built-in cosine similarity tester."""

import os
import re
import math
import httpx
import asyncio
from contextlib import asynccontextmanager
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

SYSTEM_PROMPT = """You are a question-answering agent. Your answers are evaluated using cosine similarity against a reference answer.

ABSOLUTE RULES:
1. Output ONLY the answer — nothing else.
2. No greetings, no preamble, no "Sure!", no "Here's", no "I think".
3. No markdown formatting, no bullet points, no numbered lists.
4. Give a complete, natural, single-paragraph answer.
5. Do NOT over-explain. Do NOT under-explain. Match what a knowledgeable human would say.

FOR MATH QUESTIONS:
- "What is 10 + 15?" → "The sum is 25."
- Show the computation naturally. State the result.

FOR FACTUAL QUESTIONS:
- Give a clear, accurate, concise answer in 1-3 sentences.
- Include key facts but no filler words.

FOR QUESTIONS WITH CONTEXT/ASSETS:
- The answer is IN the provided context. Extract it directly.
- Restate the relevant information from the context cleanly.
- Do NOT add information that isn't in the context.
- Do NOT say "based on the context" or "according to the provided information".

FOR EXPLANATIONS:
- Give a clear, concise explanation in 2-4 sentences.
- Cover the key points without rambling.

CRITICAL: Your goal is to match the reference answer as closely as possible in meaning and word choice. Be natural, precise, and complete."""


# ── Scoring utilities ──
def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using word frequencies."""
    words1 = Counter(tokenize(text1))
    words2 = Counter(tokenize(text2))
    
    if not words1 or not words2:
        return 0.0
    
    all_words = set(words1.keys()) | set(words2.keys())
    dot = sum(words1.get(w, 0) * words2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(v ** 2 for v in words1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in words2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


# ── Keep-alive ──
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


class TestCase(BaseModel):
    query: str
    expected_output: str
    assets: list[str] = []


class TestRequest(BaseModel):
    test_cases: list[TestCase]


class TestResultItem(BaseModel):
    query: str
    expected: str
    actual: str
    cosine: float
    jaccard: float


class TestResponse(BaseModel):
    results: list[TestResultItem]
    avg_cosine: float
    avg_jaccard: float


async def fetch_asset(url: str) -> str:
    """Download text content from an asset URL."""
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
            return f"[Could not read content from {url}]"
    except Exception as e:
        return f"[Failed to fetch {url}: {e}]"


async def ask_llm(query: str, asset_context: str = "") -> str:
    """Send query to Groq and return the answer."""

    if asset_context and asset_context.strip():
        user_content = (
            f"Context:\n{asset_context}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question using the context above. "
            f"Output only the answer, nothing else."
        )
    else:
        user_content = f"Question: {query}\n\nOutput only the answer, nothing else."

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

    # Clean up artifacts
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

    answer = answer.strip()
    return answer


async def get_answer(query: str, assets: list[str]) -> str:
    """Shared logic — fetch assets + get LLM answer."""
    asset_context = ""
    if assets:
        tasks = [fetch_asset(url) for url in assets]
        results = await asyncio.gather(*tasks)
        asset_context = "\n\n".join(results)
    return await ask_llm(query, asset_context)


# ── Main endpoint ──
@app.post("/v1/answer")
async def answer(request: AgentRequest) -> AgentResponse:
    try:
        result = await get_answer(request.query, request.assets)
        return AgentResponse(answer=result)
    except Exception as e:
        return AgentResponse(answer=f"Error: {str(e)}")


# ── Self-test endpoint ──
@app.post("/v1/test")
async def test_agent(request: TestRequest) -> TestResponse:
    """Test the agent against known query/expected pairs and return scores."""
    results = []

    for tc in request.test_cases:
        try:
            actual = await get_answer(tc.query, tc.assets)
        except Exception as e:
            actual = f"Error: {str(e)}"

        cos = cosine_similarity(actual, tc.expected_output)
        jac = jaccard_similarity(actual, tc.expected_output)

        results.append(TestResultItem(
            query=tc.query,
            expected=tc.expected_output,
            actual=actual,
            cosine=round(cos * 100, 1),
            jaccard=round(jac * 100, 1),
        ))

    avg_cosine = sum(r.cosine for r in results) / len(results) if results else 0
    avg_jaccard = sum(r.jaccard for r in results) / len(results) if results else 0

    return TestResponse(
        results=results,
        avg_cosine=round(avg_cosine, 1),
        avg_jaccard=round(avg_jaccard, 1),
    )


@app.get("/")
async def health():
    return {"status": "ok", "service": "cherry-ai-agent"}
