"""Main FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ai_proxy.llm_client import LLMClient
from ai_proxy.schemas import ChatCompletionRequest, ChatCompletionResponse

llm_client = LLMClient()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    yield
    await llm_client.close()


app = FastAPI(
    title="AI Proxy",
    description="OpenAI-compatible proxy with self-assessment routing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    messages = [msg.model_dump() for msg in request.messages]

    if request.stream:
        generator = await llm_client.chat_completion(
            messages=messages,
            model=request.model,
            stream=True,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    result = await llm_client.chat_completion(
        messages=messages,
        model=request.model,
        stream=False,
    )
    return ChatCompletionResponse(**result)
