"""Main FastAPI application."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ai_proxy.llm_client import LLMClient
from ai_proxy.schemas import ChatCompletionRequest, ChatCompletionResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
async def chat_completions(request: Request) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    body = await request.body()
    logger.debug("Request body: %s", body.decode("utf-8", errors="replace"))

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON: %s", e)
        return {"error": f"Invalid JSON: {e}"}

    request_obj = ChatCompletionRequest(**data)

    # Convert messages, removing None values and ensuring content is never None
    messages = []
    for msg in request_obj.messages:
        msg_dict = msg.model_dump(exclude_none=True)
        if msg_dict.get("content") is None:
            msg_dict["content"] = ""
        messages.append(msg_dict)

    logger.debug("Processed messages: %s", messages)

    if request_obj.stream:
        generator = await llm_client.chat_completion(
            messages=messages,
            model=request_obj.model,
            stream=True,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    result = await llm_client.chat_completion(
        messages=messages,
        model=request_obj.model,
        stream=False,
    )
    return ChatCompletionResponse(**result)
