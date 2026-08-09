"""LLM client for local and cloud models."""

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from ai_proxy.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLM APIs."""

    def __init__(self) -> None:
        self.local_client = httpx.AsyncClient(
            base_url=settings.local_llm_url,
            timeout=120.0,
        )
        self.cloud_client = httpx.AsyncClient(
            base_url=settings.cloud_llm_url,
            headers={"Authorization": f"Bearer {settings.cloud_llm_api_key}"},
            timeout=120.0,
        )

    async def classify_query(self, query: str) -> str:
        """Classify query complexity using local model."""
        prompt = settings.classification_prompt.format(query=query)

        response = await self.local_client.post(
            "/chat/completions",
            json={
                "model": settings.local_llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip().lower()

        return "cloud" if "cloud" in answer else "local"

    async def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
    ) -> dict | AsyncGenerator:
        """Send chat completion request to appropriate model."""
        # Classify if model not specified
        if model is None:
            query = messages[-1].get("content", "")
            target = await self.classify_query(query)
        else:
            target = "cloud" if "mimo" in model else "local"

        # Select client and model
        if target == "cloud":
            client = self.cloud_client
            actual_model = model or settings.cloud_llm_model
        else:
            client = self.local_client
            actual_model = model or settings.local_llm_model

        payload = {
            "model": actual_model,
            "messages": messages,
            "stream": stream,
        }
        logger.debug("Sending to %s: %s", target, json.dumps(payload, ensure_ascii=False)[:500])

        # Make request
        if stream:
            return self._stream_completion(client, actual_model, messages)
        else:
            response = await client.post(
                "/chat/completions",
                json=payload,
            )
            logger.debug("Response status: %s", response.status_code)
            logger.debug("Response body: %s", response.text[:500])
            response.raise_for_status()
            return response.json()

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response."""
        async with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

    async def close(self) -> None:
        """Close HTTP clients."""
        await self.local_client.aclose()
        await self.cloud_client.aclose()
