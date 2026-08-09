"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """Chat message."""

    role: str
    content: str | list | None = None
    name: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Convert list content to string."""
        if isinstance(v, list):
            # Extract text from content blocks: [{"type": "text", "text": "..."}]
            parts = []
            for item in v:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts) if parts else None
        return v


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str | None = None
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    """Chat completion choice."""

    index: int
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    """Token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
