# AI Proxy

OpenAI-compatible proxy server with self-assessment routing.

Routes queries between local (llama.cpp) and cloud (MiMo) LLMs based on complexity.

## Architecture

```
Client → AI Proxy → Local LLM (llama.cpp)
                  ↘ Cloud LLM (MiMo)
```

## Installation

```bash
# Clone repository
git clone https://github.com/Floggony/ai-proxy.git
cd ai-proxy

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Create `.env` file:

```env
# Local LLM (llama.cpp)
LOCAL_LLM_URL=http://192.168.0.193:5001/v1
LOCAL_LLM_MODEL=Ternary-Bonsai-27B-Q2_g64

# Cloud LLM (MiMo)
CLOUD_LLM_URL=https://token-plan-sgp.xiaomimimo.com/v1
CLOUD_LLM_API_KEY=your-api-key
CLOUD_LLM_MODEL=mimo-v2.5-pro

# Server
HOST=0.0.0.0
PORT=8080
```

## Usage

```bash
# Start server
poetry run uvicorn ai_proxy.main:app --host 0.0.0.0 --port 8080

# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Development

```bash
# Run linter
poetry run ruff check .

# Run formatter
poetry run ruff format .

# Run tests
poetry run pytest

# Run type checker
poetry run mypy ai_proxy
```

## API

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "mimo-v2.5-pro",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mimo-v2.5-pro",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

## License

MIT
