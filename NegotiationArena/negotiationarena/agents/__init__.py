from .chatgpt import ChatGPTAgent

# Optional providers must not prevent OpenAI-compatible local models from
# importing the platform.
try:
    from .claude import ClaudeAgent
except ImportError:  # pragma: no cover - depends on optional anthropic extra
    ClaudeAgent = None

try:
    from .llama2 import LLama2ChatAgent
except ImportError:  # pragma: no cover - depends on optional provider extra
    LLama2ChatAgent = None
