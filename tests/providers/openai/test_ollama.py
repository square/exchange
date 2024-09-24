import pytest

from exchange import Text
from exchange.providers.ollama import OllamaProvider
from .conftest import complete, OLLAMA_MODEL


@pytest.mark.vcr()
def test_ollama_complete(default_openai_api_key):
    reply_message, reply_usage = complete(OllamaProvider, OLLAMA_MODEL)

    assert reply_message.content == [Text(text="Hello! I'm here to help. How can I assist you today? Let's chat. 😊")]
    assert reply_usage.total_tokens == 33


@pytest.mark.integration
def test_ollama_complete_integration():
    reply = complete(OllamaProvider, OLLAMA_MODEL)

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)
