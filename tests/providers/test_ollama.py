import pytest
from exchange import Message
from exchange.providers.ollama import OllamaProvider, OLLAMA_MODEL


@pytest.mark.integration
def test_ollama_integration():
    provider = OllamaProvider.from_env()
    model = OLLAMA_MODEL
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]

    reply = provider.complete(model=model, system=system, messages=messages, tools=None)

    assert reply[0].content is not None
    print("Completion content from Ollama:", reply[0].content)
