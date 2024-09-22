from typing import Tuple

import pytest

from exchange import Text
from exchange.message import Message
from exchange.providers.base import Usage
from exchange.providers.ollama import OllamaProvider, OLLAMA_MODEL


@pytest.mark.vcr()
def test_ollama_completion():
    reply_message, reply_usage = ollama_complete()

    assert reply_message.content == [Text(text="Hello! I'm here to help. How can I assist you today? Let's chat. 😊")]
    assert reply_usage.total_tokens == 33


@pytest.mark.integration
def test_ollama_completion_integration():
    reply = ollama_complete()

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)


def ollama_complete() -> Tuple[Message, Usage]:
    provider = OllamaProvider.from_env()
    model = OLLAMA_MODEL
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    return provider.complete(model=model, system=system, messages=messages, tools=None)
