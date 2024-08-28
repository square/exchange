import os
from unittest.mock import patch

import pytest
from exchange import Message, Text
from exchange.providers.ollama import OllamaProvider


@pytest.fixture
@patch.dict(os.environ, {})
def ollama_provider():
    return OllamaProvider.from_env()


@patch("httpx.Client.post")
@patch("time.sleep", return_value=None)
@patch("logging.warning")
@patch("logging.error")
def test_ollama_completion(mock_error, mock_warning, mock_sleep, mock_post, ollama_provider):
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
    }

    mock_post.return_value.json.return_value = mock_response

    model = "llama2"
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    tools = ()

    reply_message, _ = ollama_provider.complete(model=model, system=system, messages=messages, tools=tools)

    assert reply_message.content == [Text(text="Hello!")]
    mock_post.assert_called_once_with(
        "v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hello"},
            ],
            "model": model,
        },
    )


@pytest.mark.integration
def test_ollama_integration():
    provider = OllamaProvider.from_env()
    model = "llama2"  # specify a valid model
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]

    reply = provider.complete(model=model, system=system, messages=messages, tools=None)

    assert reply[0].content is not None
    print("Completion content from Ollama:", reply[0].content)
