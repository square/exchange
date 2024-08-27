import os
from unittest.mock import patch

import pytest
from ai_exchange import Message, Text
from ai_exchange.providers.openai import OpenAiProvider


@pytest.fixture
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
def openai_provider():
    return OpenAiProvider.from_env()


@patch("httpx.Client.post")
@patch("time.sleep", return_value=None)
@patch("logging.warning")
@patch("logging.error")
def test_openai_completion(mock_error, mock_warning, mock_sleep, mock_post, openai_provider):
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
    }

    mock_post.return_value.json.return_value = mock_response

    model = "gpt-4"
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    tools = ()

    reply_message, reply_usage = openai_provider.complete(model=model, system=system, messages=messages, tools=tools)

    assert reply_message.content == [Text(text="Hello!")]
    assert reply_usage.total_tokens == 35
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
def test_openai_integration():
    provider = OpenAiProvider.from_env()
    model = "gpt-4"  # specify a valid model
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]

    reply = provider.complete(model=model, system=system, messages=messages, tools=None)

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)
