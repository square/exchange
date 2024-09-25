import os
from unittest.mock import ANY, patch

import pytest
from exchange import Message, Text
from exchange.providers.openai import OpenAiProvider
from exchange.tool import Tool


def dummy_tool() -> str:
    """An example tool"""
    return "dummy response"


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


@patch("httpx.Client.post")
def test_openai_completion_o1_model(mock_post, openai_provider):
    # Mock response from 'o1' model
    mock_reply_content = '{"tool_calls": [{"function": "dummy_tool", "parameters": {}}]}'
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": mock_reply_content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
    }

    mock_post.return_value.json.return_value = mock_response

    model = "o1-mini"
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    tools = [Tool.from_function(dummy_tool)]

    # Call the complete method
    reply_message, reply_usage = openai_provider.complete(model=model, system=system, messages=messages, tools=tools)

    # Check that the assistant's reply was parsed correctly
    assert len(reply_message.tool_use) == 1
    assert reply_message.tool_use[0].name == "dummy_tool"

    # Check that the request payload was constructed correctly
    # For 'o1' models, the system prompt and user query are combined
    expected_user_content = ANY  # We can use ANY because the exact content is constructed dynamically
    expected_messages = [
        {
            "role": "user",
            "content": expected_user_content,
        }
    ]
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == model
    assert kwargs["json"]["messages"] == expected_messages
    assert "tools" not in kwargs["json"]  # 'o1' models should not have 'tools' in the payload


@pytest.mark.integration
def test_openai_integration():
    provider = OpenAiProvider.from_env()
    model = "gpt-4"  # specify a valid model
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]

    reply = provider.complete(model=model, system=system, messages=messages, tools=None)

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)
