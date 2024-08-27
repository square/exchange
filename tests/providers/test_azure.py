import os
from unittest.mock import patch

import pytest
from ai_exchange import Message, Text
from ai_exchange.providers.azure import AzureProvider


@pytest.fixture
@patch.dict(os.environ, {
    "AZURE_CHAT_COMPLETIONS_HOST_NAME": "https://test.openai.azure.com/",
    "AZURE_CHAT_COMPLETIONS_DEPLOYMENT_NAME": "test-deployment",
    "AZURE_CHAT_COMPLETIONS_DEPLOYMENT_API_VERSION": "2024-02-15-preview",
    "AZURE_CHAT_COMPLETIONS_KEY": "test_api_key"
})
def azure_provider():
    return AzureProvider.from_env()


@patch("httpx.Client.post")
@patch("time.sleep", return_value=None)
@patch("logging.warning")
@patch("logging.error")
def test_azure_completion(mock_error, mock_warning, mock_sleep, mock_post, azure_provider):
    mock_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
    }

    mock_post.return_value.json.return_value = mock_response

    model = "gpt-4"
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    tools = ()

    reply_message, reply_usage = azure_provider.complete(model=model, system=system, messages=messages, tools=tools)

    assert reply_message.content == [Text(text="Hello!")]
    assert reply_usage.total_tokens == 35
    mock_post.assert_called_once_with(
        f"{azure_provider.client.base_url}/chat/completions?api-version={azure_provider.api_version}",
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hello"},
            ],
        },
    )


@pytest.mark.integration
def test_azure_integration():
    provider = AzureProvider.from_env()
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]

    reply = provider.complete(system=system, messages=messages, tools=None)

    assert reply[0].content is not None
    print("Completion content from Azure:", reply[0].content)
