import os

import pytest

from exchange import Text, Message, ToolUse, ToolResult, Tool
from exchange.providers import OpenAiProvider, AzureProvider
from exchange.providers.ollama import OllamaProvider, OLLAMA_MODEL
from .conftest import mark_parametrized
from ..conftest import read_file


# provider configuration for all tests except vision
providers = [
    (OllamaProvider, os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)),
    (OpenAiProvider, os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
    (AzureProvider, os.getenv("AZURE_MODEL", "gpt-4o-mini")),
]

# Currently, only OpenAI (not yet Azure) supports vision in its model.
vision_providers = [t for t in providers if t[0] == OpenAiProvider]


@pytest.mark.vcr
@mark_parametrized(
    providers,
    {
        AzureProvider: ("Hello! How can I assist you today?", 27),
        OllamaProvider: ("Hello! I'm here to help. How can I assist you today? Let's chat. 😊", 33),
        OpenAiProvider: ("Hello! How can I assist you today?", 27),
    },
)
def test_complete(provider_cls, model, expected_text: str, expected_total_tokens: int):
    reply_message, reply_usage = complete(provider_cls, model)

    text = reply_message.content[0]
    assert isinstance(text, Text), f"Expected Text, but was {type(text).__name__}"
    assert text.text == expected_text
    assert reply_usage.total_tokens == expected_total_tokens


@pytest.mark.integration
@mark_parametrized(providers)
def test_complete_integration(provider_cls, model):
    reply_message, reply_usage = complete(provider_cls, model)

    text = reply_message.content[0]
    assert isinstance(text, Text), f"Expected Text, but was {type(text).__name__}"
    assert text.text is not None
    assert reply_usage.total_tokens > 0


def complete(provider_cls, model):
    provider = provider_cls.from_env()
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    return provider.complete(
        model=model,
        system=system,
        messages=messages,
        tools=(),
        seed=3,
        temperature=0.1,  # Always set seed and temperature for determinism
    )


@pytest.mark.vcr
@mark_parametrized(
    providers,
    {
        AzureProvider: ("call_a47abadDxlGKIWjvYYvGVAHa", 125),
        OllamaProvider: ("call_d14omgr7", 133),
        OpenAiProvider: ("call_xXYlw4A7Ud1qtCopuK5gEJrP", 122),
    },
)
def test_complete_tools(provider_cls, model, expected_tool_use_id, expected_total_tokens):
    reply_message, reply_usage = complete_tools(provider_cls, model)

    tool_use = reply_message.content[0]
    assert isinstance(tool_use, ToolUse), f"Expected ToolUse, but was {type(tool_use).__name__}"
    assert tool_use.id == expected_tool_use_id
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
    assert reply_usage.total_tokens == expected_total_tokens


@pytest.mark.integration
@mark_parametrized(providers)
def test_complete_tools_integration(provider_cls, model):
    reply_message, reply_usage = complete_tools(provider_cls, model)

    tool_use = reply_message.content[0]
    assert isinstance(tool_use, ToolUse), f"Expected ToolUse, but was {type(tool_use).__name__}"
    assert tool_use.id is not None
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
    assert reply_usage.total_tokens > 0


def complete_tools(provider_cls, model):
    system = "You are a helpful assistant. Expect to need to read a file using read_file."
    messages = [Message.user("What are the contents of this file? test.txt")]
    provider = provider_cls.from_env()
    return provider.complete(
        model=model,
        system=system,
        messages=messages,
        tools=(Tool.from_function(read_file),),
        seed=3,
        temperature=0.1,  # Always set seed and temperature for determinism
    )


@pytest.mark.vcr
@mark_parametrized(
    vision_providers,
    {
        OpenAiProvider: ('The first entry in the menu says "Ask Goose."', 14241),
    },
)
def test_complete_vision(provider_cls, model, expected_text, expected_total_tokens):
    reply_message, reply_usage = complete_vision(provider_cls, model)

    text = reply_message.content[0]
    assert isinstance(text, Text), f"Expected Text, but was {type(text).__name__}"
    assert text.text == expected_text
    assert reply_usage.total_tokens == expected_total_tokens


@pytest.mark.integration
@mark_parametrized(vision_providers)
def test_complete_vision_integration(provider_cls, model):
    reply_message, reply_usage = complete_vision(provider_cls, model)

    text = reply_message.content[0]
    assert isinstance(text, Text), f"Expected Text, but was {type(text).__name__}"
    assert "ask goose" in text.text.lower()
    assert reply_usage.total_tokens > 0


def complete_vision(provider_cls, model):
    system = "You are a helpful assistant."
    messages = [
        Message.user("What does the first entry in the menu say?"),
        Message(
            role="assistant",
            content=[ToolUse(id="xyz", name="screenshot", parameters={})],
        ),
        Message(
            role="user",
            content=[ToolResult(tool_use_id="xyz", output='"image:tests/test_image.png"')],
        ),
    ]
    provider = provider_cls.from_env()
    return provider.complete(
        model=model,
        system=system,
        messages=messages,
        tools=(),
        seed=3,  # Always set seed and temperature for determinism
        temperature=0.1,
    )
