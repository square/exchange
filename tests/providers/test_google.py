import os

import pytest
from exchange import Message, Text
from exchange.content import ToolResult, ToolUse
from exchange.providers.google import GoogleProvider
from exchange.tool import Tool
from tests.providers.conftest import complete, tools

GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")


def example_fn(param: str) -> None:
    """
    Testing function.

    Args:
        param (str): Description of param1
    """
    pass


def test_google_response_to_text_message() -> None:
    response = {"candidates": [{"content": {"parts": [{"text": "Hello from Gemini!"}], "role": "model"}}]}
    message = GoogleProvider.google_response_to_message(response)
    assert message.content[0].text == "Hello from Gemini!"


def test_google_response_to_tool_use_message() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"functionCall": {"name": "example_fn", "args": {"param": "value"}}}],
                    "role": "model",
                }
            }
        ]
    }

    message = GoogleProvider.google_response_to_message(response)
    assert message.content[0].name == "example_fn"
    assert message.content[0].parameters == {"param": "value"}


def test_tools_to_google_spec() -> None:
    tools = (Tool.from_function(example_fn),)
    expected_spec = {
        "functionDeclarations": [
            {
                "name": "example_fn",
                "description": "Testing function.",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string", "description": "Description of param1"}},
                    "required": ["param"],
                },
            }
        ]
    }
    result = GoogleProvider.tools_to_google_spec(tools)
    assert result == expected_spec


def test_message_text_to_google_spec() -> None:
    messages = [Message.user("Hello, Gemini")]
    expected_spec = [{"role": "user", "parts": [{"text": "Hello, Gemini"}]}]
    result = GoogleProvider.messages_to_google_spec(messages)
    assert result == expected_spec


def test_messages_to_google_spec() -> None:
    messages = [
        Message(role="user", content=[Text(text="Hello, Gemini")]),
        Message(
            role="assistant",
            content=[ToolUse(id="1", name="example_fn", parameters={"param": "value"})],
        ),
        Message(role="user", content=[ToolResult(tool_use_id="1", output="Result")]),
    ]
    actual_spec = GoogleProvider.messages_to_google_spec(messages)
    #  !=
    expected_spec = [
        {"role": "user", "parts": [{"text": "Hello, Gemini"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "example_fn", "args": {"param": "value"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "1", "response": {"content": "Result"}}}]},
    ]

    assert actual_spec == expected_spec


@pytest.mark.vcr()
def test_google_complete(default_google_env):
    reply_message, reply_usage = complete(GoogleProvider, GOOGLE_MODEL)

    assert reply_message.content == [Text("Hello! 👋  How can I help you today? 😊 \n")]
    assert reply_usage.total_tokens == 20


@pytest.mark.integration
def test_google_complete_integration():
    reply = complete(GoogleProvider, GOOGLE_MODEL)

    assert reply[0].content is not None
    print("Completion content from Google:", reply[0].content)


@pytest.mark.vcr()
def test_google_tools(default_google_env):
    reply_message, reply_usage = tools(GoogleProvider, GOOGLE_MODEL)

    tool_use = reply_message.content[0]
    assert isinstance(tool_use, ToolUse), f"Expected ToolUse, but was {type(tool_use).__name__}"
    assert tool_use.id == "read_file"
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
    assert reply_usage.total_tokens == 118


@pytest.mark.integration
def test_google_tools_integration():
    reply = tools(GoogleProvider, GOOGLE_MODEL)

    tool_use = reply[0].content[0]
    assert isinstance(tool_use, ToolUse), f"Expected ToolUse, but was {type(tool_use).__name__}"
    assert tool_use.id is not None
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
