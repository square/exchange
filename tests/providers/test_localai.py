import os
import pytest

from exchange import Text, ToolUse
from exchange.providers.localai import LocalAIProvider, LOCALAI_MODEL
from tests.providers.conftest import complete, tools

LOCALAI_MODEL = os.getenv("LOCALAI_MODEL", LOCALAI_MODEL)


@pytest.mark.vcr()
def test_localai_complete():
    reply_message, reply_usage = complete(LocalAIProvider, LOCALAI_MODEL)

    assert reply_message.content == [Text(text=" How can I help you today? 😊")]
    assert reply_usage.total_tokens == 32


@pytest.mark.integration
def test_localai_complete_integration():
    reply = complete(LocalAIProvider, LOCALAI_MODEL)

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)


@pytest.mark.vcr()
def test_localai_tools():
    reply_message, reply_usage = tools(LocalAIProvider, LOCALAI_MODEL)

    tool_use = reply_message.content[0]
    assert isinstance(tool_use, ToolUse)
    assert tool_use.id == "1b5c0238-a897-4b59-9bb8-3687da685992"
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
    assert reply_usage.total_tokens == 204


@pytest.mark.integration
def test_localai_tools_integration():
    reply = tools(LocalAIProvider, LOCALAI_MODEL)

    tool_use = reply[0].content[0]
    assert isinstance(tool_use, ToolUse)
    assert tool_use.id is not None
    assert tool_use.name == "read_file"
    assert tool_use.parameters == {"filename": "test.txt"}
