import pytest

from exchange import Text
from exchange.providers.openai import OpenAiProvider
from .conftest import complete, OPENAI_MODEL


@pytest.mark.vcr()
def test_openai_complete(default_openai_api_key):
    reply_message, reply_usage = complete(OpenAiProvider, OPENAI_MODEL)

    assert reply_message.content == [Text(text="Hello! How can I assist you today?")]
    assert reply_usage.total_tokens == 27


@pytest.mark.integration
def test_openai_complete_integration():
    reply = complete(OpenAiProvider, OPENAI_MODEL)

    assert reply[0].content is not None
    print("Complete content from OpenAI:", reply[0].content)
