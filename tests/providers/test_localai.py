from typing import Tuple

import os
import pytest

from exchange import Text
from exchange.message import Message
from exchange.providers.base import Usage
from exchange.providers.localai import LocalAIProvider, LOCALAI_MODEL


@pytest.mark.vcr()
def test_localai_completion(default_openai_api_key):
    reply_message, reply_usage = localai_complete()

    assert reply_message.content == [Text(text="Hi! How can I help you today?")]
    assert reply_usage.total_tokens == 25


@pytest.mark.integration
def test_localai_completion_integration():
    reply = localai_complete()

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)


def localai_complete() -> Tuple[Message, Usage]:
    provider = LocalAIProvider.from_env()
    model = os.getenv("LOCALAI_MODEL", LOCALAI_MODEL)
    system = "You are a helpful assistant who is succinct."
    messages = [Message.user("Hello")]
    return provider.complete(model=model, system=system, messages=messages, tools=None)
