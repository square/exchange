from typing import Tuple

import os
import pytest

from exchange import Text
from exchange.message import Message
from exchange.providers.base import Usage
from exchange.providers.openai import OpenAiProvider
from .conftest import OPENAI_MODEL, OPENAI_API_KEY


@pytest.mark.vcr()
def test_openai_completion(monkeypatch):
    # When running VCR tests the first time, it needs OPENAI_API_KEY to call
    # the real service. Afterward, it is not needed as VCR mocks the service.
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", OPENAI_API_KEY)

    reply_message, reply_usage = openai_complete()

    assert reply_message.content == [Text(text="Hello! How can I assist you today?")]
    assert reply_usage.total_tokens == 27


@pytest.mark.integration
def test_openai_completion_integration():
    reply = openai_complete()

    assert reply[0].content is not None
    print("Completion content from OpenAI:", reply[0].content)


def openai_complete() -> Tuple[Message, Usage]:
    provider = OpenAiProvider.from_env()
    model = OPENAI_MODEL
    system = "You are a helpful assistant."
    messages = [Message.user("Hello")]
    return provider.complete(model=model, system=system, messages=messages, tools=None)
