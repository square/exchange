import os

import pytest

from exchange import Text
from exchange.providers.azure import AzureProvider
from .conftest import complete

AZURE_MODEL = os.getenv("AZURE_MODEL", "gpt-4o-mini")


@pytest.mark.vcr()
def test_azure_complete(default_azure_env):
    reply_message, reply_usage = complete(AzureProvider, AZURE_MODEL)

    assert reply_message.content == [Text(text="Hello! How can I assist you today?")]
    assert reply_usage.total_tokens == 27


@pytest.mark.integration
def test_azure_complete_integration():
    reply = complete(AzureProvider, AZURE_MODEL)

    assert reply[0].content is not None
    print("Complete content from Azure:", reply[0].content)
