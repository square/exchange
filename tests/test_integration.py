import pytest
from exchange.exchange import Exchange
from exchange.message import Message
from exchange.moderators import ContextTruncate
from exchange.providers import get_provider
from exchange.providers.ollama import OLLAMA_MODEL
from exchange.tool import Tool

too_long_chars = "x" * (2**20 + 1)

cases = [
    (get_provider("ollama"), OLLAMA_MODEL),
    (get_provider("openai"), "gpt-4o-mini"),
    (get_provider("databricks"), "databricks-meta-llama-3-70b-instruct"),
    (get_provider("bedrock"), "anthropic.claude-3-5-sonnet-20240620-v1:0"),
]


@pytest.mark.integration  # skipped in CI/CD
@pytest.mark.parametrize("provider,model", cases)
def test_simple(provider, model):
    provider = provider.from_env()

    ex = Exchange(
        provider=provider,
        model=model,
        moderator=ContextTruncate(model),
        system="You are a helpful assistant.",
    )

    ex.add(Message.user("Who is the most famous wizard from the lord of the rings"))

    response = ex.reply()

    # It's possible this can be flakey, but in experience so far haven't seen it
    assert "gandalf" in response.text.lower()


@pytest.mark.integration  # skipped in CI/CD
@pytest.mark.parametrize("provider,model", cases)
def test_tools(provider, model):
    provider = provider.from_env()

    def get_password() -> str:
        """Return the password for authentication"""
        return "mellon"

    ex = Exchange(
        provider=provider,
        model=model,
        moderator=ContextTruncate(model),
        system="You are a helpful assistant. Expect to need to authenticate using get_password.",
        tools=(Tool.from_function(get_password),),
    )

    ex.add(Message.user("Can you authenticate this session by responding with the password"))

    response = ex.reply()

    # It's possible this can be flakey, but in experience so far haven't seen it
    assert "mellon" in response.text.lower()


@pytest.mark.integration
@pytest.mark.parametrize("provider,model", cases)
def test_tool_use_output_chars(provider, model):
    provider = provider.from_env()

    def get_password() -> str:
        """Return the password for authentication"""
        return too_long_chars

    ex = Exchange(
        provider=provider,
        model=model,
        moderator=ContextTruncate(model),
        system="You are a helpful assistant. Expect to need to authenticate using get_password.",
        tools=(Tool.from_function(get_password),),
    )

    ex.add(Message.user("Can you authenticate this session by responding with the password"))

    ex.reply()

    # Without our error handling, this would raise
    # string too long. Expected a string with maximum length 1048576, but got a string with length ...
