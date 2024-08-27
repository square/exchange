import pytest
from ai_exchange.content import ToolResult, ToolUse
from ai_exchange.exchange import Exchange
from ai_exchange.message import Message
from ai_exchange.providers import get_provider


cases = [
    (get_provider("openai"), "gpt-4o-mini"),
]


@pytest.mark.integration  # skipped in CI/CD
@pytest.mark.parametrize("provider,model", cases)
def test_simple(provider, model):
    provider = provider.from_env()

    ex = Exchange(
        provider=provider,
        model=model,
        system="You are a helpful assistant.",
    )

    ex.add(Message.user("What does the first entry in the menu say?"))
    ex.add(
        Message(
            role="assistant",
            content=[ToolUse(id="xyz", name="screenshot", parameters={})],
        )
    )
    ex.add(
        Message(
            role="user",
            content=[ToolResult(tool_use_id="xyz", output='"image:tests/test_image.png"')],
        )
    )

    response = ex.reply()

    # It's possible this can be flakey, but in experience so far haven't seen it
    assert "ask goose" in response.text.lower()
