import pytest
from attr.exceptions import FrozenInstanceError
from ai_exchange.content import Text
from ai_exchange.exchange import Exchange
from ai_exchange.moderators import PassiveModerator
from ai_exchange.message import Message
from ai_exchange.providers import Provider, Usage
from ai_exchange.tool import Tool


def dummy_tool() -> str:
    """An example tool"""
    return "dummy response"


class MockProvider(Provider):
    def complete(self, model, system, messages, tools=None):
        return Message(role="assistant", content=[Text(text="This is a mock response.")]), Usage.from_dict(
            {"total_tokens": 35}
        )


def test_exchange_immutable():
    # Create an instance of Exchange
    provider = MockProvider()
    # intentionally setting a list instead of tuple on tools, it should be converted
    exchange = Exchange(
        provider=provider,
        model="test-model",
        system="test-system",
        tools=(Tool.from_function(dummy_tool),),
        messages=[Message(role="user", content=[Text(text="Hello!")])],
        moderator=PassiveModerator(),
    )

    # Try to directly modify a field (should raise an error)
    with pytest.raises(FrozenInstanceError):
        exchange.system = ""

    with pytest.raises(AttributeError):
        exchange.tools.append("anything")

    # Replace method should return a new instance with deepcopy of messages
    new_exchange = exchange.replace(system="changed")

    assert new_exchange.system == "changed"
    assert len(exchange.messages) == 1
    assert len(new_exchange.messages) == 1

    # Ensure that the messages are deep copied
    new_exchange.messages[0].content[0].text = "Changed!"
    assert exchange.messages[0].content[0].text == "Hello!"
    assert new_exchange.messages[0].content[0].text == "Changed!"
