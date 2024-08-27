from typing import List

from ai_exchange.content import Text, ToolResult, ToolUse
from ai_exchange.exchange import Exchange
from ai_exchange.message import Message
from ai_exchange.moderators import PassiveModerator
from ai_exchange.providers import Provider, Usage
from ai_exchange.tool import Tool


def dummy_tool() -> str:
    """An example tool"""
    return "dummy response"


too_long_output = "x" * (2**20 + 1)
too_long_token_output = "word " * 128000


class MockProvider(Provider):
    def __init__(self, sequence: List[Message], usage_dicts: List[dict]):
        # We'll use init to provide a preplanned reply sequence
        self.sequence = sequence
        self.call_count = 0
        self.usage_dicts = usage_dicts

    @staticmethod
    def get_usage(data: dict) -> Usage:
        usage = data.pop("usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def complete(self, model: str, system: str, messages: List[Message], tools: List[Tool]) -> Message:
        output = self.sequence[self.call_count]
        usage = self.get_usage(self.usage_dicts[self.call_count])
        self.call_count += 1
        return (output, usage)


def test_reply_with_unsupported_tool():
    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(
                    role="assistant",
                    content=[ToolUse(id="1", name="unsupported_tool", parameters={})],
                ),
                Message(
                    role="assistant",
                    content=[Text(text="Here is the completion after tool call")],
                ),
            ],
            usage_dicts=[
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=(Tool.from_function(dummy_tool),),
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="test")]))

    ex.reply()

    content = ex.messages[-2].content[0]
    assert isinstance(content, ToolResult) and content.is_error and "no tool exists" in content.output.lower()


def test_invalid_tool_parameters():
    """Test handling of invalid tool parameters response"""
    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(
                    role="assistant",
                    content=[ToolUse(id="1", name="dummy_tool", parameters="invalid json")],
                ),
                Message(
                    role="assistant",
                    content=[Text(text="Here is the completion after tool call")],
                ),
            ],
            usage_dicts=[
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=[Tool.from_function(dummy_tool)],
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="test invalid parameters")]))

    ex.reply()

    content = ex.messages[-2].content[0]
    assert isinstance(content, ToolResult) and content.is_error and "invalid json" in content.output.lower()


def test_max_tool_use_when_limit_reached():
    """Test the max_tool_use parameter in the reply method."""
    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(
                    role="assistant",
                    content=[ToolUse(id="1", name="dummy_tool", parameters={})],
                ),
                Message(
                    role="assistant",
                    content=[ToolUse(id="2", name="dummy_tool", parameters={})],
                ),
                Message(
                    role="assistant",
                    content=[ToolUse(id="3", name="dummy_tool", parameters={})],
                ),
            ],
            usage_dicts=[
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=[Tool.from_function(dummy_tool)],
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="test max tool use")]))

    response = ex.reply(max_tool_use=3)

    assert ex.provider.call_count == 3
    assert "reached the limit" in response.content[0].text.lower()

    assert isinstance(ex.messages[-2].content[0], ToolResult) and ex.messages[-2].content[0].tool_use_id == "3"

    assert ex.messages[-1].role == "assistant"


def test_tool_output_too_long_character_error():
    """Test tool handling when output exceeds character limit."""

    def long_output_tool_char() -> str:
        return too_long_output

    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(
                    role="assistant",
                    content=[ToolUse(id="1", name="long_output_tool_char", parameters={})],
                ),
                Message(
                    role="assistant",
                    content=[Text(text="Here is the completion after tool call")],
                ),
            ],
            usage_dicts=[
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=[Tool.from_function(long_output_tool_char)],
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="test long output char")]))

    ex.reply()

    content = ex.messages[-2].content[0]
    assert (
        isinstance(content, ToolResult)
        and content.is_error
        and "output that was too long to handle" in content.output.lower()
    )


def test_tool_output_too_long_token_error():
    """Test tool handling when output exceeds token limit."""

    def long_output_tool_token() -> str:
        return too_long_token_output

    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(
                    role="assistant",
                    content=[ToolUse(id="1", name="long_output_tool_token", parameters={})],
                ),
                Message(
                    role="assistant",
                    content=[Text(text="Here is the completion after tool call")],
                ),
            ],
            usage_dicts=[
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
                {"usage": {"input_tokens": 12, "output_tokens": 23}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=[Tool.from_function(long_output_tool_token)],
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="test long output token")]))

    ex.reply()

    content = ex.messages[-2].content[0]
    assert (
        isinstance(content, ToolResult)
        and content.is_error
        and "output that was too long to handle" in content.output.lower()
    )


def test_usage_param():
    usage_tests = {
        0: ({"usage": {"total_tokens": 35}}, 35),
        1: ({"usage": {"input_tokens": 12, "output_tokens": 23}}, 35),
    }

    for count, (usage_dict, outcome) in usage_tests.items():
        ex = Exchange(
            provider=MockProvider(
                sequence=[
                    Message(
                        role="assistant",
                        content=[Text(text="Here is the completion after tool call")],
                    ),
                ],
                usage_dicts=[usage_dict],
            ),
            model="gpt-4o-2024-05-13",
            system="You are a helpful assistant.",
            tools=(Tool.from_function(dummy_tool),),
            moderator=PassiveModerator(),
        )

        ex.add(Message(role="user", content=[Text(text="test")]))

        ex.reply()
        assert ex.checkpoints.pop(-1).token_count == outcome


def test_checkpoints_on_exchange():
    """Test checkpoints on an exchange."""
    ex = Exchange(
        provider=MockProvider(
            sequence=[
                Message(role="assistant", content=[Text(text="Message 1")]),
                Message(role="assistant", content=[Text(text="Message 2")]),
                Message(role="assistant", content=[Text(text="Message 3")]),
            ],
            usage_dicts=[
                {"usage": {"total_tokens": 10}},
                {"usage": {"total_tokens": 28}},
                {"usage": {"total_tokens": 33}},
            ],
        ),
        model="gpt-4o-2024-05-13",
        system="You are a helpful assistant.",
        tools=(Tool.from_function(dummy_tool),),
        moderator=PassiveModerator(),
    )

    ex.add(Message(role="user", content=[Text(text="User message")]))
    ex.reply()
    ex.add(Message(role="user", content=[Text(text="User message")]))
    ex.reply()
    ex.add(Message(role="user", content=[Text(text="User message")]))
    ex.reply()

    # Check if checkpoints are created correctly
    checkpoints = ex.checkpoints
    print(checkpoints)
    assert len(checkpoints) == 3
    assert checkpoints[0].token_count == 10
    assert checkpoints[1].token_count == 18
    assert checkpoints[2].token_count == 5

    # Check if the messages are ordered correctly
    assert [msg.content[0].text for msg in ex.messages] == [
        "User message",
        "Message 1",
        "User message",
        "Message 2",
        "User message",
        "Message 3",
    ]
