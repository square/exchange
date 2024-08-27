import pytest
from ai_exchange import Exchange
from ai_exchange.checkpoint import Checkpoint
from ai_exchange.message import Message
from ai_exchange.providers import Provider, Usage
from ai_exchange.moderators.truncate import ContextTruncate, pop_checkpoint
from typing import List
from ai_exchange.content import Text
from ai_exchange.tool import Tool


class MockProvider(Provider):
    def __init__(self, sequence: List[Message], usage_dicts: List[dict]):
        # We'll use init to provide a preplanned reply sequence
        self.total_counts = len(sequence)
        self.sequence = sequence
        self.call_count = 0
        self.prev_count = 0
        self.usage_dicts = usage_dicts

    @staticmethod
    def get_usage(data: dict) -> Usage:
        usage = data.get("usage", {})
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
        # truncate moderator does a second call to generate in the first pass, hence this skip
        output = self.sequence[self.call_count % self.total_counts]
        usage = self.get_usage(self.usage_dicts[self.call_count % self.total_counts])
        # truncate moderator does a second call to generate in the first pass, hence this skip
        if self.call_count == 0 and self.prev_count == 0:
            self.prev_count += 1
        else:
            self.call_count += 1
        return (output, usage)


@pytest.fixture
def exchange():
    messages = [Message(role="user", content=[Text(text="Hello")])]
    checkpoints = [Checkpoint(start_index=0, end_index=1, token_count=100, latest_generated_tokens=50)]
    provider = MockProvider(
        sequence=[
            Message(
                role="assistant",
                content=[Text(text="Mock response")],
            ),
            Message(
                role="assistant",
                content=[Text(text="Mock response 2")],
            ),
            Message(
                role="assistant",
                content=[Text(text="This is a long context response")],
            ),
            Message(
                role="assistant",
                content=[Text(text="Normal Mock response")],
            ),
        ],
        usage_dicts=[
            {"usage": {"input_tokens": 100 + 20, "output_tokens": 124000 // 2}},
            {
                "usage": {
                    "input_tokens": (100 + 20 + 124000 // 2) + 30,
                    "output_tokens": 40,
                }
            },
            {
                "usage": {
                    "input_tokens": (100 + 20 + 30 + 40 + 124000 // 2) + 50,
                    "output_tokens": 126000 // 2,
                }
            },
            #  this is where truncation will get adjusted the input tokens for
            # next batch will be total tokens from last two usage dicts
            # which equals 63000 + 50 + 40 + 30
            {"usage": {"input_tokens": 63220 + 11, "output_tokens": 26}},
        ],
    )
    exchange = Exchange(
        provider=provider,
        model="test_model",
        system="test_system",
        moderator=ContextTruncate(),
        tools=[],
        messages=messages,
        checkpoints=checkpoints,
    )
    return exchange


def test_pop_checkpoint(exchange):
    exchange.reply()  # get assistant response and adds to checkpoint/messages
    exchange.add(Message(role="user", content=[Text(text="Hi2")]))
    exchange.reply()  # get assistant response and adds to checkpoint/messages for 2

    new_exchange = pop_checkpoint(exchange)
    assert len(new_exchange.checkpoints) == 2
    assert len(new_exchange.messages) == 3

    assert new_exchange.checkpoints[0].start_index == 0
    assert new_exchange.checkpoints[0].end_index == 1
    assert new_exchange.checkpoints[0].token_count == 124000 // 2 + 20
    assert new_exchange.checkpoints[1].start_index == 1
    assert new_exchange.checkpoints[1].end_index == 3
    assert new_exchange.checkpoints[1].token_count == 30 + 40


def test_context_truncate_rewrite(exchange):
    # exchange.checkpoints.append(Checkpoint(start_index=1, end_index=2, token_count=112000+250,
    # latest_generated_tokens=50))
    exchange.messages.append(Message(role="user", content=[Text(text="user Hi again")]))
    exchange.reply()
    exchange.messages.append(Message(role="user", content=[Text(text="go AI!")]))
    exchange.reply()
    exchange.messages.append(Message(role="user", content=[Text(text="long context coming")]))
    exchange.reply()
    exchange.messages.append(Message(role="user", content=[Text(text="small_context")]))
    exchange.reply()

    # note we create a new exchange in ContextTruncate to get the system_prompt tokens.
    # In this test, it executes the sequence in the MockProvider above and thus the system prompt is 112 tokens
    # exchange.moderator.rewrite()
    assert sum(cp.token_count for cp in exchange.checkpoints) <= 112000
    assert exchange.checkpoints[0].token_count == 190
    assert exchange.checkpoints[0].latest_generated_tokens == 40
