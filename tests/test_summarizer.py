import pytest
from exchange import Exchange, Message
from exchange.checkpoint import Checkpoint
from exchange.moderators.summarizer import ContextSummarizer
from exchange.providers import Usage


class MockProvider:
    def complete(self, model, system, messages, tools):
        assistant_message_text = "Summarized content here."
        output_tokens = len(assistant_message_text)
        total_input_tokens = sum(len(msg.text) for msg in messages)
        message = Message.assistant(assistant_message_text)
        total_tokens = total_input_tokens + output_tokens
        usage = Usage(input_tokens=total_input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
        return message, usage


@pytest.fixture
def exchange_instance():
    return Exchange(
        provider=MockProvider(),
        model="test-model",
        system="test-system",
        messages=[
            Message.user("Hi, can you help me with my homework?"),
            Message.assistant("Of course! What do you need help with?"),
            Message.user("I need help with math problems."),
            Message.assistant("Sure, I can help with that. Let's get started."),
            Message.user("Can you also help with my science homework?"),
            Message.assistant("Yes, I can help with science too."),
            Message.user("That's great! How about history?"),
            Message.assistant("Of course, I can help with history as well."),
            Message.user("Thanks! You're very helpful."),
            Message.assistant("You're welcome! I'm here to help."),
        ],
        checkpoints=[
            Checkpoint(start_index=0, end_index=2, token_count=134, latest_generated_tokens=23),
            Checkpoint(start_index=2, end_index=4, token_count=135, latest_generated_tokens=112),
            Checkpoint(start_index=4, end_index=6, token_count=143, latest_generated_tokens=31),
            Checkpoint(start_index=6, end_index=8, token_count=135, latest_generated_tokens=104),
            Checkpoint(start_index=8, end_index=10, token_count=139, latest_generated_tokens=35),
        ],
    )


@pytest.fixture
def summarizer_instance():
    return ContextSummarizer(max_tokens=500, summarization_offset=200)


def test_context_summarizer_rewrite(exchange_instance, summarizer_instance):
    # Pre-checks
    assert len(exchange_instance.messages) == 10

    # Perform summarization
    summarizer_instance.rewrite(exchange_instance)

    # Post-checks
    total_tokens = sum(cp.token_count for cp in exchange_instance.checkpoints)
    assert total_tokens <= 200

    # Assert that summarized content is the first message
    first_message = exchange_instance.messages[0]
    assert first_message.role == "user" or first_message.role == "assistant"
    assert any("summarized" in content.text.lower() for content in first_message.content)

    # Ensure roles alternate in the output
    for i in range(len(exchange_instance.messages) - 1):
        assert (
            exchange_instance.messages[i].role != exchange_instance.messages[i + 1].role
        ), "Messages must alternate between user and assistant"
