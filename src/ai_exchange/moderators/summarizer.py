from typing import List, Literal, Optional, Tuple, Type, Union

from ai_exchange import Message
from ai_exchange.checkpoint import Checkpoint
from ai_exchange.moderators import ContextTruncate, Moderator, PassiveModerator

MAX_TOKENS = 112000
SUMMARIZATION_OFFSET = 40000  # Keep a max of this many tokens


def pop_checkpoint(
    exchange: Type["ai_exchange.exchange.Exchange"],  # noqa: F821
    exclude_last: Literal[0, 1] = 0,
    return_messages: bool = False,
) -> Union[Tuple[List[Message], Checkpoint], Type["ai_exchange.exchange.Exchange"]]:  # noqa: F821
    """Pop messages from the front of the list in sections

    Inputs:
        exchange (Exchange): An Exchange instance to pop checkpoints off of
        exclude_last (Literal[0,1]): integer flag whether to remove all messages in
            the checkpoint (0), or remove all except the last message (1).
    """
    removed_messages = []
    checkpoint = exchange.checkpoints.pop(0)
    for _ in range(checkpoint.end_index - checkpoint.start_index - exclude_last):
        removed_messages.append(exchange.messages.pop(checkpoint.start_index))

    # rewrite checkpoint indexes
    for cp in exchange.checkpoints:
        cp.start_index = cp.start_index - checkpoint.end_index + exclude_last
        cp.end_index = cp.end_index - checkpoint.end_index + exclude_last

    if exclude_last:
        # there is still one entry in the first checkpoint only
        checkpoint.token_count = checkpoint.latest_generated_tokens
        checkpoint.end_index = exclude_last
        exchange.checkpoints.insert(0, checkpoint)

    if return_messages:
        return removed_messages, checkpoint
    else:
        return exchange


class ContextSummarizer(Moderator):
    def __init__(
        self,
        model: Optional[str] = "gpt-4o-mini",
        max_tokens: Optional[int] = MAX_TOKENS,
        summarization_offset: Optional[int] = SUMMARIZATION_OFFSET,
    ) -> None:
        self.model = model
        self.system_prompt_token_count = None
        self.max_tokens = max_tokens
        self.summarization_offset = summarization_offset

    def rewrite(self, exchange: Type["ai_exchange.exchange.Exchange"]) -> None:  # noqa: F821
        """Summarize the context history up to the last few messages in the exchange"""
        if not self.system_prompt_token_count:
            # calculate the system prompt tokens (includes functions etc...)
            _system_token_exchange = exchange.replace(
                messages=[],
                checkpoints=[],
                moderator=PassiveModerator(),
                model=self.model if self.model else exchange.model,
            )
            _ = _system_token_exchange.generate()
            checkpoint = _system_token_exchange.checkpoints.pop()
            self.system_prompt_token_count = checkpoint.token_count - checkpoint.latest_generated_tokens

        if sum(cp.token_count for cp in exchange.checkpoints) > self.max_tokens:
            # this keeps all the messages/checkpoints
            throwaway_exchange = exchange.replace(
                moderator=PassiveModerator(),
            )

            # keep latest summarization_offset tokens in context, summarize the rest
            removed_checkpoints = 0
            messages_to_summarize = []
            while sum(cp.token_count for cp in throwaway_exchange.checkpoints) > self.summarization_offset:
                m, c = pop_checkpoint(throwaway_exchange, return_messages=True)
                messages_to_summarize.extend(m)
                removed_checkpoints += 1

            exclude_last = False
            if throwaway_exchange.messages[0].tool_result:
                m, c = pop_checkpoint(throwaway_exchange, exclude_last=1, return_messages=True)
                messages_to_summarize.extend(m)
                exclude_last = True

            if messages_to_summarize[-1].role == "assistant" and (not messages_to_summarize[-1].tool_use):
                messages_to_summarize.append(Message.user("Summarize our the above conversation"))

            summarizer_exchange = exchange.replace(
                system=Message.load("summarizer.jinja").text,
                moderator=ContextTruncate(),
                model=self.model,
                messages=messages_to_summarize,
                checkpoints=[],
            )

            summary = summarizer_exchange.reply()
            summary_checkpoint = summarizer_exchange.checkpoints[-1]

            # pop out all the messages up
            for _ in range(removed_checkpoints):
                pop_checkpoint(exchange)

            # if first message is a tool_result, it means there's no matching pair. remove it
            if exclude_last:
                pop_checkpoint(exchange, exclude_last=1)

            # insert summary as first message/checkpoint
            if exchange.messages[0].role == "assistant":
                summary_message = Message.user(summary.text)
            else:
                summary_message = Message.assistant(summary.text)

            exchange.messages.insert(0, summary_message)

            # insert a new checkpoint with the summary message
            exchange.checkpoints.insert(
                0,
                Checkpoint(
                    start_index=0,
                    end_index=1,
                    token_count=summary_checkpoint.latest_generated_tokens + self.system_prompt_token_count,
                    latest_generated_tokens=summary_checkpoint.latest_generated_tokens,
                ),
            )

            # update checkpoint indices
            for cp in exchange.checkpoints[1:]:
                cp.start_index = cp.start_index + 1
                cp.end_index = cp.end_index + 1
