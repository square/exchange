from typing import Literal, Optional, Type

from ai_exchange.moderators import PassiveModerator
from ai_exchange.moderators.base import Moderator

MAX_TOKENS = 112000


def pop_checkpoint(
    exchange: Type["exchange.exchange.Exchange"],  # noqa: F821
    exclude_last: Literal[0, 1] = 0,
) -> Type["exchange.exchange.Exchange"]:  # noqa: F821
    """Pop messages from the front of the list in sections"""
    checkpoint = exchange.checkpoints.pop(0)
    for _ in range(checkpoint.end_index - checkpoint.start_index - exclude_last):
        exchange.messages.pop(checkpoint.start_index)

    # rewrite checkpoint indexes
    for cp in exchange.checkpoints:
        cp.start_index = cp.start_index - checkpoint.end_index + exclude_last
        cp.end_index = cp.end_index - checkpoint.end_index + exclude_last

    if exclude_last:
        # there is still one entry in the first checkpoint only
        checkpoint.token_count = checkpoint.latest_generated_tokens
        checkpoint.end_index = exclude_last
        exchange.checkpoints.insert(0, checkpoint)

    return exchange


class ContextTruncate(Moderator):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model
        self.system_prompt_token_count = None

    def rewrite(self, exchange: Type["exchange.exchange.Exchange"]) -> None:  # noqa: F821
        """Truncate the exchange messages with a FIFO strategy."""
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

        while sum(cp.token_count for cp in exchange.checkpoints) > MAX_TOKENS:
            exchange = pop_checkpoint(exchange)

            # if first message is a tool_result, it means there's no matching pair. remove it
            if exchange.messages[0].tool_result:
                pop_checkpoint(exchange, exclude_last=1)

            # Update the token count on the the first checkpoint to reflect the system prompt
            exchange.checkpoints[0].token_count += self.system_prompt_token_count
