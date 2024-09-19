from collections import defaultdict
from dataclasses import dataclass
import queue
from typing import List

from exchange.providers.base import Usage


@dataclass
class TokenUsage:
    model: str
    input_tokens: int
    output_tokens: int


class _TokenUsageCollector:
    def __init__(self) -> None:
        # use thread-safe queue to store usage data from multiple threads
        # as data may be collected from multiple threads
        self.usage_data_queue = queue.Queue()

    def collect(self, model: str, usage: Usage) -> None:
        self.usage_data_queue.put((model, usage.input_tokens, usage.output_tokens))

    def get_token_usage_group_by_model(self) -> List[TokenUsage]:
        all_usage_data = list(self.usage_data_queue.queue)
        token_count_group_by_model = defaultdict(lambda: [0, 0])
        for model, input_tokens, output_tokens in all_usage_data:
            if input_tokens is not None:
                token_count_group_by_model[model][0] += input_tokens
            if output_tokens is not None:
                token_count_group_by_model[model][1] += output_tokens
        token_usage_list = [
            TokenUsage(model, input_tokens, output_tokens)
            for model, (input_tokens, output_tokens) in token_count_group_by_model.items()
        ]
        return token_usage_list


_token_usage_collector = _TokenUsageCollector()
