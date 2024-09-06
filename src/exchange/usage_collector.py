from collections import defaultdict
import queue
from typing import Dict

from exchange.providers.base import Usage


class UsageCollector:
    def __init__(self) -> None:
        # use thread-safe queue to store usage data from multiple threads
        # as data may be collected from multiple threads
        self.usage_data_queue = queue.Queue()

    def collect(self, model: str, usage: Usage) -> None:
        self.usage_data_queue.put((model, usage.total_tokens))

    def get_token_count_group_by_model(self) -> Dict[str, int]:
        all_usage_data = list(self.usage_data_queue.queue)
        token_count_group_by_model = defaultdict(lambda: 0)
        for model, total_tokens in all_usage_data:
            if total_tokens is not None:
                token_count_group_by_model[model] += total_tokens
        return token_count_group_by_model
