from typing import Optional
from exchange.providers.base import Usage
from exchange.usage_collector import UsageCollector


def create_usage(total_tokens: Optional[int]) -> Usage:
    return Usage(input_tokens=100, output_tokens=200, total_tokens=total_tokens)


def test_collect():
    usage_collector = UsageCollector()
    usage_collector.collect("model1", create_usage(100))
    usage_collector.collect("model1", create_usage(200))
    usage_collector.collect("model2", create_usage(400))
    usage_collector.collect("model3", create_usage(500))
    usage_collector.collect("model3", create_usage(600))
    assert usage_collector.get_token_count_group_by_model() == {
        "model1": 300,
        "model2": 400,
        "model3": 1100,
    }


def test_collect_with_non_total_token():
    usage_collector = UsageCollector()
    usage_collector.collect("model1", create_usage(100))
    usage_collector.collect("model1", create_usage(None))
    assert usage_collector.get_token_count_group_by_model() == {
        "model1": 100,
    }
