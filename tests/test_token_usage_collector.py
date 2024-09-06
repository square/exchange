from exchange.token_usage_collector import TokenUsageCollector


def test_collect(usage_factory):
    usage_collector = TokenUsageCollector()
    usage_collector.collect("model1", usage_factory(total_tokens=100))
    usage_collector.collect("model1", usage_factory(total_tokens=200))
    usage_collector.collect("model2", usage_factory(total_tokens=400))
    usage_collector.collect("model3", usage_factory(total_tokens=500))
    usage_collector.collect("model3", usage_factory(total_tokens=600))
    assert usage_collector.get_token_count_group_by_model() == {
        "model1": 300,
        "model2": 400,
        "model3": 1100,
    }


def test_collect_with_non_total_token(usage_factory):
    usage_collector = TokenUsageCollector()
    usage_collector.collect("model1", usage_factory(total_tokens=100))
    usage_collector.collect("model1", usage_factory(total_tokens=None))
    assert usage_collector.get_token_count_group_by_model() == {
        "model1": 100,
    }
