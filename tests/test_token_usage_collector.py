from exchange.token_usage_collector import _TokenUsageCollector, TokenUsage

def test_collect(usage_factory):
    usage_collector = _TokenUsageCollector()
    usage_collector.collect("model1", usage_factory(input_tokens=100, output_tokens=1000))
    usage_collector.collect("model1", usage_factory(input_tokens=200, output_tokens=2000))
    usage_collector.collect("model2", usage_factory(input_tokens=400, output_tokens=4000))
    usage_collector.collect("model3", usage_factory(input_tokens=500, output_tokens=5000))
    usage_collector.collect("model3", usage_factory(input_tokens=600, output_tokens=6000))
    assert usage_collector.get_token_usage_group_by_model() == [
        TokenUsage(model="model1", input_tokens=300, output_tokens=3000),
        TokenUsage(model="model2", input_tokens=400, output_tokens=4000),
        TokenUsage(model="model3", input_tokens=1100, output_tokens=11000),
    ]


def test_collect_with_non_input_or_output_token(usage_factory):
    usage_collector = _TokenUsageCollector()
    usage_collector.collect("model1", usage_factory(input_tokens=100, output_tokens=None))
    usage_collector.collect("model1", usage_factory(input_tokens=None, output_tokens=2000))
    assert usage_collector.get_token_usage_group_by_model() == [
        TokenUsage(model="model1", input_tokens=100, output_tokens=2000),
    ]
