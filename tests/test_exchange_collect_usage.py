from unittest.mock import MagicMock
from exchange.exchange import Exchange
from exchange.message import Message
from exchange.moderators.passive import PassiveModerator
from exchange.providers.base import Provider
from exchange.tool import Tool
from exchange.usage_collector import UsageCollector

MODEL_NAME = "test-model"


def create_exchange(mock_provider, mock_usage_collector, dummy_tool):
    return Exchange(
        provider=mock_provider,
        model=MODEL_NAME,
        system="test-system",
        tools=(Tool.from_function(dummy_tool),),
        messages=[],
        moderator=PassiveModerator(),
        usage_collector=mock_usage_collector,
    )


def test_exchange_generate_collect_usage(usage_factory, dummy_tool):
    mock_provider = MagicMock(spec=Provider)
    mock_usage_collector = MagicMock(spec=UsageCollector)
    usage = usage_factory()
    mock_provider.complete.return_value = (Message.assistant("msg"), usage)
    exchange = create_exchange(mock_provider, mock_usage_collector, dummy_tool)

    exchange.generate()

    mock_usage_collector.collect.assert_called_once_with(MODEL_NAME, usage)


def test_exchange_generate_not_collect_usage_when_total_tokens_is_none(usage_factory, dummy_tool):
    mock_provider = MagicMock(spec=Provider)
    mock_usage_collector = MagicMock(spec=UsageCollector)
    mock_provider.complete.return_value = (Message.assistant("msg"), usage_factory(total_tokens=None))
    exchange = create_exchange(mock_provider, mock_usage_collector, dummy_tool)

    exchange.generate()

    mock_usage_collector.collect.assert_not_called()


def test_exchange_generate_not_collect_usage_when_total_tokens_is_0(usage_factory, dummy_tool):
    mock_provider = MagicMock(spec=Provider)
    mock_usage_collector = MagicMock(spec=UsageCollector)
    mock_provider.complete.return_value = (Message.assistant("msg"), usage_factory(total_tokens=0))
    exchange = create_exchange(mock_provider, mock_usage_collector, dummy_tool)

    exchange.generate()

    mock_usage_collector.collect.assert_not_called()
