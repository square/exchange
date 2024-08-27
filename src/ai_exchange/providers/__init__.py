from functools import cache
from typing import Type

from ai_exchange.providers.anthropic import AnthropicProvider  # noqa
from ai_exchange.providers.base import Provider, Usage  # noqa
from ai_exchange.providers.databricks import DatabricksProvider  # noqa
from ai_exchange.providers.openai import OpenAiProvider  # noqa
from ai_exchange.providers.azure import AzureProvider  # noqa
from ai_exchange.utils import load_plugins


@cache
def get_provider(name: str) -> Type[Provider]:
    return load_plugins(group="exchange.provider")[name]
