from functools import cache
from typing import Type

from ai_exchange.moderators.base import Moderator
from ai_exchange.utils import load_plugins
from ai_exchange.moderators.passive import PassiveModerator  # noqa
from ai_exchange.moderators.truncate import ContextTruncate  # noqa
from ai_exchange.moderators.summarizer import ContextSummarizer  # noqa


@cache
def get_moderator(name: str) -> Type[Moderator]:
    return load_plugins(group="exchange.moderator")[name]
