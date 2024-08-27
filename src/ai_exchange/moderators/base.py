from abc import ABC, abstractmethod
from typing import Type


class Moderator(ABC):
    @abstractmethod
    def rewrite(self, exchange: Type["ai_exchange.exchange.Exchange"]) -> None:  # noqa: F821
        pass
