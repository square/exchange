from abc import ABC, abstractmethod
from attrs import define, field
from typing import List, Tuple, Type

from ai_exchange.message import Message
from ai_exchange.tool import Tool


@define
class Usage:
    input_tokens: int = field(factory=None)
    output_tokens: int = field(default=None)
    total_tokens: int = field(default=None)


class Provider(ABC):
    @classmethod
    def from_env(cls: Type["Provider"]) -> "Provider":
        return cls()

    @abstractmethod
    def complete(
        self,
        model: str,
        system: str,
        messages: List[Message],
        tools: Tuple[Tool],
    ) -> Tuple[Message, Usage]:
        """Generate the next message using the specified model"""
        pass
