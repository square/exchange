import os
from typing import Any, Dict, List, Tuple, Type

import httpx

from exchange.message import Message
from exchange.providers.base import Provider, Usage
from exchange.providers.retry_with_back_off_decorator import retry_httpx_request
from exchange.providers.utils import (
    messages_to_openai_spec,
    openai_response_to_message,
    openai_single_message_context_length_exceeded,
    raise_for_status,
    tools_to_openai_spec,
)
from exchange.tool import Tool

OLLAMA_HOST = "http://localhost:11434/"


#
# NOTE: this is experimental, best used with 70B model or larger if you can.
# Example profile config to try:
class OllamaProvider(Provider):
    """Provides chat completions for models hosted by Ollama"""

    """

        ollama:
          provider: ollama
          processor: llama3.1
          accelerator: llama3.1
          moderator: passive
          toolkits:
          - name: developer
            requires: {}
    """

    def __init__(self, client: httpx.Client) -> None:
        print("PLEASE NOTE: the ollama provider is experimental, use with care")
        super().__init__()
        self.client = client

    @classmethod
    def from_env(cls: Type["OllamaProvider"]) -> "OllamaProvider":
        url = os.environ["OLLAMA_HOST"]
        client = httpx.Client(
            base_url=url,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(60 * 10),
        )
        return cls(client)

    @staticmethod
    def get_usage(data: dict) -> Usage:
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def complete(
        self,
        model: str,
        system: str,
        messages: List[Message],
        tools: Tuple[Tool],
        **kwargs: Dict[str, Any],
    ) -> Tuple[Message, Usage]:
        payload = dict(
            messages=[
                {"role": "system", "content": system},
                *messages_to_openai_spec(messages),
            ],
            model=model,
            tools=tools_to_openai_spec(tools) if tools else [],
            **kwargs,
        )
        payload = {k: v for k, v in payload.items() if v}
        response = self._send_request(payload)

        # Check for context_length_exceeded error for single, long input message
        if "error" in response.json() and len(messages) == 1:
            openai_single_message_context_length_exceeded(response.json()["error"])

        data = raise_for_status(response).json()

        message = openai_response_to_message(data)
        usage = self.get_usage(data)
        return message, usage

    @retry_httpx_request()
    def _send_request(self, payload: Any) -> httpx.Response:  # noqa: ANN401
        return self.client.post("v1/chat/completions", json=payload)
