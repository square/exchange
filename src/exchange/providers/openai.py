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
from typing import Generator

OPENAI_HOST = "https://api.openai.com/"


class LogResponse(httpx.Response):
    def iter_bytes(self, *args: object, **kwargs: Dict[str, object]) -> Generator[bytes, None, None]:
        print("Response content:")
        for chunk in super().iter_bytes(*args, **kwargs):
            print(chunk.decode('utf-8'))
            yield chunk

class LogTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport) -> None:
        self.transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Log the request details
        print(f"Request method: {request.method}")
        print(f"Request URL: {request.url}")
        print(f"Request headers: {request.headers}")
        print(f"Request content: {request.content}")

        response = self.transport.handle_request(request)

        return LogResponse(
            status_code=response.status_code,
            headers=response.headers,
            stream=response.stream,
            extensions=response.extensions,
        )


class OpenAiProvider(Provider):
    """Provides chat completions for models hosted directly by OpenAI"""

    def __init__(self, client: httpx.Client) -> None:
        super().__init__()
        self.client = client

    @classmethod
    def from_env(cls: Type["OpenAiProvider"]) -> "OpenAiProvider":
        url = os.environ.get("OPENAI_HOST", OPENAI_HOST)
        try:
            key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise RuntimeError(
                "Failed to get OPENAI_API_KEY from the environment, see https://platform.openai.com/docs/api-reference/api-keys"
            )
        client = httpx.Client(
            base_url=url,
            auth=("Bearer", key),
            timeout=httpx.Timeout(60 * 10),
            transport=LogTransport(httpx.HTTPTransport()),
        )
        return cls(client)

    @staticmethod
    def get_usage(data: dict) -> Usage:
        usage = data.pop("usage")
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

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
