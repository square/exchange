import json
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

OPENAI_HOST = "https://api.openai.com/"


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
        tools=tools_to_openai_spec(tools) if tools else []

        tools_usage = '''
                        The tools will be parsed from json into python that is like this, including an id you will generate:                        
                        ToolUse(
                            id=tool_call["id"],
                            name=function_name,
                            parameters=json.loads(tool_call["function"]["arguments"]),
                        )
                    '''
        
        
        
        tools_message = "Here are some tools you can call and their details. If you want me to invoke them for you, return the tools at the end of your response with [TOOL_CALL] as json: " + json.dumps(tools)
        tools_message += tools_usage

        
        payload = dict(
            messages=[
                {"role": "user", "content": system},
                {"role": "user", "content": tools_message},
                *messages_to_openai_spec(messages),
            ],
            model=model,            
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
