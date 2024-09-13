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
from exchange.content import ToolUse, ToolResult, Text


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

        resoning_model = self.get_reasoning_model()
        if resoning_model and len(self.tool_use(message)) == 0:
            if len(data["choices"][0]["message"]["content"]) > 100:                
                print("---> using deep reasoning")
                payload = dict(
                    messages=[                        
                        *self.messages_filtered(messages),
                    ],
                    model=resoning_model,
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

    def tool_use(self, message):
        """ checks if the returned message is asking for tool usage or not """
        return [content for content in message.content if isinstance(content, ToolUse) or isinstance(content, ToolResult)]

    @retry_httpx_request()
    def _send_request(self, payload: Any) -> httpx.Response:  # noqa: ANN401
        return self.client.post("v1/chat/completions", json=payload)


    def messages_filtered(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """This is for models that don't handle tool call output or images directly"""
        messages_spec = []
        for message in messages:
            converted = {"role": "user"}
            output = []
            for content in message.content:
                if isinstance(content, Text):
                    converted["content"] = content.text
                elif isinstance(content, ToolResult):
                    output.append(
                        {
                            "role": "user",
                            "content": content.output,
                        }
                    )

            if "content" in converted or "tool_calls" in converted:
                output = [converted] + output
            messages_spec.extend(output)
        return messages_spec
    
    def get_reasoning_model(self):
        return os.environ.get("OPENAI_REASONING", None)

