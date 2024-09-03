import os
from typing import Any, Dict, List, Tuple, Type

import httpx

from exchange import Message, Tool
from exchange.content import Text, ToolResult, ToolUse
from exchange.providers.base import Provider, Usage
from exchange.providers.retry_with_back_off_decorator import retry_httpx_request
from exchange.providers.utils import raise_for_status

ANTHROPIC_HOST = "https://api.anthropic.com/v1/messages"


class AnthropicProvider(Provider):
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    @classmethod
    def from_env(cls: Type["AnthropicProvider"]) -> "AnthropicProvider":
        url = os.environ.get("ANTHROPIC_HOST", ANTHROPIC_HOST)
        try:
            key = os.environ["ANTHROPIC_API_KEY"]
        except KeyError:
            raise RuntimeError("Failed to get ANTHROPIC_API_KEY from the environment")
        client = httpx.Client(
            base_url=url,
            headers={
                "x-api-key": key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            timeout=httpx.Timeout(60 * 10),
        )
        return cls(client)

    @staticmethod
    def get_usage(data: Dict) -> Usage:  # noqa: ANN401
        usage = data.get("usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def anthropic_response_to_message(response: Dict) -> Message:
        content_blocks = response.get("content", [])
        content = []
        for block in content_blocks:
            if block["type"] == "text":
                content.append(Text(text=block["text"]))
            elif block["type"] == "tool_use":
                content.append(
                    ToolUse(
                        id=block["id"],
                        name=block["name"],
                        parameters=block["input"],
                    )
                )
        return Message(role="assistant", content=content)

    @staticmethod
    def tools_to_anthropic_spec(tools: Tuple[Tool]) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    @staticmethod
    def messages_to_anthropic_spec(messages: List[Message]) -> List[Dict[str, Any]]:
        messages_spec = []
        # if messages is empty - just make a default
        for message in messages:
            converted = {"role": message.role}
            for content in message.content:
                if isinstance(content, Text):
                    converted["content"] = [{"type": "text", "text": content.text}]
                elif isinstance(content, ToolUse):
                    converted.setdefault("content", []).append(
                        {
                            "type": "tool_use",
                            "id": content.id,
                            "name": content.name,
                            "input": content.parameters,
                        }
                    )
                elif isinstance(content, ToolResult):
                    converted.setdefault("content", []).append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.tool_use_id,
                            "content": content.output,
                        }
                    )
            messages_spec.append(converted)
        if len(messages_spec) == 0:
            converted = {
                "role": "user",
                "content": [{"type": "text", "text": "Ignore"}],
            }
            messages_spec.append(converted)
        return messages_spec

    def complete(
        self,
        model: str,
        system: str,
        messages: List[Message],
        tools: List[Tool] = [],
        **kwargs: Dict[str, Any],
    ) -> Tuple[Message, Usage]:
        tools_set = set()
        unique_tools = []
        for tool in tools:
            if tool.name not in tools_set:
                unique_tools.append(tool)
                tools_set.add(tool.name)

        payload = dict(
            system=system,
            model=model,
            max_tokens=4096,
            messages=self.messages_to_anthropic_spec(messages),
            tools=self.tools_to_anthropic_spec(tuple(unique_tools)),
            **kwargs,
        )
        payload = {k: v for k, v in payload.items() if v}

        response = self._send_request(payload)

        response_data = raise_for_status(response).json()
        message = self.anthropic_response_to_message(response_data)
        usage = self.get_usage(response_data)

        return message, usage

    @retry_httpx_request()
    def _send_request(self, payload: Dict[str, Any]) -> httpx.Response:
        return self.client.post(ANTHROPIC_HOST, json=payload)
