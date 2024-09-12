import os
from typing import Any, Dict, List, Tuple, Type

import httpx

from exchange.message import Message
from exchange.providers.base import Provider, Usage
from tenacity import retry, wait_fixed, stop_after_attempt
from exchange.providers.utils import retry_if_status
from exchange.providers.utils import (
    messages_to_openai_spec,
    openai_response_to_message,
    openai_single_message_context_length_exceeded,
    raise_for_status,
    tools_to_openai_spec,
)
from exchange.tool import Tool

retry_procedure = retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(2),
    retry=retry_if_status(codes=[429], above=500),
    reraise=True,
)


class AzureProvider(Provider):
    """Provides chat completions for models hosted directly by OpenAI"""

    def __init__(self, client: httpx.Client, deployment_name: str, api_version: str) -> None:
        super().__init__()
        self.client = client
        self.deployment_name = deployment_name
        self.api_version = api_version

    @classmethod
    def from_env(cls: Type["AzureProvider"]) -> "AzureProvider":
        try:
            url = os.environ["AZURE_CHAT_COMPLETIONS_HOST_NAME"]
        except KeyError:
            raise RuntimeError("Failed to get AZURE_CHAT_COMPLETIONS_HOST_NAME from the environment.")

        try:
            deployment_name = os.environ["AZURE_CHAT_COMPLETIONS_DEPLOYMENT_NAME"]
        except KeyError:
            raise RuntimeError("Failed to get AZURE_CHAT_COMPLETIONS_DEPLOYMENT_NAME from the environment.")

        try:
            api_version = os.environ["AZURE_CHAT_COMPLETIONS_DEPLOYMENT_API_VERSION"]
        except KeyError:
            raise RuntimeError("Failed to get AZURE_CHAT_COMPLETIONS_DEPLOYMENT_API_VERSION from the environment.")

        try:
            key = os.environ["AZURE_CHAT_COMPLETIONS_KEY"]
        except KeyError:
            raise RuntimeError("Failed to get AZURE_CHAT_COMPLETIONS_KEY from the environment.")

        # format the url host/"openai/deployments/" + deployment_name + "/chat/completions?api-version=" + api_version
        url = f"{url}/openai/deployments/{deployment_name}"
        client = httpx.Client(
            base_url=url,
            headers={"api-key": key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(60 * 10),
        )
        return cls(client, deployment_name, api_version)

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
            tools=tools_to_openai_spec(tools) if tools else [],
            **kwargs,
        )

        payload = {k: v for k, v in payload.items() if v}
        request_url = f"{self.client.base_url}/chat/completions?api-version={self.api_version}"
        response = self._post(payload, request_url)

        # Check for context_length_exceeded error for single, long input message
        if "error" in response and len(messages) == 1:
            openai_single_message_context_length_exceeded(response["error"])

        message = openai_response_to_message(response)
        usage = self.get_usage(response)
        return message, usage

    @retry_procedure
    def _post(self, payload: dict, request_url: str) -> dict:
        response = self.client.post(request_url, json=payload)
        return raise_for_status(response).json()
