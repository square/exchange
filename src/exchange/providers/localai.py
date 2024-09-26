import os
from typing import Type

import httpx

from exchange.providers.openai import OpenAiProvider

LOCALAI_HOST = "http://localhost:8080/"
LOCALAI_MODEL = "mistral-7b-instruct-v0.3"


class LocalAIProvider(OpenAiProvider):
    """Provides chat completions for models hosted by LocalAI"""

    __doc__ += f"""

Here's an example profile configuration to try:

    localai:
      provider: localai
      processor: {LOCALAI_HOST}
      accelerator: {LOCALAI_MODEL}
      moderator: passive
      toolkits:
      - name: developer
        requires: {{}}
"""

    def __init__(self, client: httpx.Client) -> None:
        print("PLEASE NOTE: the localai provider is experimental, use with care")
        super().__init__(client)

    @classmethod
    def from_env(cls: Type["LocalAIProvider"]) -> "LocalAIProvider":
        url = os.environ.get("LOCALAI_HOST", LOCALAI_HOST)
        client = httpx.Client(
            base_url=url,
            timeout=httpx.Timeout(60 * 10),
        )
        # from_env is expected to fail if provider is not available
        # so we run a quick test that the endpoint is running
        client.get("readyz")
        return cls(client)
