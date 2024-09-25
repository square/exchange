import os
from typing import Type

import httpx

from exchange.providers.openai import OpenAiProvider

OLLAMA_HOST = "http://localhost:11434/"
OLLAMA_MODEL = "mistral-nemo"


class OllamaProvider(OpenAiProvider):
    """Provides chat completions for models hosted by Ollama"""

    __doc__ += f"""

Here's an example profile configuration to try:

    ollama:
      provider: ollama
      processor: {OLLAMA_MODEL}
      accelerator: {OLLAMA_MODEL}
      moderator: passive
      toolkits:
      - name: developer
        requires: {{}}
"""

    def __init__(self, client: httpx.Client) -> None:
        print("PLEASE NOTE: the ollama provider is experimental, use with care")
        super().__init__(client)

    @classmethod
    def from_env(cls: Type["OllamaProvider"]) -> "OllamaProvider":
        url = os.environ.get("OLLAMA_HOST", OLLAMA_HOST)
        client = httpx.Client(
            base_url=url,
            timeout=httpx.Timeout(60 * 10),
        )
        # from_env is expected to fail if required ENV variables are not
        # available. Since this provider can run with defaults, we substitute
        # a health check to verify the endpoint is running.
        client.get("")
        # The OpenAI API is defined after "v1/", so we need to join it here.
        client.base_url = client.base_url.join("v1/")
        return cls(client)
