import os
import re
from typing import Type, Tuple, Callable, Dict, Any

import pytest

from exchange.providers import Provider, AzureProvider, OpenAiProvider
from exchange.providers.ollama import OllamaProvider


def mark_parametrized(providers: Tuple[Type[Provider], str], expected_params: Dict[Type[Provider], Tuple[Any]] = None):
    """When expected_params is present, we assume it is a VCR test which needs
    to fake ENV variables to avoid validation failures initializing the
    provider. This is done via the provider_cls function"""

    def decorator(func: Callable):
        env_func_name = provider_cls.__name__
        # Create ids based on provider class names in lowercase without 'Provider' suffix
        ids = [cls.__name__.replace("Provider", "").lower() for cls, _ in providers]
        if expected_params is None:
            return pytest.mark.parametrize("provider_cls,model", providers, ids=ids)(func)

        # Extract expected values based on provider_cls
        def inner(provider_cls: Type[Provider], model: str) -> Tuple[Any]:
            expected_values = expected_params.get(provider_cls)
            if expected_values is None:
                raise ValueError(f"No expectations found for provider class: {provider_cls}")
            return func(provider_cls, model, *expected_values)

        return pytest.mark.parametrize("provider_cls,model", providers, ids=ids, indirect=[env_func_name])(inner)

    return decorator


@pytest.fixture(scope="function")
def provider_cls(request, monkeypatch):
    """Returns the appropriate environment setup based on the provider class."""
    provider_cls = request.param
    if provider_cls == AzureProvider:
        default_azure_env(monkeypatch)
        return provider_cls
    elif provider_cls == OllamaProvider:
        return provider_cls
    elif provider_cls == OpenAiProvider:
        default_openai_env(monkeypatch)
        return provider_cls
    else:
        raise ValueError(f"Unexpected provider: {provider_cls}")


OPENAI_API_KEY = "test_openai_api_key"
OPENAI_ORG_ID = "test_openai_org_key"
OPENAI_PROJECT_ID = "test_openai_project_id"


def default_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    This fixture prevents OpenAIProvider.from_env() from erring on missing
    environment variables.

    When running VCR tests for the first time or after deleting a cassette
    recording, set required environment variables, so that real requests don't
    fail. Subsequent runs use the recorded data, so don't need them.
    """
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", OPENAI_API_KEY)


AZURE_ENDPOINT = "https://test.openai.azure.com"
AZURE_DEPLOYMENT_NAME = "test-azure-deployment"
AZURE_API_VERSION = "2024-05-01-preview"
AZURE_API_KEY = "test_azure_api_key"


def default_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    This fixture prevents AzureProvider.from_env() from erring on missing
    environment variables.

    When running VCR tests for the first time or after deleting a cassette
    recording, set required environment variables, so that real requests don't
    fail. Subsequent runs use the recorded data, so don't need them.
    """
    if "AZURE_CHAT_COMPLETIONS_HOST_NAME" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_HOST_NAME", AZURE_ENDPOINT)
    if "AZURE_CHAT_COMPLETIONS_DEPLOYMENT_NAME" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_DEPLOYMENT_NAME", AZURE_DEPLOYMENT_NAME)
    if "AZURE_CHAT_COMPLETIONS_DEPLOYMENT_API_VERSION" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_DEPLOYMENT_API_VERSION", AZURE_API_VERSION)
    if "AZURE_CHAT_COMPLETIONS_KEY" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_KEY", AZURE_API_KEY)


@pytest.fixture(scope="module")
def vcr_config():
    """
    This scrubs sensitive data and gunzips bodies when in recording mode.

    Without this, you would leak cookies and auth tokens in the cassettes.
    Also, depending on the request, some responses would be binary encoded
    while others plain json. This ensures all bodies are human-readable.
    """
    return {
        "decode_compressed_response": True,
        "filter_headers": [
            ("authorization", "Bearer " + OPENAI_API_KEY),
            ("openai-organization", OPENAI_ORG_ID),
            ("openai-project", OPENAI_PROJECT_ID),
            ("cookie", None),
        ],
        "before_record_request": scrub_request_url,
        "before_record_response": scrub_response_headers,
    }


def scrub_request_url(request):
    """
    This scrubs sensitive request data in provider-specific way. Note that headers
    are case-sensitive!
    """
    if "openai.azure.com" in request.uri:
        request.uri = re.sub(r"https://[^/]+", AZURE_ENDPOINT, request.uri)
        request.uri = re.sub(r"/deployments/[^/]+", f"/deployments/{AZURE_DEPLOYMENT_NAME}", request.uri)
        request.headers["host"] = AZURE_ENDPOINT.replace("https://", "")
        request.headers["api-key"] = AZURE_API_KEY

    return request


def scrub_response_headers(response):
    """
    This scrubs sensitive response headers. Note they are case-sensitive!
    """
    response["headers"]["openai-organization"] = OPENAI_ORG_ID
    response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response
