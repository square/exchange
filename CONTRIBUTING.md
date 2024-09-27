# Contributing

We welcome pull requests for general contributions. If you have a large new feature or any questions on how
to develop a fix, we recommend you open an issue before starting.

## Prerequisites

*exchange* uses [uv][uv] for dependency management, and formats with [ruff][ruff]. 

We provide a shortcut to standard commands using [just][just] in our `justfile`.

## Developing

Now that you have a local environment, you can make edits and run our tests. 

```
uv run pytest tests -m "not integration"
```

or, as a shortcut, 

```bash
just test
```

Generally if you are not developing a new provider, you can test most functionality through mocking and the normal
test suite.

However, to ensure the providers work, we also have integration tests which actually require a credential and connect
to the provider endpoints. Those can be run with

```bash
uv run pytest tests -m integration
# or `just integration`
```

### Integration testing with LocalAI

To run integration tests against LocalAI, you need the model that tests expect available locally.

First, run `local-ai` and pull the models you want to test.
```bash
local-ai run
# Then in another terminal, install the model
LOCALAI_MODEL=$(uv run python -c "from src.exchange.providers.localai import LOCALAI_MODEL; print(LOCALAI_MODEL)")
local-ai models install $LOCALAI_MODEL
```

Finally, run LocalAI integration tests.
```bash
uv run pytest tests -m integration -k localai
# or `just integration -k localai`
```

Note: The `LOCALAI_MODEL` variable controls which model is used in tests. If you want to run with a
different model, set that before invoking them.

### Integration testing with Ollama

To run integration tests against Ollama, you need the model that tests expect available locally.

First, run `ollama` and pull the models you want to test.
```bash
ollama serve
# Then in another terminal, pull the model
OLLAMA_MODEL=$(uv run python -c "from src.exchange.providers.ollama import OLLAMA_MODEL; print(OLLAMA_MODEL)")
ollama pull $OLLAMA_MODEL
```

Finally, run ollama integration tests.
```bash
uv run pytest tests -m integration -k ollama
# or `just integration -k ollama`
```

Note: The `OLLAMA_MODEL` variable controls which model is used in tests. If you want to run with a
different model, set that before invoking them.

## Pull Requests

When opening a pull request, please ensure that your PR title adheres to the [Conventional Commits specification](https://www.conventionalcommits.org/).
This helps us maintain a consistent and meaningful changelog.

[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[just]: https://github.com/casey/just

