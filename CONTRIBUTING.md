# Contributing

We welcome pull requests for general contributions. If you have a large new feature or any questions on how
to develop a fix, we recommend you open an issue before starting.

## Prerequisites

*exchange* uses [uv][uv] for dependency management, and formats with [ruff][ruff]. 

We provide a shortcut to standard commands using [just][just] in our `justfile`.

## Developing

Now that you have a local environment, you can make edits and run our tests. 

```bash
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

### Integration tests with OpenTelemetry

Exchange primarily uses http to access model providers. If you are receiving failures, it can be helpful to see traces
of the underlying HTTP requests. For example, a 404 could be indicative of an incorrect URL or a missing model.

First, ensure you have an OpenTelemetry compatible collector listening on port 4318, such as
[otel-tui](https://github.com/ymtdzzz/otel-tui).

```bash
brew install ymtdzzz/tap/otel-tui
otel-tui
```

Then, trace your integration tests like this:
```bash
uv run dotenv -f ./tests/otel.env run -- opentelemetry-instrument pytest tests -m integration
# or `just integration-otel` 

### Integration testing with Ollama

To run integration tests against Ollama, you need the model that tests expect available locally.

First, run ollama and pull the models you want to test.
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

Now, you can see failure details like this:

<img width="1694" alt="otel-tui" src="https://github.com/user-attachments/assets/711135ad-e199-438f-a175-913ab2344f17">

## Pull Requests

When opening a pull request, please ensure that your PR title adheres to the [Conventional Commits specification](https://www.conventionalcommits.org/).
This helps us maintain a consistent and meaningful changelog.

[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[just]: https://github.com/casey/just

