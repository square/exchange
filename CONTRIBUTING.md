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

### Integration testing OpenAI with Ollama

The OpenAI provider uses the OpenAI API to access models. The OpenAI API is supported by many tools, and using this can
save you time and money when developing. One such tool is [Ollama](https://github.com/ollama/ollama).

First, run ollama and pull the models you want to test.
```bash
ollama serve
# Then in another terminal
ollama pull mistral-nemo:12B
ollama pull llava:7b
```

Now, export OpenAI variables that control the tests
```bash
export OPENAI_MODEL_TOOL=mistral-nemo
export OPENAI_MODEL_VISION=llava:7b
export OPENAI_HOST=http://localhost:11434
export OPENAI_API_KEY=unused
```

Finally, run openai integration tests against your ollama server.
```bash
uv run pytest tests -m integration -k openai
# or `just integration -k openai`
```

## Pull Requests

When opening a pull request, please ensure that your PR title adheres to the [Conventional Commits specification](https://www.conventionalcommits.org/).
This helps us maintain a consistent and meaningful changelog.

[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[just]: https://github.com/casey/just

