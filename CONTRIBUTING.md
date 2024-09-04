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

```
just test
```

Generally if you are not developing a new provider, you can test most functionality through mocking and the normal
test suite.

However to ensure the providers work, we also have integration tests which actually require a credential and connect
to the provider endpoints. Those can be run with

```
uv run pytest tests -m integration
# or `just integration` 
```

## Pull Requests

When opening a pull request, please ensure that your PR title adheres to the [Conventional Commits specification](https://www.conventionalcommits.org/).
This helps us maintain a consistent and meaningful changelog.

[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[just]: https://github.com/casey/just

