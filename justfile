# This is the default recipe when no arguments are provided
[private]
default:
  @just --list --unsorted

test *FLAGS:
  uv run pytest tests -m "not integration" {{FLAGS}}

integration *FLAGS:
  uv run pytest tests -m integration {{FLAGS}}

integration-otel *FLAGS:
  uv run dotenv -f ./tests/otel.env run -- opentelemetry-instrument pytest tests -m integration {{FLAGS}}

format:
  ruff check . --fix

coverage *FLAGS:
  uv run coverage run -m pytest tests -m "not integration" {{FLAGS}}
  uv run coverage report
  uv run coverage lcov -o lcov.info
