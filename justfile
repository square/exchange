# This is the default recipe when no arguments are provided
[private]
default:
  @just --list --unsorted

test *FLAGS:
  uv run pytest tests -m "not integration" {{FLAGS}}

integration *FLAGS:
  uv run pytest tests -m integration {{FLAGS}}

format:
  uvx ruff check --fix && uvx ruff format

coverage *FLAGS:
  uv run coverage run -m pytest tests -m "not integration" {{FLAGS}}
  uv run coverage report
  uv run coverage lcov -o lcov.info

# bump project version, push, create pr
release version:
  uvx --from=toml-cli toml set --toml-path=pyproject.toml project.version {{version}}
  git co -b release-version-{{version}}
  git add pyproject.toml
  git commit -m "chore(release): release version {{version}}"

tag:
  current_version=`grep 'version' pyproject.toml | cut -d '"' -f 2`
  tag_name="v${current_version}"
  git tag ${tag_name}

# this will kick of ci for release
# use this when release branch is merged to main
tag-push:
  just tag
  tag_name=`git describe --tags --abbrev=0`
  git push origin tag ${tag_name}
