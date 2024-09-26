# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.3] - 2024-09-25

- test: reduce code redundancy in openai based test
- fix: improve ollama workflow from CI (#53)
- ci: enables ollama integration tests (#23)
- feat: convert openai related tests to VCR (#50)

## [0.9.2] - 2024-09-20

- feat: collect total token usages (#32)

## [0.9.1] - 2024-09-15

- fix: retry only some 400s and raise error details

## [0.9.0] - 2024-09-09

- chore: add just command for releases and update pyproject for changelog (#43)
- feat: convert ollama provider to an openai configuration (#34)
- fix: Bedrock Provider request (#29)
- test: Update truncate and summarize tests to check for sytem prompt (#42)
- chore: update test_tools to read a file instead of get a password (#38)
- fix: Use placeholder message to check tokens (#41)
- feat: rewind to user message (#30)
- chore: Update LICENSE (#40)
- fix: shouldn't hardcode truncate to gpt4o mini (#35)
- ci: enforce PR title follows conventional commit (#6)
- chore: Apply ruff and add to CI (#27)

## [0.8.4] - 2024-09-02

- Catch any HTTP errors the provider emits and retry the call to `generate` with different messages

## [0.8.3] - 2024-09-02

- Refactor checkpoints to allow exchange to stay in-sync across messages and checkpoints
- Fix typos
- Add retry to providers when sending HTTP requests

## [0.8.0] - 2024-08-23

### Added

- Initial open source release
- Includes Exchange, Message, Tool, Moderator, and Provider
- Supported providers: Anthropic, Bedrock, Databricks, OpenAI
