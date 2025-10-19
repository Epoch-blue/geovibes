# Contributing

We welcome contributions to this project! Please take a moment to review these guidelines before getting started.

## How to Report Bugs

- If you find a bug, please open an issue on GitHub.
- Provide a clear and concise description of the bug.
- Include steps to reproduce the bug, if possible.
- Specify your environment (e.g., operating system, Python version).

## How to Suggest Enhancements

- If you have an idea for an enhancement, please open an issue on GitHub.
- Provide a clear and concise description of the enhancement.
- Explain why this enhancement would be useful.

## Style Guidelines

- **Python code:** Formatting and linting is enforced with [Ruff](https://docs.astral.sh/ruff/). Follow PEP 8 conventions where applicable.
- **Line length:** Keep lines under 88 characters to match the Ruff and Black configuration.
- **Comments:** Write clear and concise comments to explain complex code sections.
- **Commit messages:** Write clear and descriptive commit messages. Start with a capitalized imperative verb (e.g., "Fix: ...", "Feat: ...", "Docs: ...").

## Pre-commit Hooks

We rely on pre-commit to run Ruff automatically and keep the codebase consistent.

1. Ensure dependencies are installed (e.g., `uv sync` or `pip install -e .` in your virtual environment).
2. Install the hooks once per clone:

   ```bash
   pre-commit install
   ```

3. Run the hooks across the full codebase before submitting a pull request:

   ```bash
   pre-commit run --all-files
   ```

The hooks will format code and report lint failures. Please fix any issues they flag before opening a PR.

## Testing

- Include tests for any new features or bug fixes.
- Ensure all existing tests pass before submitting your changes.
- Describe how to run the tests in your pull request.

## How to Submit Changes

1.  **Fork the repository.**
2.  **Create a new branch** for your changes (e.g., `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix`).
3.  **Make your changes.**
4.  **Test your changes thoroughly.**
5.  **Commit your changes** with a clear and descriptive commit message.
6.  **Push your changes** to your forked repository.
7.  **Open a pull request** to the main repository.
    - Provide a clear title and description for your pull request.
    - Explain the changes you've made and why.
    - Link to any relevant issues.

Thank you for contributing!
