# Contributing to Kokoro-FastAPI

Always appreciate community involvement in making this project better. 

## Development Setup

We use `uv` for managing Python environments and dependencies, and `ruff` for linting and formatting.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/remsky/Kokoro-FastAPI.git
    cd Kokoro-FastAPI
    ```

2.  **Install `uv`:**
    Follow the instructions on the [official `uv` documentation](https://docs.astral.sh/uv/install/).

3.  **Create a virtual environment and install dependencies:**
    It's recommended to use a virtual environment. `uv` can create one for you. Install the base dependencies along with the `test` and `cpu` extras (needed for running tests locally).
    ```bash
    # Create and activate a virtual environment (e.g., named .venv)
    uv venv
    source .venv/bin/activate # On Linux/macOS
    # .venv\Scripts\activate # On Windows

    # Install dependencies including test requirements
    uv pip install -e ".[test,cpu]"
    ```
    *Note: If you have an NVIDIA GPU and want to test GPU-specific features locally, you can install `.[test,gpu]` instead, ensuring you have the correct CUDA toolkit installed.*

    *Note: If running via uv locally, you will have to install espeak and handle any pathing issues that arise. The Docker images handle this automatically*

4.  **Install `ruff` (if not already installed globally):**
    While `ruff` might be included via dependencies, installing it explicitly ensures you have it available.
    ```bash
    uv pip install ruff
    ```

## Running Tests

Before submitting changes, please ensure all tests pass as this is a automated requirement. The tests are run using `pytest`. 
```bash
# Make sure your virtual environment is activated
uv run pytest
```
*Note: The CI workflow runs tests using `uv run pytest api/tests/ --asyncio-mode=auto --cov=api --cov-report=term-missing`. Running `uv run pytest` locally should cover the essential checks.*

## Testing with Docker Compose

In addition to local `pytest` runs, test your changes using Docker Compose to ensure they work correctly within the containerized environment. If you aren't able to test on CUDA hardware, make note so it can be tested by another maintainer

```bash

docker compose -f docker/cpu/docker-compose.yml up --build
+
docker compose -f docker/gpu/docker-compose.yml up --build
```
This command will build the Docker images (if they've changed) and start the services defined in the respective compose file. Verify the application starts correctly and test the relevant functionality.

## Code Formatting and Linting

We use `ruff` to maintain code quality and consistency. Please format and lint your code before committing. 

1.  **Format the code:**
    ```bash
    # Make sure your virtual environment is activated
    ruff format .
    ```

2.  **Lint the code (and apply automatic fixes):**
    ```bash
    # Make sure your virtual environment is activated
    ruff check . --fix
    ```
    Review any changes made by `--fix` and address any remaining linting errors manually.

## Submitting Changes

0.  Clone the repo
1.  Create a new branch for your feature or bug fix.
2.  Make your changes, following setup, testing, and formatting guidelines above.
3.  Please try to keep your changes inline with the current design, and modular. Large-scale changes will take longer to review and integrate, and have less chance of being approved outright.
4.  Push your branch to your fork.
5.  Open a Pull Request against the `master` branch of the main repository.

Thank you for contributing!
