.PHONY: setup test clean

# Create/update the local virtualenv and install all dependencies (including
# dev dependencies like pytest), using uv and pyproject.toml/uv.lock.
setup:
	uv sync

# Run the test suite.
test:
	uv run pytest training/tests -v

# Remove the virtualenv and Python cache artifacts.
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
