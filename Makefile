.PHONY: install run check format

install:
	uv sync --extra dev

run:
	uv run streamlit run src/portfolio_evolution/daily_app.py

# Run before committing: lint + format check
check:
	uv run --extra dev ruff check .
	uv run --extra dev black --check .

# Fix formatting
format:
	uv run --extra dev black .
	uv run --extra dev ruff check --fix .

