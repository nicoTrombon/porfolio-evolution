.PHONY: install run

install:
	uv sync

run:
	uv run streamlit run src/portfolio_evolution/daily_app.py

