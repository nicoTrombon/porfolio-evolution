# Portfolio Evolution (DEGIRO CSV Tracker)

Streamlit app to track portfolio evolution and compute time-weighted return (TWR) from DEGIRO export data.

The app loads **Movements.csv** (DEGIRO account movements with ISIN and trade descriptions), optionally uses **isin_mapping.json** to map ISINs to tickers, and shows daily portfolio value, cashflows, and TWR.

## Project layout

- `src/portfolio_evolution/daily_app.py` — main Streamlit application.
- `src/portfolio_evolution/tickers.py` — ticker lookup and mapping utilities.
- `Movements.csv` — DEGIRO movements export (required; columns: Fecha/Fecha valor, ISIN, Descripción/Description).
- `isin_mapping.json` — optional ISIN → ticker mapping for price lookups.

The project uses a modern `src/` layout and `uv` for dependency management.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (e.g. `pip install uv` or install from [astral.sh](https://docs.astral.sh/uv/))

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Or via the Makefile:

```bash
make install
```

## Running the app

With `uv`:

```bash
uv run streamlit run src/portfolio_evolution/daily_app.py
```

Or with the Makefile:

```bash
make run
```

Then open the URL shown by Streamlit (usually `http://localhost:8501`).

## Development

Optional dev dependencies (in `pyproject.toml`):

- **ruff** — linting
- **black** — formatting

Install with:

```bash
uv sync --group dev
```
