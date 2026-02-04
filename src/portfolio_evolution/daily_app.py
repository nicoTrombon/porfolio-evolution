from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from portfolio_evolution.tickers import best_ticker_suggestion


def parse_euro_number(s: str) -> float:
    """
    Parse numbers like "17.555,93" -> 17555.93, "555,71" -> 555.71, "17555.93" -> 17555.93.
    """
    import re as _re

    s = str(s).strip()
    s = _re.sub(r"[^\d,.\-+]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return float("nan")


class DegiroMovementsPositionsLoader:
    """
    A partir de Movements.csv, reconstruye unidades por ISIN día a día
    (solo para operaciones de compra/venta de productos con ISIN).
    """

    import re as _re

    TRADE_REGEX = _re.compile(
        r"(compra|venta)\s+(\d+)\s+.+?@([0-9\.,]+)\s+eur",
        flags=_re.IGNORECASE,
    )

    def load_positions(self, csv_file) -> pd.DataFrame:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]

        # Fecha valor como referencia; si no existe, usamos Fecha
        date_col = "Fecha valor" if "Fecha valor" in df.columns else "Fecha"
        df["date"] = pd.to_datetime(
            df[date_col], errors="coerce", dayfirst=True
        ).dt.tz_localize(None)

        # Necesitamos ISIN y Descripción
        if "ISIN" not in df.columns:
            raise ValueError("No encuentro columna 'ISIN' en Movements.csv.")

        if "Descripción" in df.columns:
            desc_col = "Descripción"
        elif "Description" in df.columns:
            desc_col = "Description"
        else:
            raise ValueError(
                "No encuentro columna 'Descripción' ni 'Description' en Movements.csv. "
                f"Columnas: {list(df.columns)}"
            )

        isin_to_name: Dict[str, str] = {}
        trades = []
        for _, row in df.iterrows():
            desc = str(row[desc_col])
            isin = str(row["ISIN"]).strip()
            if not isin or isin == "nan":
                continue

            # Nombre de producto para mostrar en la UI (solo guardamos uno por ISIN)
            product_name = str(row.get("Producto", "")).strip()
            if isin not in isin_to_name and product_name:
                isin_to_name[isin] = product_name

            m = self.TRADE_REGEX.search(desc.lower())
            if not m:
                continue

            side_raw, qty_str, _price_str = m.groups()
            side = side_raw.lower()
            try:
                qty = int(qty_str)
            except ValueError:
                continue

            sign = 1 if "compra" in side else -1
            dt = row["date"]
            if pd.isna(dt):
                continue

            trades.append((dt.normalize(), isin, sign * qty))

        if not trades:
            # Aun así exponemos el mapping de nombres (por si queremos mostrar metadata)
            self.isin_to_name = isin_to_name
            return pd.DataFrame()

        trades_df = pd.DataFrame(trades, columns=["date", "isin", "qty_delta"])
        trades_df = (
            trades_df.groupby(["date", "isin"], as_index=False)["qty_delta"]
            .sum()
            .sort_values("date")
        )

        all_dates = pd.date_range(
            trades_df["date"].min(),
            trades_df["date"].max(),
            freq="D",
        )

        isins = sorted(trades_df["isin"].unique())
        data = {}
        for isin in isins:
            sub = trades_df.loc[
                trades_df["isin"] == isin, ["date", "qty_delta"]
            ].set_index("date")
            sub = sub.reindex(all_dates, fill_value=0.0)
            units = sub["qty_delta"].cumsum()
            data[isin] = units

        positions_df = pd.DataFrame(data, index=all_dates)
        positions_df.index.name = "date"

        if not positions_df.empty:
            mask_any = positions_df.abs().sum(axis=1) != 0
            positions_df = positions_df.loc[mask_any]

        self.isin_to_name = isin_to_name
        return positions_df


class DegiroMovementsCsvCashflowLoader:
    """
    Extrae cashflows externos (aportes/retiros) desde Movements.csv (locale ES).

    Devuelve un DataFrame con columnas: date, amount.
    """

    def load(self, csv_file) -> pd.DataFrame:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]

        # Fecha valor como referencia; si no existe, usamos Fecha
        date_col = "Fecha valor" if "Fecha valor" in df.columns else "Fecha"
        df["date"] = pd.to_datetime(
            df[date_col], errors="coerce", dayfirst=True
        ).dt.tz_localize(None)

        if "Variación" not in df.columns:
            raise ValueError(
                f"No encuentro columna 'Variación' en Movements.csv. Columnas: {list(df.columns)}"
            )

        var_idx = df.columns.get_loc("Variación")
        if var_idx + 1 >= len(df.columns):
            raise ValueError(
                "No encuentro la columna numérica de 'Variación' justo después de 'Variación' en Movements.csv. "
                f"Columnas: {list(df.columns)}"
            )

        amount_col = df.columns[var_idx + 1]
        df["amount"] = df[amount_col].map(parse_euro_number)

        if "Descripción" in df.columns:
            desc_col = "Descripción"
        elif "Description" in df.columns:
            desc_col = "Description"
        else:
            raise ValueError(
                "No encuentro columna 'Descripción' ni 'Description' en Movements.csv. "
                f"Columnas: {list(df.columns)}"
            )

        d = df[desc_col].astype(str).str.lower()

        is_deposit = d.str.contains(r"flatex deposit")
        is_transfer_in = d.str.contains(r"transferir desde su cuenta de efectivo")
        is_transfer_out = d.str.contains(r"transferir a su cuenta de efectivo")
        is_withdraw_other = d.str.contains(
            r"withdraw|reembolso|transfer out|auszahlung|uitboeking"
        )
        is_fx = d.str.contains(r"cambio de divisa")
        is_withdraw = (is_transfer_out | is_withdraw_other) & (~is_fx)
        is_cash_sweep = d.str.contains(r"cash sweep")

        # Comisiones y tasas (queremos que cuenten como cashflows negativos para el net gain)
        is_trade_cost = d.str.contains(r"costes de transacción y/o externos de degiro")
        is_connectivity_fee = d.str.contains(r"comisión de conectividad con el mercado")
        is_commission = is_trade_cost | is_connectivity_fee

        external = (is_deposit | is_transfer_in | is_withdraw | is_commission) & (
            ~is_cash_sweep
        )

        out = df.loc[external, ["date", "amount"]].dropna().copy()
        out = (
            out.groupby("date", as_index=False)["amount"]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return out


def get_daily_prices_yahoo(
    isins: List[str],
    start: date,
    end: date,
    isin_to_ticker: Dict[str, str],
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con precios diarios (Adj Close) por ISIN.
    Índice: fecha; columnas: ISIN.
    """
    if not isins:
        return pd.DataFrame()

    missing = [i for i in isins if not isin_to_ticker.get(i)]
    if missing:
        raise ValueError(
            "Faltan tickers para algunos ISIN: " + ", ".join(sorted(missing))
        )

    tickers = [isin_to_ticker[i] for i in isins]

    data = yf.download(
        tickers,
        start=start,
        end=end
        + timedelta(
            days=1
        ),  # yfinance usa end exclusivo; sumamos 1 día para incluir 'end'
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        data = data["Adj Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    data.columns = [str(c) for c in data.columns]

    mapping_back = {isin_to_ticker[i]: i for i in isins}
    data = data.rename(columns=mapping_back)

    cols_present = [c for c in data.columns if c in mapping_back.values()]
    data = data[cols_present]
    data = data.sort_index()
    data.index = data.index.tz_localize(None)
    data.index.name = "date"
    return data


ISIN_MAPPING_PATH = Path("isin_mapping.json")


def load_isin_mapping(path: Path = ISIN_MAPPING_PATH) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = f.read()
        import json

        raw = json.loads(data)
        return {str(k): str(v) for k, v in raw.items() if v}
    except Exception:
        return {}


def save_isin_mapping(mapping: Dict[str, str], path: Path = ISIN_MAPPING_PATH) -> None:
    try:
        import json

        with path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


def main() -> None:
    st.set_page_config(page_title="DEGIRO – Evolución diaria", layout="wide")
    st.title("DEGIRO → Evolución diaria de la cartera")

    st.markdown("""
Subí tu archivo **Movements.csv** de DEGIRO y te muestro un único gráfico con la
**evolución diaria estimada del valor total de tu cartera**, usando:

- posiciones diarias por ISIN reconstruidas desde Movements.csv
- precios diarios de Yahoo Finance (tickers sugeridos automáticamente por ISIN)
""")

    movements_file = st.file_uploader(
        "Subí el archivo Movements.csv (export de DEGIRO, locale ES)",
        type=["csv"],
        accept_multiple_files=False,
        key="movements_csv_daily",
    )

    positions_df = None
    isin_to_name: Dict[str, str] = {}
    cashflow_df: pd.DataFrame | None = None
    if movements_file is not None:
        try:
            raw_bytes = movements_file.getvalue()

            # Posiciones (unidades por ISIN)
            buf_pos = io.BytesIO(raw_bytes)
            loader = DegiroMovementsPositionsLoader()
            positions_df = loader.load_positions(buf_pos)
            isin_to_name = getattr(loader, "isin_to_name", {})

            # Cashflows externos (aportes / retiros) por día
            buf_cf = io.BytesIO(raw_bytes)
            cf_loader = DegiroMovementsCsvCashflowLoader()
            cf = cf_loader.load(buf_cf)
            cashflow_df = cf if not cf.empty else None
        except Exception as e:
            st.error(f"No pude procesar Movements.csv: {e}")

    if positions_df is None or positions_df.empty:
        st.info(
            "Subí Movements.csv con operaciones de compra/venta para ver la evolución diaria "
            "de tu cartera."
        )
        return

    pos_min_d = positions_df.index.min().date()
    pos_max_d = positions_df.index.max().date()

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input(
            "Desde", pos_min_d, min_value=pos_min_d, max_value=pos_max_d
        )
    with c2:
        end = st.date_input(
            "Hasta", pos_max_d, min_value=pos_min_d, max_value=date.today()
        )

    mask_pos = (positions_df.index.date >= start) & (positions_df.index.date <= end)
    positions_window = positions_df.loc[mask_pos]

    if positions_window.empty:
        st.warning("No hay posiciones en el rango seleccionado.")
        return

    # Extendemos posiciones hasta la fecha 'end', manteniendo las unidades constantes tras la última operación
    all_dates = pd.date_range(start, end, freq="D")
    positions_window = positions_window.reindex(all_dates).ffill()
    positions_window.index.name = "date"

    isins_in_positions = [
        c for c in positions_window.columns if positions_window[c].abs().sum() > 0
    ]

    st.markdown("#### Mapping ISIN → ticker (elegí qué ticker usar en Yahoo Finance)")

    existing_mapping = load_isin_mapping()

    rows = []
    for isin in isins_in_positions:
        current_ticker = existing_mapping.get(isin, "")
        display_name = isin_to_name.get(isin, "")
        suggested = best_ticker_suggestion(isin) if not current_ticker else ""
        rows.append(
            {
                "ISIN": isin,
                "Producto": display_name,
                "Ticker sugerido (Yahoo)": suggested,
                "Ticker elegido": current_ticker,
            }
        )

    mapping_df = pd.DataFrame(rows)
    edited_mapping_df = st.data_editor(
        mapping_df,
        use_container_width=True,
        num_rows="fixed",
        key="isin_mapping_editor_daily",
    )

    isin_to_ticker: Dict[str, str] = {}
    for _, row in edited_mapping_df.iterrows():
        ticker = str(row["Ticker elegido"]).strip()
        if ticker:
            isin_to_ticker[str(row["ISIN"])] = ticker

    col_save, col_info = st.columns([1, 3])
    with col_save:
        if st.button("Guardar mapping ISIN → ticker"):
            new_mapping = load_isin_mapping()
            new_mapping.update(isin_to_ticker)
            save_isin_mapping(new_mapping)
            st.success("Mapping guardado en 'isin_mapping.json'.")

    with col_info:
        st.caption(
            "Usá 'Ticker elegido' para forzar el listing correcto (bolsa/divisa). "
            "Las sugerencias se basan en el ISIN."
        )

    if not isin_to_ticker:
        st.warning(
            "Completá al menos un 'Ticker elegido' para poder descargar precios diarios."
        )
        return

    try:
        prices_df = get_daily_prices_yahoo(
            isins=isins_in_positions,
            start=start,
            end=end,
            isin_to_ticker=isin_to_ticker,
        )
    except Exception as e:
        st.error(f"No pude descargar precios desde Yahoo Finance: {e}")
        return

    if prices_df.empty:
        st.warning(
            "No se obtuvieron precios diarios (revisá los tickers elegidos o la conexión de red)."
        )
        return

    # Alineamos precios al calendario de posiciones y rellenamos huecos con forward-fill
    all_dates = positions_window.index
    prices_aligned = prices_df.reindex(all_dates).ffill()
    prices_aligned = prices_aligned[isins_in_positions]
    positions_aligned = positions_window.loc[all_dates, isins_in_positions]

    value_by_isin = positions_aligned * prices_aligned
    portfolio_daily = value_by_isin.sum(axis=1).to_frame(name="portfolio_value")
    portfolio_daily.index.name = "date"

    # KPIs básicos del período seleccionado
    def fmt_money(x: float) -> str:
        return "—" if pd.isna(x) else f"{x:,.2f}"

    initial_value = float(portfolio_daily["portfolio_value"].iloc[0])
    final_value = float(portfolio_daily["portfolio_value"].iloc[-1])
    net_cashflows_range = 0.0

    # Opcional: cashflows en el eje secundario (si los pudimos detectar)
    daily_cf = None
    if cashflow_df is not None and not cashflow_df.empty:
        cf_window = cashflow_df[
            (cashflow_df["date"].dt.date >= start)
            & (cashflow_df["date"].dt.date <= end)
        ].copy()
        if not cf_window.empty:
            net_cashflows_range = float(cf_window["amount"].sum())
            cf_window = cf_window.set_index("date").sort_index()
            cf_window = cf_window.reindex(portfolio_daily.index, fill_value=0.0)
            daily_cf = cf_window["amount"]

    # Ganancia neta en euros en el rango seleccionado:
    # valor final menos (valor inicial + aportes/retiros netos en el período)
    net_euros_gain = final_value - (initial_value + net_cashflows_range)

    st.markdown("#### Resumen del período seleccionado")
    col_current, col_net_gain = st.columns(2)
    col_current.metric("Valor actual", fmt_money(final_value))
    col_net_gain.metric("Ganancia neta (€)", fmt_money(net_euros_gain))

    # Main chart: valor diario por holding (área apilada) + cashflows (barras) en eje secundario
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Área apilada por ISIN (usamos nombre de producto en la leyenda cuando esté disponible)
    for isin in isins_in_positions:
        label = isin_to_name.get(isin, isin)
        fig.add_trace(
            go.Scatter(
                x=portfolio_daily.index,
                y=value_by_isin[isin],
                mode="lines",
                stackgroup="holdings",
                name=label,
            ),
            secondary_y=False,
        )

    if daily_cf is not None:
        fig.add_trace(
            dict(
                type="bar",
                x=daily_cf.index,
                y=daily_cf,
                name="Cashflow neto",
                opacity=0.4,
            ),
            secondary_y=True,
        )

    fig.update_layout(title_text="Valor diario estimado de la cartera (por holding)")
    fig.update_yaxes(title_text="Valor cartera", secondary_y=False)
    if daily_cf is not None:
        fig.update_yaxes(title_text="Cashflow diario", secondary_y=True, showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    # Precios diarios por ISIN
    st.markdown("#### Precios diarios por ISIN")
    melted_prices = prices_aligned.reset_index().melt(
        id_vars="date", var_name="ISIN", value_name="price"
    )
    fig_prices = px.line(
        melted_prices,
        x="date",
        y="price",
        color="ISIN",
        title="Evolución del precio por ISIN",
    )
    st.plotly_chart(fig_prices, use_container_width=True)


if __name__ == "__main__":
    main()
