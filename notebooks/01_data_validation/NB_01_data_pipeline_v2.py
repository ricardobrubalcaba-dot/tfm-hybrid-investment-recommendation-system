# %%
# =============================================================================
# NB-01 v2: Adquisición, limpieza y enriquecimiento técnico de datos OHLCV
# =============================================================================
# TFM: Sistema Híbrido de Recomendación de Inversión
# Autor: Ricardo Benítez Rubalcaba
# Formato: módulo .py compatible con Jupyter (importable o ejecutable)
#
# Objetivo del NB-01:
# 1) Descargar datos OHLCV diarios de un universo de tickers
# 2) Limpiar y estandarizar los datos
# 3) Construir OHLC ajustado de forma explícita
# 4) Calcular indicadores técnicos base del MVP
# 5) Exportar datasets raw / processed / quality reports
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

try:
    from ta.trend import IchimokuIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange
except ImportError as exc:
    raise ImportError(
        "Falta la librería 'ta'. Instálala antes de ejecutar este notebook: pip install ta"
    ) from exc

warnings.filterwarnings("ignore")
pd.options.display.width = 160
pd.options.display.max_columns = 200


# %%
# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEFAULT_TICKERS: List[str] = [
    # Tecnología (13)
    "AAPL", "MSFT", "NVDA", "META", "AVGO", "CSCO", "ACN",
    "CRM", "ADBE", "ORCL", "TXN", "AMAT", "QCOM",
    # Salud (7)
    "JNJ", "UNH", "LLY", "MRK", "TMO", "ABT", "PFE",
    # Finanzas (4)
    "JPM", "V", "MA", "BAC",
    # Consumo (10)
    "AMZN", "PG", "PEP", "KO", "HD", "WMT", "MCD", "NKE", "DIS", "NFLX",
    # Energía (2)
    "XOM", "CVX",
    # Industriales / materiales (1)
    "LIN",
    # Conglomerados (1)
    "BRK-B",
    # Automotriz / growth (1)
    "TSLA",
    # Búsqueda / publicidad (1)
    "GOOGL",
]


@dataclass
class PipelineConfig:
    # Universo y horizonte
    tickers: List[str] = None
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    benchmark_ticker: str = "SPY"

    # Estructura de carpetas
    project_root: Optional[str] = None
    raw_subdir: str = "data/raw"
    processed_subdir: str = "data/processed"
    reports_subdir: str = "data/reports"
    figures_subdir: str = "data/figures"

    # Descarga
    interval: str = "1d"
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0

    # Calidad y limpieza
    min_rows_required: int = 500
    min_coverage_ratio: float = 0.95
    drop_rows_if_core_ohlc_missing: bool = True

    # Exportación
    save_individual_csvs: bool = True
    save_master_parquet: bool = True
    save_master_csv: bool = True
    save_quality_report: bool = True
    make_validation_plot: bool = True
    example_plot_ticker: str = "AAPL"

    # Criterio operacional del NB-01
    minimum_processed_tickers_done: int = 30

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = DEFAULT_TICKERS.copy()


# %%
# =============================================================================
# UTILIDADES DE RUTAS Y FORMATO
# =============================================================================

def resolve_project_root(project_root: Optional[str] = None) -> Path:
    """
    Resuelve una raíz de proyecto robusta tanto para Jupyter como para script.

    Prioridad:
    1) project_root explícito
    2) directorio de trabajo actual
    """
    if project_root:
        return Path(project_root).expanduser().resolve()
    return Path.cwd().resolve()


def build_paths(cfg: PipelineConfig) -> Dict[str, Path]:
    root = resolve_project_root(cfg.project_root)
    paths = {
        "root": root,
        "raw": root / cfg.raw_subdir,
        "processed": root / cfg.processed_subdir,
        "reports": root / cfg.reports_subdir,
        "figures": root / cfg.figures_subdir,
    }
    for path in paths.values():
        if path != root:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_ticker_name(ticker: str) -> str:
    return ticker.replace("/", "-").replace(" ", "_")


# %%
# =============================================================================
# DESCARGA Y ESTANDARIZACIÓN OHLCV
# =============================================================================

def download_single_ticker(ticker: str, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Descarga OHLCV sin auto-adjust para preservar OHLC bruto,
    y luego derivar OHLC ajustado de manera explícita.
    """
    last_error = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=cfg.start_date,
                end=cfg.end_date,
                interval=cfg.interval,
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise ValueError("DataFrame vacío")
            return df
        except Exception as exc:  # pragma: no cover - defensivo
            last_error = exc
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_sleep_seconds)
    raise RuntimeError(f"No se pudo descargar {ticker}: {last_error}")


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: str(col).strip().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=rename_map).copy()

    expected_cols = [
        "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Dividends", "Stock_Splits"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan if col != "Volume" else 0

    df = df[expected_cols]
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    numeric_cols = [c for c in df.columns if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def explicit_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye OHLC ajustado de forma explícita.
    Si Adj_Close no está disponible, usa factor 1.0.
    """
    df = df.copy()
    close = df["Close"].replace(0, np.nan)
    adj_close = df["Adj_Close"].copy()
    factor = (adj_close / close).replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(1.0)

    df["Adj_Factor"] = factor
    df["Open_Adj"] = df["Open"] * df["Adj_Factor"]
    df["High_Adj"] = df["High"] * df["Adj_Factor"]
    df["Low_Adj"] = df["Low"] * df["Adj_Factor"]
    df["Close_Adj"] = df["Close"] * df["Adj_Factor"]
    return df


def clean_ohlcv(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Limpieza mínima reproducible sin forward-fill artificial sobre OHLC.
    """
    df = df.copy()
    report = {
        "rows_before_clean": int(len(df)),
        "duplicate_dates_removed": 0,
        "rows_dropped_core_ohlc_missing": 0,
        "rows_after_clean": 0,
        "coverage_ratio": np.nan,
        "missing_pct_after_clean": np.nan,
    }

    # Duplicados (por seguridad, aunque ya se manejan en standardize)
    duplicated_count = int(df.index.duplicated(keep="last").sum())
    if duplicated_count:
        df = df[~df.index.duplicated(keep="last")]
    report["duplicate_dates_removed"] = duplicated_count

    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_core_mask = df[core_cols].isna().any(axis=1)
    dropped_core = int(missing_core_mask.sum())
    if cfg.drop_rows_if_core_ohlc_missing and dropped_core > 0:
        df = df.loc[~missing_core_mask].copy()
    report["rows_dropped_core_ohlc_missing"] = dropped_core

    # Volumen negativo no tiene sentido
    df.loc[df["Volume"] < 0, "Volume"] = np.nan
    df["Volume"] = df["Volume"].fillna(0)

    # Indicador operativo útil
    df["Has_Corporate_Action"] = (
        df["Dividends"].fillna(0).ne(0) | df["Stock_Splits"].fillna(0).ne(0)
    ).astype(int)

    report["rows_after_clean"] = int(len(df))
    report["coverage_ratio"] = float(len(df) / report["rows_before_clean"]) if report["rows_before_clean"] else 0.0
    report["missing_pct_after_clean"] = float(df.isna().mean().mean() * 100)
    return df, report


# %%
# =============================================================================
# INDICADORES TÉCNICOS
# =============================================================================

def calculate_heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open_Adj"] + df["High_Adj"] + df["Low_Adj"] + df["Close_Adj"]) / 4.0
    ha["HA_Open"] = np.nan

    if len(ha) > 0:
        ha.iloc[0, ha.columns.get_loc("HA_Open")] = (df["Open_Adj"].iloc[0] + df["Close_Adj"].iloc[0]) / 2.0
        for i in range(1, len(ha)):
            ha.iloc[i, ha.columns.get_loc("HA_Open")] = (ha["HA_Open"].iloc[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2.0

    ha["HA_High"] = pd.concat([df["High_Adj"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"] = pd.concat([df["Low_Adj"], ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)
    ha["HA_Alcista"] = (ha["HA_Close"] > ha["HA_Open"]).astype(int)
    return ha


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # Ichimoku
    ichimoku = IchimokuIndicator(
        high=result["High_Adj"],
        low=result["Low_Adj"],
        window1=9,
        window2=26,
        window3=52,
        visual=False,
        fillna=False,
    )
    result["Ichimoku_Tenkan"] = ichimoku.ichimoku_conversion_line()
    result["Ichimoku_Kijun"] = ichimoku.ichimoku_base_line()
    result["Ichimoku_Senkou_A"] = ichimoku.ichimoku_a()
    result["Ichimoku_Senkou_B"] = ichimoku.ichimoku_b()

    # Derivadas útiles para NB-02
    kumo_top = np.maximum(result["Ichimoku_Senkou_A"], result["Ichimoku_Senkou_B"])
    kumo_bottom = np.minimum(result["Ichimoku_Senkou_A"], result["Ichimoku_Senkou_B"])
    result["Kumo_Top"] = kumo_top
    result["Kumo_Bottom"] = kumo_bottom
    result["Price_Above_Kumo"] = (result["Close_Adj"] > result["Kumo_Top"]).astype("Int64")

    # MACD
    macd = MACD(close=result["Close_Adj"], window_slow=26, window_fast=12, window_sign=9)
    result["MACD_Linea"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()
    result["MACD_Histograma"] = macd.macd_diff()
    result["MACD_Bullish"] = (result["MACD_Linea"] > result["MACD_Signal"]).astype("Int64")

    # RSI
    rsi = RSIIndicator(close=result["Close_Adj"], window=14)
    result["RSI"] = rsi.rsi()

    # Estocástico
    stoch = StochasticOscillator(
        high=result["High_Adj"],
        low=result["Low_Adj"],
        close=result["Close_Adj"],
        window=14,
        smooth_window=3,
    )
    result["Estocastico_K"] = stoch.stoch()
    result["Estocastico_D"] = stoch.stoch_signal()

    # ATR (sobre OHLC ajustado)
    atr = AverageTrueRange(
        high=result["High_Adj"],
        low=result["Low_Adj"],
        close=result["Close_Adj"],
        window=14,
    )
    result["ATR"] = atr.average_true_range()
    result["ATR_Normalizado"] = result["ATR"] / result["Close_Adj"].replace(0, np.nan)

    # Volumen relativo
    result["Vol_Media_20"] = result["Volume"].rolling(window=20, min_periods=20).mean()
    result["Volumen_Relativo"] = result["Volume"] / result["Vol_Media_20"]

    # Heiken Ashi
    ha = calculate_heiken_ashi(result)
    result = pd.concat([result, ha], axis=1)

    # Metadatos útiles
    result["Ticker"] = result.get("Ticker", np.nan)
    result["Fecha"] = result.index
    return result


# %%
# =============================================================================
# CONTROL DE CALIDAD POR TICKER
# =============================================================================

def build_quality_manifest_row(
    ticker: str,
    df_raw: pd.DataFrame,
    df_processed: Optional[pd.DataFrame],
    clean_report: Dict[str, float],
    cfg: PipelineConfig,
    error_message: str = "",
) -> Dict[str, object]:
    status = "OK"
    if error_message:
        status = "ERROR"
    elif df_processed is None or len(df_processed) < cfg.min_rows_required:
        status = "INSUFICIENTE"
    elif clean_report.get("coverage_ratio", 0) < cfg.min_coverage_ratio:
        status = "COBERTURA_BAJA"

    start_date = df_processed.index.min().strftime("%Y-%m-%d") if df_processed is not None and len(df_processed) else None
    end_date = df_processed.index.max().strftime("%Y-%m-%d") if df_processed is not None and len(df_processed) else None

    indicators_nan_pct = None
    if df_processed is not None and len(df_processed):
        indicator_cols = [
            "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B",
            "MACD_Linea", "MACD_Signal", "MACD_Histograma", "RSI",
            "Estocastico_K", "Estocastico_D", "ATR", "Volumen_Relativo", "HA_Alcista",
        ]
        existing_cols = [c for c in indicator_cols if c in df_processed.columns]
        indicators_nan_pct = float(df_processed[existing_cols].isna().mean().mean() * 100) if existing_cols else None

    return {
        "Ticker": ticker,
        "Status": status,
        "Rows_Raw": int(len(df_raw)) if df_raw is not None else 0,
        "Rows_Processed": int(len(df_processed)) if df_processed is not None else 0,
        "Start_Date": start_date,
        "End_Date": end_date,
        "Duplicate_Dates_Removed": clean_report.get("duplicate_dates_removed", 0),
        "Rows_Dropped_Core_Missing": clean_report.get("rows_dropped_core_ohlc_missing", 0),
        "Coverage_Ratio": clean_report.get("coverage_ratio", np.nan),
        "Missing_Pct_After_Clean": clean_report.get("missing_pct_after_clean", np.nan),
        "Indicators_NaN_Pct": indicators_nan_pct,
        "Error_Message": error_message,
    }


# %%
# =============================================================================
# EXPORTACIÓN Y VISUALIZACIÓN
# =============================================================================

def save_ticker_outputs(
    ticker: str,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    paths: Dict[str, Path],
    cfg: PipelineConfig,
) -> None:
    fname = safe_ticker_name(ticker)
    if cfg.save_individual_csvs:
        raw_df.to_csv(paths["raw"] / f"{fname}_ohlcv_raw.csv", index=True)
        processed_df.to_csv(paths["processed"] / f"{fname}_ohlcv_indicadores.csv", index=True)


def save_master_outputs(
    master_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    paths: Dict[str, Path],
    cfg: PipelineConfig,
) -> None:
    if cfg.save_master_parquet:
        master_df.to_parquet(paths["processed"] / "market_ohlcv_indicators_master.parquet", index=True)
    if cfg.save_master_csv:
        master_df.to_csv(paths["processed"] / "market_ohlcv_indicators_master.csv", index=True)
    if cfg.save_quality_report:
        quality_df.to_csv(paths["reports"] / "quality_manifest_nb01.csv", index=False)

        summary = pd.DataFrame([
            {
                "processed_tickers": int((quality_df["Status"] == "OK").sum()),
                "failed_tickers": int((quality_df["Status"] == "ERROR").sum()),
                "insufficient_or_low_coverage": int((quality_df["Status"] != "OK").sum() - (quality_df["Status"] == "ERROR").sum()),
                "rows_master": int(len(master_df)),
                "min_processed_required": cfg.minimum_processed_tickers_done,
                "criterion_done": bool((quality_df["Status"] == "OK").sum() >= cfg.minimum_processed_tickers_done),
            }
        ])
        summary.to_csv(paths["reports"] / "summary_nb01.csv", index=False)


def create_validation_plot(master_df: pd.DataFrame, ticker: str, figure_path: Path) -> None:
    plot_df = master_df.loc[master_df["Ticker"] == ticker].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_index().tail(126)
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"NB-01 v2 - Verificación de indicadores ({ticker}, últimos 6 meses)", fontsize=14, fontweight="bold")

    # Precio + Ichimoku
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df["Close_Adj"], label="Close_Adj", linewidth=1.4)
    ax1.plot(plot_df.index, plot_df["Ichimoku_Tenkan"], label="Tenkan", linewidth=0.9)
    ax1.plot(plot_df.index, plot_df["Ichimoku_Kijun"], label="Kijun", linewidth=0.9)
    ax1.fill_between(plot_df.index, plot_df["Ichimoku_Senkou_A"], plot_df["Ichimoku_Senkou_B"], alpha=0.15)
    ax1.set_title("Precio ajustado + Ichimoku")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)

    # MACD
    ax2 = axes[1]
    ax2.plot(plot_df.index, plot_df["MACD_Linea"], label="MACD", linewidth=1.0)
    ax2.plot(plot_df.index, plot_df["MACD_Signal"], label="Signal", linewidth=1.0)
    colors = ["green" if x >= 0 else "red" for x in plot_df["MACD_Histograma"].fillna(0)]
    ax2.bar(plot_df.index, plot_df["MACD_Histograma"].fillna(0), color=colors, alpha=0.45)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_title("MACD (12,26,9)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)

    # RSI + Estocástico
    ax3 = axes[2]
    ax3.plot(plot_df.index, plot_df["RSI"], label="RSI", linewidth=1.0)
    ax3.plot(plot_df.index, plot_df["Estocastico_K"], label="Stoch %K", linewidth=1.0)
    ax3.plot(plot_df.index, plot_df["Estocastico_D"], label="Stoch %D", linewidth=0.8, linestyle="--")
    ax3.axhline(70, linestyle="--", linewidth=0.8)
    ax3.axhline(50, linestyle=":", linewidth=0.8)
    ax3.axhline(80, linestyle=":", linewidth=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI + Estocástico")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.25)

    # ATR
    ax4 = axes[3]
    ax4.plot(plot_df.index, plot_df["ATR"], label="ATR", linewidth=1.0)
    ax4.set_title("ATR (14)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(alpha=0.25)

    # Volumen relativo + HA
    ax5 = axes[4]
    colors_ha = ["green" if int(v) == 1 else "red" for v in plot_df["HA_Alcista"].fillna(0)]
    ax5.bar(plot_df.index, plot_df["Volumen_Relativo"], color=colors_ha, alpha=0.6)
    ax5.axhline(1.0, linestyle="--", linewidth=0.8, label="Media = 1.0")
    ax5.axhline(1.2, linestyle=":", linewidth=0.8, label="Umbral diseño = 1.2")
    ax5.set_title("Volumen relativo + Heiken Ashi")
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# %%
# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def process_ticker(ticker: str, cfg: PipelineConfig, paths: Dict[str, Path]) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    try:
        raw_df = download_single_ticker(ticker, cfg)
        raw_df = standardize_ohlcv_columns(raw_df)
        raw_df["Ticker"] = ticker

        clean_df, clean_report = clean_ohlcv(explicit_adjusted_ohlc(raw_df), cfg)
        clean_df["Ticker"] = ticker

        processed_df = add_technical_indicators(clean_df)
        processed_df["Ticker"] = ticker

        save_ticker_outputs(ticker, raw_df, processed_df, paths, cfg)

        manifest_row = build_quality_manifest_row(
            ticker=ticker,
            df_raw=raw_df,
            df_processed=processed_df,
            clean_report=clean_report,
            cfg=cfg,
            error_message="",
        )
        return processed_df, manifest_row

    except Exception as exc:
        empty_clean_report = {
            "duplicate_dates_removed": 0,
            "rows_dropped_core_ohlc_missing": 0,
            "coverage_ratio": 0.0,
            "missing_pct_after_clean": np.nan,
        }
        manifest_row = build_quality_manifest_row(
            ticker=ticker,
            df_raw=pd.DataFrame(),
            df_processed=None,
            clean_report=empty_clean_report,
            cfg=cfg,
            error_message=str(exc),
        )
        return None, manifest_row


def run_pipeline(cfg: Optional[PipelineConfig] = None) -> Dict[str, object]:
    if cfg is None:
        cfg = PipelineConfig()

    paths = build_paths(cfg)

    print("=" * 80)
    print("NB-01 v2 | Adquisición y enriquecimiento técnico de datos")
    print("=" * 80)
    print(f"Root del proyecto : {paths['root']}")
    print(f"Período          : {cfg.start_date} -> {cfg.end_date}")
    print(f"Tickers          : {len(cfg.tickers)}")
    print(f"Benchmark        : {cfg.benchmark_ticker}")

    processed_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []

    for i, ticker in enumerate(cfg.tickers, start=1):
        print(f"[{i:02d}/{len(cfg.tickers):02d}] Procesando {ticker}...", end=" ")
        processed_df, manifest_row = process_ticker(ticker, cfg, paths)
        manifest_rows.append(manifest_row)

        if processed_df is not None and manifest_row["Status"] == "OK":
            processed_frames.append(processed_df)
            print(f"OK | filas={len(processed_df)} | cobertura={manifest_row['Coverage_Ratio']:.3f}")
        else:
            print(f"{manifest_row['Status']} | {manifest_row['Error_Message']}")

    quality_df = pd.DataFrame(manifest_rows).sort_values(["Status", "Ticker"]).reset_index(drop=True)

    if processed_frames:
        master_df = pd.concat(processed_frames, axis=0).sort_index()
    else:
        master_df = pd.DataFrame()

    save_master_outputs(master_df, quality_df, paths, cfg)

    if cfg.make_validation_plot and not master_df.empty:
        create_validation_plot(
            master_df=master_df,
            ticker=cfg.example_plot_ticker,
            figure_path=paths["figures"] / f"verificacion_indicadores_{safe_ticker_name(cfg.example_plot_ticker)}.png",
        )

    processed_ok = int((quality_df["Status"] == "OK").sum()) if not quality_df.empty else 0
    criterion_done = processed_ok >= cfg.minimum_processed_tickers_done

    print("\n" + "=" * 80)
    print("RESUMEN NB-01 v2")
    print("=" * 80)
    print(f"Tickers OK              : {processed_ok}/{len(cfg.tickers)}")
    print(f"Tickers con error       : {int((quality_df['Status'] == 'ERROR').sum()) if not quality_df.empty else 0}")
    print(f"Rows master             : {len(master_df)}")
    print(f"Criterio 'hecho' >= {cfg.minimum_processed_tickers_done} tickers OK: {'CUMPLIDO' if criterion_done else 'NO CUMPLIDO'}")
    print(f"Processed master        : {paths['processed']}")
    print(f"Reports                 : {paths['reports']}")
    print(f"Figures                 : {paths['figures']}")

    return {
        "config": asdict(cfg),
        "paths": {k: str(v) for k, v in paths.items()},
        "master_df": master_df,
        "quality_df": quality_df,
        "criterion_done": criterion_done,
    }


# %%
# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    cfg = PipelineConfig()
    _ = run_pipeline(cfg)
