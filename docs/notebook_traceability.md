# Trazabilidad de notebooks

Este documento resume la correspondencia entre los anexos de la memoria del TFM, los notebooks seleccionados y las carpetas de resultados incluidas en el repositorio.

## Correspondencia general

| Anexo | Bloque metodológico | Carpeta de notebooks | Carpeta de resultados |
|---|---|---|---|
| Anexo A | Preparación de datos y universo de análisis | `notebooks/01_data_validation/` | No aplica directamente |
| Anexo B | Generación de indicadores técnicos y señales candidatas | `notebooks/02_technical_indicators_and_signals/` | No aplica directamente |
| Anexo C | Calibración del screener técnico y reglas de entrada/salida | `notebooks/02_technical_indicators_and_signals/` | No aplica directamente |
| Anexo D | Consolidación del baseline financiero | `notebooks/03_technical_portfolio_baseline/` | Outputs técnicos previos |
| Anexo E | Construcción del dataset supervisado y triple barrera | `notebooks/04_supervised_dataset/` | `results/04_supervised_dataset_outputs/` |
| Anexo F | Modelos supervisados y ranking por score | `notebooks/05_ml_models_and_ranking/` | `results/05_ml_outputs_nb06/` y `results/05_ml_outputs_nb06b/` |
| Anexo G | Estudio de ablación económica | `notebooks/06_ablation_and_portfolio_backtest/` | `results/06_ablation_outputs_nb07/` |
| Anexo H | Backtest de cartera con scores de machine learning | `notebooks/06_ablation_and_portfolio_backtest/` | `results/06_portfolio_backtest_outputs_nb07b_v2/` |
| Anexo I | Overlay contextual de riesgo | `notebooks/07_context_and_news_overlay/` | `results/07_context_overlay_outputs_nb08a/` |
| Anexo J | Integración GDELT + FinBERT | `notebooks/07_context_and_news_overlay/` | `results/08_news_overlay_outputs_nb08b_v4/` |

## Notebooks seleccionados

### Preparación y validación de datos

- `NB_01_ejecucion_y_validacion_datos.ipynb`
- `NB_01_data_pipeline_v2.py`

### Indicadores técnicos, universo y screener

- `NB_02B_ichimoku_sensibilidad_7_22_44_REVISADO_CON_GRAFICOS.ipynb`
- `NB_04A_universo_sp500_indicadores_ichimoku_7_22_44_ANALIZADO_V2.ipynb`
- `NB_04B_screener_sp500_CONTRATO_REVISADO_HA2_STOCH_KUMO_ACTUAL_V4.ipynb`
- `NB_04B2_auditoria_exhaustiva_senales_V4_ANALIZADO_PRE_04C.ipynb`

### Baseline financiero

- `NB_04C_backtest_cartera_tecnica_V4_pre_FinBERT_ANALIZADO.ipynb`

### Dataset supervisado

- `NB_05_dataset_supervisado_meta_labeling_PARA_EJECUTAR.ipynb`

### Modelos supervisados y ranking

- `NB_06_modelos_supervisados_validacion_temporal_PARA_EJECUTAR_v5_matrices_visibles.ipynb`
- `NB_06B_modelos_supervisados_thresholds_ranking_PARA_EJECUTAR_v2_fix_indent.ipynb`

### Ablación y backtest de cartera

- `NB_07_ablation_backtest_ml_scores_PARA_EJECUTAR.ipynb`
- `NB_07B_backtest_cartera_con_scores_ml_PARA_EJECUTAR_v2_fix_drawdown.ipynb`

### Overlay contextual y noticioso

- `NB_08A_context_risk_overlay_exploratorio_PARA_EJECUTAR.ipynb`
- `NB_08B_fundamental_news_sentiment_overlay_PARA_EJECUTAR_v4_recalculo_gdelt.ipynb`

## Versiones excluidas

No se incluyen versiones históricas, exploratorias o superadas para evitar contradicciones metodológicas.

La versión utilizada para el backtest final es:

`NB_07B_backtest_cartera_con_scores_ml_PARA_EJECUTAR_v2_fix_drawdown.ipynb`

La versión utilizada para la capa fundamental/noticiosa final es:

`NB_08B_fundamental_news_sentiment_overlay_PARA_EJECUTAR_v4_recalculo_gdelt.ipynb`
