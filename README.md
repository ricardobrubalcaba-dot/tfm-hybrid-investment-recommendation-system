# Diseño y Validación de un Sistema Híbrido de Recomendación de Inversión

Repositorio documental y experimental asociado al Trabajo Fin de Máster:

**Diseño y Validación de un Sistema Híbrido de Recomendación de Inversión**  
Máster Universitario en Inteligencia Artificial  
Universidad Internacional de La Rioja (UNIR)

Autor: Ricardo Benítez Rubalcaba  
Director: Miguel Ángel Navarro Arenas  
Fecha: 2026

---

## Descripción general

Este repositorio contiene los notebooks, scripts auxiliares y salidas experimentales seleccionadas utilizadas para documentar el pipeline metodológico del Trabajo Fin de Máster.

El proyecto propone un sistema híbrido de recomendación de inversión para renta variable estadounidense, combinando análisis técnico determinista, generación de señales candidatas, construcción de un dataset supervisado mediante meta-labeling, validación temporal walk-forward, comparación de modelos supervisados, priorización de señales mediante ranking por score, backtest de cartera con costes y slippage, y overlay contextual/noticioso mediante GDELT y FinBERT.

La lógica central del trabajo es que la inteligencia artificial no predice de forma autónoma el mercado, sino que actúa como una capa de priorización y meta-validación sobre señales técnicas previamente definidas.

---

## Estructura del repositorio

```text
notebooks/
├── 01_data_validation/
├── 02_technical_indicators_and_signals/
├── 03_technical_portfolio_baseline/
├── 04_supervised_dataset/
├── 05_ml_models_and_ranking/
├── 06_ablation_and_portfolio_backtest/
└── 07_context_and_news_overlay/

results/
├── 04_supervised_dataset_outputs/
├── 05_ml_outputs_nb06/
├── 05_ml_outputs_nb06b/
├── 06_ablation_outputs_nb07/
├── 06_portfolio_backtest_outputs_nb07b_v2/
├── 07_context_overlay_outputs_nb08a/
└── 08_news_overlay_outputs_nb08b_v4/

data/
├── README.md
├── raw/
├── external/
└── derived/

docs/
├── reproducibility.md
└── notebook_traceability.md
```

---

## Correspondencia con los anexos del TFM

| Anexo del TFM | Contenido metodológico | Carpeta principal |
|---|---|---|
| Anexo A | Preparación de datos y universo de análisis | `notebooks/01_data_validation/` |
| Anexo B | Generación de indicadores técnicos y señales candidatas | `notebooks/02_technical_indicators_and_signals/` |
| Anexo C | Calibración del screener técnico y reglas de entrada/salida | `notebooks/02_technical_indicators_and_signals/` |
| Anexo D | Consolidación del baseline financiero | `notebooks/03_technical_portfolio_baseline/` |
| Anexo E | Dataset supervisado y triple barrera | `notebooks/04_supervised_dataset/` |
| Anexo F | Modelos supervisados y ranking por score | `notebooks/05_ml_models_and_ranking/` |
| Anexo G | Estudio de ablación económica | `notebooks/06_ablation_and_portfolio_backtest/` |
| Anexo H | Backtest de cartera con scores de machine learning | `notebooks/06_ablation_and_portfolio_backtest/` |
| Anexo I | Overlay contextual de riesgo | `notebooks/07_context_and_news_overlay/` |
| Anexo J | Integración GDELT + FinBERT | `notebooks/07_context_and_news_overlay/` |

---

## Notebooks principales

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

### Dataset supervisado y meta-labeling

- `NB_05_dataset_supervisado_meta_labeling_PARA_EJECUTAR.ipynb`

### Modelos supervisados y ranking

- `NB_06_modelos_supervisados_validacion_temporal_PARA_EJECUTAR_v5_matrices_visibles.ipynb`
- `NB_06B_modelos_supervisados_thresholds_ranking_PARA_EJECUTAR_v2_fix_indent.ipynb`

### Ablación y backtest de cartera

- `NB_07_ablation_backtest_ml_scores_PARA_EJECUTAR.ipynb`
- `NB_07B_backtest_cartera_con_scores_ml_PARA_EJECUTAR_v2_fix_drawdown.ipynb`

La versión `v2_fix_drawdown` corresponde a la versión corregida del cálculo de drawdown event-driven utilizada en la memoria final.

### Overlay contextual y capa noticiosa

- `NB_08A_context_risk_overlay_exploratorio_PARA_EJECUTAR.ipynb`
- `NB_08B_fundamental_news_sentiment_overlay_PARA_EJECUTAR_v4_recalculo_gdelt.ipynb`

---

## Resultados principales documentados

Los resultados consolidados en la memoria final incluyen:

- Dataset supervisado: 8.155 señales candidatas sobre 503 tickers.
- Período de análisis: 2018-2025.
- Validación out-of-sample: 2023-2025.
- Modelo seleccionado: Random Forest.
- Criterio final: Top 30% de señales por score dentro de cada fold.
- Capital inicial del backtest: 200.000 USD.
- Capital final Random Forest Top-30%: 395.127 USD.
- Capital final baseline técnico: 255.654 USD.
- Drawdown event-driven Random Forest Top-30%: -3,18%.
- Win rate Random Forest Top-30%: 61,93%.
- Profit factor Random Forest Top-30%: 3,74.

Estos resultados deben interpretarse dentro de los límites de una simulación histórica y no constituyen una recomendación de inversión ni una promesa de rentabilidad futura.

---

## Política de datos

Los datos brutos de mercado no se redistribuyen en este repositorio. El proyecto utiliza series históricas OHLCV obtenidas mediante `yfinance` / Yahoo Finance y noticias recuperadas mediante GDELT.

Cuando corresponde, el repositorio incluye salidas derivadas, tablas, figuras y outputs demostrativos utilizados para documentar la trazabilidad experimental del TFM. Para una reproducción completa, deben reconstruirse los datos de entrada conforme a las fuentes y procedimientos documentados en los notebooks.

---

## Reproducibilidad

El flujo lógico de ejecución es:

```text
Datos y universo
→ indicadores técnicos
→ screener V4
→ auditoría de señales
→ baseline financiero
→ dataset supervisado
→ modelos supervisados
→ ranking por score
→ ablación económica
→ backtest de cartera
→ overlay contextual y noticioso
```

Por restricciones de tamaño y condiciones de redistribución de algunas fuentes, no todos los datos brutos se incluyen en el repositorio.

---

## Limitaciones

Este repositorio documenta un piloto experimental académico. Sus resultados están sujetos a las limitaciones descritas en la memoria: posible sesgo de supervivencia, dependencia de datos históricos, cobertura parcial de noticias, ausencia de ejecución real en mercado, backtest event-driven y resultados no extrapolables automáticamente a otros regímenes de mercado.

---

## Autoría

Este repositorio ha sido preparado exclusivamente por:

**Ricardo Benítez Rubalcaba**

No contiene contribuciones de terceros. Las herramientas de inteligencia artificial generativa se utilizaron únicamente como apoyo auxiliar en tareas de revisión, organización, redacción, contraste técnico y depuración, conforme a la declaración incluida en la memoria del TFM.

---

## Aviso

Este trabajo tiene fines exclusivamente académicos y experimentales. No constituye asesoramiento financiero, recomendación de inversión ni sistema operativo de trading.
