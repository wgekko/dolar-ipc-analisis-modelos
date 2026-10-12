# Suite de Analítica Cuantitativa Macro macroeconómica y Mercado de Cambios (FX)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Statsmodels](https://img.shields.io/badge/Econometrics-Statsmodels-green.svg)](https://www.statsmodels.org/)

Una plataforma interactiva de finanzas cuantitativas, econometría aplicada y aprendizaje profundo (*Deep Learning*) desarrollada con **Python** y **Streamlit**. La suite está diseñada para el análisis de series temporales, evaluación de riesgo de mercado, pruebas de causalidad estructural, simulaciones estocásticas y modelado predictivo aplicado a la interacción entre el tipo de cambio ($USD$) y la tasa de inflación ($IPC$).

---

## 🏗️ Arquitectura del Sistema y Módulos

La solución se compone de tres paneles interactivos de alta performance:

### 1. `main.py` — Motor Macro Bivariado & Aprendizaje Automático
* **Espacio de Fases & Régimen ARAD:** Mapeo topológico bivariado ($USD$ vs. $IPC$) categorizado en cuadrantes teóricos de comportamiento económico (Guerrero, Intelectual, Logrero, Menesteroso).
* **Econometría Avanzada:** 
  * Test de Causalidad de Granger y Función de Correlación Cruzada (CCF) para determinación de rezagos dinámicos.
  * Prueba de Cointegración de Engle-Granger para evaluar relaciones de equilibrio de largo plazo.
* **Modelado Multivariado:** Vector Autorregresivo (VAR), Funciones Impulso-Respuesta (IRF) y Red Neuronal Recurrente Bivariada LSTM implementeda en PyTorch.
* **Suite de Nowcasting & Backtesting:** Agregación de frecuencias (diaria a mensual), modelos VARX/ARIMAX y *backtesting out-of-sample* en periodo de entrenamiento (*holdout*) comparando ARIMA, VAR y Random Forest mediante métricas de error ($RMSE$, $MAE$, $MAPE$).

### 2. `1-dolar-ipc.py` — Análisis Topológico Individual & Riesgo de Cola
* **Descomposición Estacional:** Aislamiento aditivo de tendencia, estacionalidad y residuos/componente aleatorio.
* **Métricas de Riesgo de Cola:** Valor en Riesgo Histórico ($VaR_{95\%}$, $VaR_{99\%}$) y Valor en Riesgo Condicional / *Expected Shortfall* ($CVaR$).
* **Volatilidad Dinámica:** Estimación móvil de desviación estándar y trazado de bandas de dispersión.
* **Aprendizaje Supervisado:** Regresores Random Forest y Gradient Boosting basados en rezagos (*lags*) y aceleración (*momentum*).
* **Monte Carlo Fractal:** Simulación estocástica multiescenario aplicando un factor de escala fractal ($p = 0.6366$).

### 3. `2-dolar-diario.py` — Dashboard de Alta Frecuencia & Gestión de Riesgo FX
* **Análisis Técnico Algorítmico:** Medias Móviles Exponenciales (EMA 20, EMA 50) y Bandas de Bollinger.
* **Gestión Cuantitativa de Riesgo:** Volatilidad anualizada (base 252 ruedas), estimación de *Maximum Drawdown* (MDD) y retornos logarítmicos.
* **Proyección Estocástica:** Movimiento Browniano Geométrico (GBM) para generación de trayectorias futuras.
* **Deep Learning Temporal:** Arquitectura de red LSTM univariada optimizada con PyTorch para predicción de precios promedio a corto plazo.

---

## 📊 Metodología Econométrica y Modelado Matemático

### Cointegración de Engle-Granger
Evalúa si dos series no estacionarias $I(1)$ comparten una combinación lineal estacionaria $I(0)$, indicando una relación estructural de largo plazo:
$$y_t - \beta x_t = e_t \sim I(0)$$

### Vector Autorregresivo (VAR)
Modela el sistema dinámico endógeno considerando rezagos cruzados:
$$Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + \dots + A_p Y_{t-p} + \epsilon_t$$

### Red Neuronal Recurrente LSTM Bivariada
Aprovecha las puertas de memoria (*gates*) para capturar dependencias temporales no lineales de largo alcance en las series de tiempo:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

---

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/macro-fx-analytics-suite.git
   cd macro-fx-analytics-suite
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar los Dashboards:**
   * **Dashboard Principal Bivariado:**
     ```bash
     streamlit run main.py
     ```
   * **Dashboard Individual USD/IPC:**
     ```bash
     streamlit run 1-dolar-ipc.py
     ```
   * **Dashboard Diario USD:**
     ```bash
     streamlit run 2-dolar-diario.py
     ```

---

## 📂 Estructura del Proyecto

```text
├── main.py              # Engine Bivariado Macro, Econometría & ML
├── 1-dolar-ipc.py       # Análisis Topológico Individual & Riesgo de Cola
├── 2-dolar-diario.py    # Dashboard FX Diario, Movimiento Browniano & PyTorch LSTM
├── requirements.txt     # Librerías y dependencias del sistema
└── data/                # Archivos de datos históricos (.xlsx)
```


🎨 Configuración Visual (Opcional)

Para mantener una estética limpia tipo terminal financiera en Streamlit, crea una carpeta .streamlit en la raíz del proyecto y un archivo config.toml:

[theme]
primaryColor = "#00FF7F"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#e6edf3"
font = "sans serif"

## Licencia
Este proyecto está bajo la Licencia MIT.

