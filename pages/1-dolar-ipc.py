import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# IMPORTACIONES Y CONFIGURACIÓN
# ==========================================

# Importación opcional y segura de pmdarima.
# Si falla la importación, se utilizará statsmodels.
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    pm = None
    HAS_PMDARIMA = False

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Análisis Individual: Dólar e IPC + ML",
    page_icon=":material/analytics:",
    layout="wide"
)


# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================

@st.cache_data
def load_and_prep_dolar(filename="data/dolar.xlsx"):
    path = filename if os.path.exists(filename) else "dolar.xlsx"

    try:
        xls = pd.ExcelFile(path)

        sheet = (
            "data"
            if "data" in xls.sheet_names
            else xls.sheet_names[0]
        )

        df = pd.read_excel(path, sheet_name=sheet)

        date_cols = [
            c for c in df.columns
            if "fecha" in str(c).lower()
        ]

        date_col = date_cols[0] if date_cols else df.columns[0]

        df["fecha"] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        if "Unnamed: 6" in df.columns:
            df["Variacion_Porcentual"] = pd.to_numeric(
                df["Unnamed: 6"],
                errors="coerce"
            )
        else:
            val_cols = [
                c for c in df.columns
                if any(
                    k in str(c).lower()
                    for k in [
                        "venta",
                        "variacion",
                        "var",
                        "compra"
                    ]
                )
            ]

            val_col = (
                val_cols[0]
                if val_cols
                else df.columns[1]
            )

            vals = pd.to_numeric(
                df[val_col],
                errors="coerce"
            )

            if vals.max() > 100:
                df["Variacion_Porcentual"] = (
                    vals.pct_change() * 100
                )
            else:
                df["Variacion_Porcentual"] = vals

        df["Variacion_Porcentual"] = pd.to_numeric(
            df["Variacion_Porcentual"],
            errors="coerce"
        )

        df = df.replace([np.inf, -np.inf], np.nan)

        return (
            df.dropna(
                subset=["fecha", "Variacion_Porcentual"]
            )
            .sort_values("fecha")
            .reset_index(drop=True)
        )

    except Exception as e:
        st.error(f"Error cargando {path}: {e}")
        return None


@st.cache_data
def load_and_prep_ipc(filename="data/ipc.xlsx"):
    path = filename if os.path.exists(filename) else "ipc.xlsx"

    try:
        xls = pd.ExcelFile(path)
        sheet_name = xls.sheet_names[0]

        df_raw = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None
        )

        nivel_gen_idx = None
        dates_idx = None

        for idx, row in df_raw.iterrows():
            val = (
                str(row.iloc[0])
                .strip()
                .lower()
                .replace(" ", "")
            )

            if "nivelgeneral" in val:
                nivel_gen_idx = idx
            elif val in ["totalnacional", "totalnacional-"]:
                dates_idx = idx

        if (
            nivel_gen_idx is not None
            and dates_idx is not None
        ):
            raw_dates = df_raw.iloc[dates_idx].values[1:]
            raw_values = df_raw.iloc[nivel_gen_idx].values[1:]

            df = pd.DataFrame({
                "fecha": pd.to_datetime(
                    raw_dates,
                    errors="coerce"
                ),
                "Variacion_Porcentual": pd.to_numeric(
                    raw_values,
                    errors="coerce"
                )
            })

            df = df.replace([np.inf, -np.inf], np.nan)

            return (
                df.dropna(
                    subset=["fecha", "Variacion_Porcentual"]
                )
                .sort_values("fecha")
                .reset_index(drop=True)
            )

        df = pd.read_excel(
            path,
            sheet_name=sheet_name
        )

        date_cols = [
            col for col in df.columns
            if any(
                k in str(col).lower()
                for k in [
                    "fecha",
                    "periodo",
                    "date",
                    "mes"
                ]
            )
        ]

        val_cols = [
            col for col in df.columns
            if any(
                k in str(col).lower()
                for k in [
                    "general",
                    "ipc",
                    "var",
                    "valor",
                    "nivel"
                ]
            )
        ]

        if date_cols and val_cols:
            df["fecha"] = pd.to_datetime(
                df[date_cols[0]],
                errors="coerce"
            )

            df["Variacion_Porcentual"] = pd.to_numeric(
                df[val_cols[-1]],
                errors="coerce"
            )

            df = df.replace([np.inf, -np.inf], np.nan)

            return (
                df.dropna(
                    subset=["fecha", "Variacion_Porcentual"]
                )
                .sort_values("fecha")
                .reset_index(drop=True)
            )

        return None

    except Exception as e:
        st.error(f"Error cargando {path}: {e}")
        return None


# ==========================================
# 2. CÁLCULO DEL ESPACIO DE FASES
# ==========================================

def calcular_espacio_fases(df):
    df = df.copy()

    df["Momentum_Aceleracion"] = (
        df["Variacion_Porcentual"]
        .diff()
        .fillna(0)
    )

    mean_var = df["Variacion_Porcentual"].mean()

    cond_G = (
        (df["Variacion_Porcentual"] > mean_var)
        & (df["Momentum_Aceleracion"] > 0)
    )

    cond_I = (
        (df["Variacion_Porcentual"] <= mean_var)
        & (df["Momentum_Aceleracion"] > 0)
    )

    cond_M = (
        (df["Variacion_Porcentual"] <= mean_var)
        & (df["Momentum_Aceleracion"] <= 0)
    )

    cond_L = (
        (df["Variacion_Porcentual"] > mean_var)
        & (df["Momentum_Aceleracion"] <= 0)
    )

    df["Estado_ARAD"] = np.select(
        [cond_G, cond_I, cond_M, cond_L],
        [
            "Guerrero (Expansión)",
            "Intelectual (Recuperación)",
            "Menesteroso (Contracción)",
            "Logrero (Agotamiento)"
        ],
        default="Desconocido"
    )

    return df, mean_var


# ==========================================
# 3. PRONÓSTICO ARIMA ROBUSTO
# ==========================================

def pronosticar_arima(serie, meses):
    """
    Intenta entrenar pmdarima.
    Si falla, utiliza statsmodels ARIMA(1,1,1).

    Devuelve:
        predicciones, límite inferior, límite superior,
        nombre del modelo utilizado.
    """

    serie = pd.Series(
        pd.to_numeric(serie, errors="coerce")
    )

    serie = serie.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(serie) < 8:
        raise ValueError(
            "Se necesitan al menos 8 observaciones válidas "
            "para intentar entrenar el modelo ARIMA."
        )

    # Intento 1: pmdarima / Auto-ARIMA.
    if HAS_PMDARIMA:
        try:
            model_auto = pm.auto_arima(
                serie,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False
            )

            predicciones, intervalos = model_auto.predict(
                n_periods=meses,
                return_conf_int=True,
                alpha=0.05
            )

            predicciones = np.asarray(
                predicciones,
                dtype=float
            ).reshape(-1)

            intervalos = np.asarray(
                intervalos,
                dtype=float
            )

            inferior = intervalos[:, 0]
            superior = intervalos[:, 1]

            if (
                len(predicciones) == meses
                and len(inferior) == meses
                and len(superior) == meses
                and np.isfinite(predicciones).all()
            ):
                return (
                    predicciones,
                    inferior,
                    superior,
                    f"Auto-ARIMA {model_auto.order}"
                )

        except Exception as e:
            # No interrumpimos la aplicación:
            # continuamos con statsmodels.
            st.warning(
                "Auto-ARIMA no pudo ejecutarse. "
                "Se utilizará statsmodels ARIMA(1,1,1). "
                f"Detalle: {e}"
            )

    # Intento 2: modelo alternativo con statsmodels.
    model_arima = ARIMA(
        serie,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit_arima = model_arima.fit()

    forecast_res = fit_arima.get_forecast(
        steps=meses
    )

    predicciones = np.asarray(
        forecast_res.predicted_mean,
        dtype=float
    ).reshape(-1)

    intervalos = np.asarray(
        forecast_res.conf_int(alpha=0.05),
        dtype=float
    )

    inferior = intervalos[:, 0]
    superior = intervalos[:, 1]

    if (
        len(predicciones) != meses
        or not np.isfinite(predicciones).all()
    ):
        raise ValueError(
            "El modelo ARIMA generó predicciones inválidas."
        )

    return (
        predicciones,
        inferior,
        superior,
        "ARIMA(1,1,1) - statsmodels"
    )


# ==========================================
# 4. MOTOR DE RENDERIZADO
# ==========================================

def render_analysis(df, asset_name):

    st.subheader(
        f":material/problem: Análisis Individual Topológico, Riesgo & ML: {asset_name}"
    )
    st.info("se trabaja variables individuales de dolar o ipc (inflación)- período mensual")

    # Copiar y limpiar antes de calcular las fases.
    df = df.copy()

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="coerce"
    )

    df["Variacion_Porcentual"] = pd.to_numeric(
        df["Variacion_Porcentual"],
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = (
        df.dropna(
            subset=["fecha", "Variacion_Porcentual"]
        )
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    if df.empty:
        st.error(
            f"No hay datos válidos para analizar {asset_name}."
        )
        return

    # Calcular fases después de limpiar los datos.
    df, mean_var = calcular_espacio_fases(df)

    last_val = float(
        df["Variacion_Porcentual"].iloc[-1]
    )

    std_val = float(
        df["Variacion_Porcentual"].std()
    ) if len(df) > 1 else 0.0

    # Definir todas las pestañas antes de utilizarlas.
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        ":material/list_alt: Fases y ARAD",
        ":material/list_alt: Descomposición Estacional",
        ":material/list_alt: Riesgo de Cola (VaR / CVaR)",
        ":material/list_alt: Volatilidad Dinámica",
        ":material/list_alt: ESENARIOS vs. ARIMA",
        ":material/list_alt: Random Forest & Boosting",
        ":material/list_alt: Monte Carlo Fractal"
    ])

# ==========================================
    # TAB 1: FASES Y ARAD INDIVIDUAL
    # ==========================================

    with tab1:
        st.subheader(
            f"Diagrama de Fases y Cuadrantes ARAD: {asset_name}"
        )

        # Copia de los datos del activo seleccionado.
        df_phase = df.copy()

        # Recalcular las fases exclusivamente para este activo.
        df_phase, mean_var = calcular_espacio_fases(df_phase)

        if df_phase.empty:
            st.warning(
                "No hay datos válidos para construir el diagrama."
            )
        else:
            ultimo_dato = df_phase.iloc[-1]

            nombre_variable = (
                "Variación del dólar"
                if asset_name.lower() == "dólar"
                else "Variación del IPC"
            )

            # Gráfico de fases.
            fig_phase = px.scatter(
                df_phase,
                x="Variacion_Porcentual",
                y="Momentum_Aceleracion",
                color="Estado_ARAD",
                hover_data={
                    "fecha": "|%Y-%m-%d",
                    "Variacion_Porcentual": ":.2f",
                    "Momentum_Aceleracion": ":.2f"
                },
                labels={
                    "Variacion_Porcentual": f"{nombre_variable} (%)",
                    "Momentum_Aceleracion": "Aceleración",
                    "Estado_ARAD": "Estado ARAD"
                },
                title=f"Espacio de Fases Individual: {asset_name}",
                template="plotly_white"
            )

            # Trayectoria histórica.
            fig_phase.add_trace(
                go.Scatter(
                    x=df_phase["Variacion_Porcentual"],
                    y=df_phase["Momentum_Aceleracion"],
                    mode="lines",
                    line=dict(
                        color="rgba(120,120,120,0.25)",
                        width=1
                    ),
                    name="Trayectoria histórica",
                    hoverinfo="skip"
                )
            )

            # Resaltar la última observación.
            fig_phase.add_trace(
                go.Scatter(
                    x=[ultimo_dato["Variacion_Porcentual"]],
                    y=[ultimo_dato["Momentum_Aceleracion"]],
                    mode="markers+text",
                    marker=dict(
                        color="gold",
                        size=17,
                        symbol="star",
                        line=dict(color="black", width=1)
                    ),
                    text=["Último dato"],
                    textposition="top center",
                    name="Posición actual"
                )
            )

            # Media histórica y aceleración cero.
            fig_phase.add_vline(
                x=mean_var,
                line_dash="dash",
                line_color="gray",
                annotation_text="Media histórica"
            )

            fig_phase.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Aceleración cero"
            )

            fig_phase.update_layout(
                xaxis_title=f"{nombre_variable} (%)",
                yaxis_title="Aceleración",
                hovermode="closest"
            )

            st.plotly_chart(
                fig_phase,
                width='stretch'
            )

            # Diagnóstico del último registro.
            fecha_ultimo = pd.to_datetime(
                ultimo_dato["fecha"]
            ).strftime("%Y-%m-%d")

            valor_actual = float(
                ultimo_dato["Variacion_Porcentual"]
            )

            aceleracion_actual = float(
                ultimo_dato["Momentum_Aceleracion"]
            )

            estado_actual = ultimo_dato["Estado_ARAD"]

            posicion_media = (
                "por encima"
                if valor_actual > mean_var
                else "por debajo o igual a"
            )

            tendencia_aceleracion = (
                "positiva"
                if aceleracion_actual > 0
                else "negativa o nula"
            )

            st.markdown(
                f"""
                    ### Diagnóstico del Estado Actual

                    - **Activo analizado:** {asset_name}
                    - **Fecha del último registro:** {fecha_ultimo}
                    - **Estado ARAD:** `{estado_actual}`
                    - **Variación actual:** `{valor_actual:.2f}%`
                    - **Media histórica:** `{mean_var:.2f}%`
                    - **Aceleración actual:** `{aceleracion_actual:+.2f}` puntos porcentuales

                    **Lectura analítica:** {asset_name} se encuentra en el estado
                    **{estado_actual}**. La variación actual está {posicion_media}
                    su media histórica y presenta una aceleración {tendencia_aceleracion}.
                    """
                )

            # Indicadores resumidos.
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Estado ARAD actual",
                estado_actual
            )

            col2.metric(
                "Variación actual",
                f"{valor_actual:.2f}%",
                delta=f"{valor_actual - mean_var:+.2f} pp vs. media"
            )

            col3.metric(
                "Aceleración",
                f"{aceleracion_actual:+.2f}"
            )
    # ==========================================
    # TAB 2: DESCOMPOSICIÓN ESTACIONAL
    # ==========================================

    with tab2:
        st.subheader(
            "Descomposición de Serie de Tiempo: "
            "Tendencia, Estacionalidad y Residuos"
        )

        try:
            df_decomp = (
                df.set_index("fecha")["Variacion_Porcentual"]
                .sort_index()
            )

            decomp = seasonal_decompose(
                df_decomp,
                model="additive",
                period=12
            )

            fig_dec = go.Figure()

            fig_dec.add_trace(
                go.Scatter(
                    x=df_decomp.index,
                    y=decomp.trend,
                    name="Tendencia Pura",
                    line=dict(color="orange", width=2)
                )
            )

            fig_dec.add_trace(
                go.Scatter(
                    x=df_decomp.index,
                    y=decomp.seasonal,
                    name="Componente Estacional",
                    line=dict(color="teal", width=1.5)
                )
            )

            fig_dec.add_trace(
                go.Scatter(
                    x=df_decomp.index,
                    y=decomp.resid,
                    name="Residuos / Shocks",
                    mode="markers",
                    marker=dict(size=4, color="gray")
                )
            )

            fig_dec.update_layout(
                title="Descomposición Aditiva (Periodo = 12 Meses)",
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_dec,
                width='stretch'
            )

        except Exception as e:
            st.warning(
                "Se requieren suficientes observaciones "
                "para la descomposición estacional. "
                f"Detalle: {e}"
            )

    # ==========================================
    # TAB 3: RIESGO DE COLA
    # ==========================================

    with tab3:
        st.subheader(
            "Análisis de Riesgo de Cola y Eventos Extremos"
        )

        col_m1, col_m2, col_m3 = st.columns(3)

        valores = df["Variacion_Porcentual"].to_numpy()

        var_95 = np.percentile(valores, 5)
        var_99 = np.percentile(valores, 1)

        cvar_95 = valores[valores <= var_95].mean()

        col_m1.metric(
            "Percentil 5% (VaR de cola)",
            f"{var_95:.2f}%"
        )

        col_m2.metric(
            "CVaR 95% (cola inferior)",
            f"{cvar_95:.2f}%"
        )

        col_m3.metric(
            "Percentil 1% (cola extrema)",
            f"{var_99:.2f}%"
        )

        fig_hist = px.histogram(
            df,
            x="Variacion_Porcentual",
            marginal="violin",
            nbins=30,
            title=(
                "Distribución de Frecuencia y "
                f"Mapeo de Colas Pesadas - {asset_name}"
            ),
            template="plotly_white",
            opacity=0.7
        )

        fig_hist.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="red",
            annotation_text="Percentil 5%"
        )

        st.plotly_chart(
            fig_hist,
            width='stretch'
        )

    # ==========================================
    # TAB 4: VOLATILIDAD DINÁMICA
    # ==========================================

    with tab4:
        st.subheader(
            "Volatilidad Móvil y Agrupamiento de Regímenes"
        )

        window = st.slider(
            "Ventana Móvil (Observaciones)",
            3,
            24,
            6,
            key=f"win_{asset_name}"
        )

        df["Rolling_Std"] = (
            df["Variacion_Porcentual"]
            .rolling(window=window)
            .std()
        )

        df["Rolling_Mean"] = (
            df["Variacion_Porcentual"]
            .rolling(window=window)
            .mean()
        )

        df["Upper_Band"] = (
            df["Rolling_Mean"] + 2 * df["Rolling_Std"]
        )

        df["Lower_Band"] = (
            df["Rolling_Mean"] - 2 * df["Rolling_Std"]
        )

        fig_vol = go.Figure()

        fig_vol.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df["Variacion_Porcentual"],
                name="Serie Original",
                line=dict(color="blue", width=1.5)
            )
        )

        fig_vol.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df["Upper_Band"],
                name="Banda Sup (2x Std)",
                line=dict(color="red", dash="dash")
            )
        )

        fig_vol.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df["Lower_Band"],
                name="Banda Inf (2x Std)",
                line=dict(color="green", dash="dash")
            )
        )

        fig_vol.update_layout(
            title=f"Bandas de Volatilidad Móvil ({window} observaciones)",
            template="plotly_white"
        )

        st.plotly_chart(
            fig_vol,
            width='stretch'
        )

    # ==========================================
    # TAB 5: ARIMA CORREGIDO
    # ==========================================

    with tab5:
        st.subheader(
            "Pronóstico Dinámico ARIMA vs. Escenarios Estáticos"
        )

        meses_arima = st.slider(
            "Meses a Pronosticar con ARIMA",
            1,
            12,
            6,
            key=f"arima_slider_{asset_name}"
        )

        try:
            serie = (
                df["Variacion_Porcentual"]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(serie) < 8:
                st.warning(
                    "No hay suficientes datos válidos para "
                    "entrenar el modelo ARIMA. "
                    "Se necesitan al menos 8 observaciones."
                )
            else:
                (
                    mean_pred,
                    lower_bound,
                    upper_bound,
                    model_name
                ) = pronosticar_arima(
                    serie,
                    meses_arima
                )

                last_date = df["fecha"].iloc[-1]

                future_dates = [
                    last_date + pd.DateOffset(months=i)
                    for i in range(1, meses_arima + 1)
                ]

                fig_arima = go.Figure()

                fig_arima.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=mean_pred,
                        mode="lines+markers",
                        name="Predicción ARIMA",
                        line=dict(color="blue", width=3)
                    )
                )

                fig_arima.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=lower_bound,
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Límite Inferior (95%)"
                    )
                )

                fig_arima.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=upper_bound,
                        mode="lines",
                        line=dict(dash="dash", color="gray"),
                        name="Límite Superior (95%)",
                        fill=None
                    )
                )

                fig_arima.update_layout(
                    title=(
                        f"Pronóstico {model_name} "
                        f"para {asset_name}"
                    ),
                    xaxis_title="Fecha",
                    yaxis_title="Variación (%)",
                    template="plotly_white",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_arima,
                    width='stretch'
                )

                st.caption(
                    f"Modelo utilizado: {model_name}. "
                    "Los intervalos representan la incertidumbre "
                    "estadística del pronóstico al 95%."
                )

                resultados = pd.DataFrame({
                    "Fecha": future_dates,
                    "Predicción": mean_pred,
                    "Límite inferior 95%": lower_bound,
                    "Límite superior 95%": upper_bound
                })

                st.dataframe(
                    resultados,
                    width='stretch',
                    hide_index=True
                )

        except Exception as e:
            st.error(
                f"Error entrenando modelo ARIMA: {e}"
            )

            st.info(
                "Verifica que la serie contenga datos numéricos "
                "válidos y que statsmodels esté correctamente instalado."
            )

    # ==========================================
    # TAB 6: MACHINE LEARNING CON REZAGOS
    # ==========================================

    with tab6:
        st.subheader(
            "Aprendizaje Supervisado con Rezagos y Momentum"
        )

        df_ml = df[["Variacion_Porcentual"]].copy()

        df_ml["Lag_1"] = (
            df_ml["Variacion_Porcentual"].shift(1)
        )

        df_ml["Lag_2"] = (
            df_ml["Variacion_Porcentual"].shift(2)
        )

        df_ml["Lag_3"] = (
            df_ml["Variacion_Porcentual"].shift(3)
        )

        df_ml["Momentum"] = (
            df_ml["Lag_1"] - df_ml["Lag_2"]
        )

        df_ml = df_ml.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(df_ml) < 5:
            st.warning(
                "No hay suficientes observaciones para entrenar "
                "los modelos de Machine Learning."
            )
        else:
            X = df_ml[
                ["Lag_1", "Lag_2", "Lag_3", "Momentum"]
            ]

            y = df_ml["Variacion_Porcentual"]

            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )

            rf_model.fit(X, y)
            gb_model.fit(X, y)

            last_features = pd.DataFrame(
                [[
                    y.iloc[-1],
                    y.iloc[-2],
                    y.iloc[-3],
                    y.iloc[-1] - y.iloc[-2]
                ]],
                columns=[
                    "Lag_1",
                    "Lag_2",
                    "Lag_3",
                    "Momentum"
                ]
            )

            pred_rf = rf_model.predict(
                last_features
            )[0]

            pred_gb = gb_model.predict(
                last_features
            )[0]

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicción Próximo Mes (Random Forest)",
                f"{pred_rf:.2f}%"
            )

            col2.metric(
                "Predicción Próximo Mes (Gradient Boosting)",
                f"{pred_gb:.2f}%"
            )

            imp_df = pd.DataFrame({
                "Variable": X.columns,
                "Importancia RF": rf_model.feature_importances_,
                "Importancia GB": gb_model.feature_importances_
            })

            fig_imp = px.bar(
                imp_df,
                x="Variable",
                y=["Importancia RF", "Importancia GB"],
                barmode="group",
                template="plotly_white"
            )

            st.plotly_chart(
                fig_imp,
                width='stretch'
            )

    # ==========================================
    # TAB 7: MONTE CARLO
    # ==========================================

    with tab7:
        st.subheader(
            "Simulación Monte Carlo (Factor Fractal = 0.6366)"
        )

        n_escenarios = st.slider(
            "Simulaciones",
            10,
            100,
            50,
            key=f"sim_{asset_name}"
        )

        meses_proy = st.slider(
            "Meses a futuro",
            1,
            12,
            6,
            key=f"mes_{asset_name}"
        )

        sim_data = np.zeros(
            (meses_proy, n_escenarios)
        )

        p_fractal = 0.6366

        rng = np.random.default_rng()

        for e in range(n_escenarios):
            curr_v = last_val

            for m in range(meses_proy):
                step = (
                    rng.normal(mean_var, std_val)
                    * (
                        p_fractal
                        if rng.random() < p_fractal
                        else -p_fractal
                    )
                )

                curr_v = 0.7 * curr_v + 0.3 * step

                sim_data[m, e] = curr_v

        last_date = df["fecha"].iloc[-1]

        future_dates = [
            last_date + pd.DateOffset(months=i)
            for i in range(1, meses_proy + 1)
        ]

        fig_mc = go.Figure()

        for e in range(n_escenarios):
            fig_mc.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=sim_data[:, e],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.15,
                    showlegend=False
                )
            )

        fig_mc.add_trace(
            go.Scatter(
                x=future_dates,
                y=sim_data.mean(axis=1),
                mode="lines+markers",
                name="Consenso Promedio",
                line=dict(color="blue", width=3)
            )
        )

        fig_mc.update_layout(
            template="plotly_white",
            yaxis_title="Variación Mensual (%)",
            title=f"Simulación Monte Carlo ({meses_proy} meses)"
        )

        st.plotly_chart(
            fig_mc,
            width='stretch'
        )


# ==========================================
# 5. NAVEGACIÓN Y EJECUCIÓN
# ==========================================

st.sidebar.title("Navegación")

seleccion = st.sidebar.radio(
    "Seleccione la Base de Datos a Analizar:",
    [
        "Dólar Histórico",
        "Inflación (IPC)"
    ]
)

if seleccion == "Dólar Histórico":
    df_dolar = load_and_prep_dolar(
        "data/dolar.xlsx"
    )

    if df_dolar is not None:
        render_analysis(
            df_dolar,
            "Dólar"
        )

elif seleccion == "Inflación (IPC)":
    df_ipc = load_and_prep_ipc(
        "data/ipc.xlsx"
    )

    if df_ipc is not None:
        render_analysis(
            df_ipc,
            "IPC Nacional"
        )