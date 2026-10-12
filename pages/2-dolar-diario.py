import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(
    page_title="Dashboard Cuantitativo: Dólar Diario & Gestión de Riesgo",
    page_icon="material/paid",
    layout="wide",
)

st.subheader(
    ":material/paid: Dashboard Cuantitativo Avanzado: Análisis Diario, Técnico, Riesgo & Machine Learning"
)

# Resultados reproducibles y entrenamiento más estable
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ==========================================
def preparar_dataframe(df):
    """Normaliza nombres, tipos, fechas y métricas base del Excel."""
    if df is None or df.empty:
        raise ValueError("El archivo no contiene registros.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    requeridas = {"fecha", "compra", "venta"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltantes)}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Excel puede entregar fechas como datetime, texto o número de serie.
    if pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    elif pd.api.types.is_numeric_dtype(df["fecha"]):
        # Excel utiliza como origen de fechas el 30/12/1899.
        df["fecha"] = pd.to_datetime(
            df["fecha"], unit="D", origin="1899-12-30", errors="coerce"
        )
    else:
        df["fecha"] = pd.to_datetime(
            df["fecha"], errors="coerce", dayfirst=True
        )

    for columna in ("compra", "venta"):
        # Soporta columnas numéricas y números guardados como texto.
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    if "promedio" in df.columns:
        promedio_excel = pd.to_numeric(df["promedio"], errors="coerce")
        df["promedio"] = promedio_excel.where(
            promedio_excel.notna(), (df["compra"] + df["venta"]) / 2.0
        )
    else:
        df["promedio"] = (df["compra"] + df["venta"]) / 2.0

    df = df.dropna(subset=["fecha", "compra", "venta", "promedio"])
    df = df[
        (df["compra"] > 0)
        & (df["venta"] > 0)
        & (df["promedio"] > 0)
    ].copy()

    df = (
        df.sort_values("fecha")
        .drop_duplicates(subset=["fecha"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No quedaron filas válidas después de limpiar los datos.")

    df["spread"] = df["venta"] - df["compra"]
    df["spread_pct"] = np.where(
        df["promedio"] != 0,
        df["spread"] / df["promedio"] * 100,
        np.nan,
    )
    # Retorno logarítmico expresado en porcentaje.
    df["retorno_log"] = (
        np.log(df["promedio"] / df["promedio"].shift(1)) * 100
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


@st.cache_data
def load_data(ruta):
    """Carga el Excel local y prepara los datos."""
    df_excel = pd.read_excel(ruta, sheet_name=0)
    return preparar_dataframe(df_excel)


# Primero se permite cargar un Excel propio; de lo contrario se buscan
# las rutas habituales del proyecto.
uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo Excel personalizado", type=["xlsx"]
)

df = None
if uploaded_file is not None:
    try:
        df = preparar_dataframe(pd.read_excel(uploaded_file, sheet_name=0))
        st.sidebar.success("Archivo Excel cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo procesar el archivo subido: {e}")
else:
    posibles_rutas = [
        "data/dolar_diario.xlsx",
        "dolar_diario.xlsx",
        "data/dolar-diario.xlsx",
        "dolar-diario.xlsx",
    ]
    ruta = next((p for p in posibles_rutas if os.path.isfile(p)), None)
    if ruta is None:
        st.error(
            "No se encontró el archivo Excel. Colocá dolar-diario.xlsx "
            "en la carpeta del proyecto o cargalo desde la barra lateral."
        )
    else:
        try:
            df = load_data(ruta)
        except Exception as e:
            st.error(f"Error cargando el archivo Excel: {e}")


# ==========================================
# 2. FILTROS Y VALIDACIONES
# ==========================================
if df is not None and not df.empty:
    st.sidebar.subheader(":material/settings: Filtros de fecha")

    # Convertir explícitamente a datetime.date para evitar incompatibilidades
    # del date_input con tipos Timestamp, numpy o valores no normalizados.
    min_date = pd.Timestamp(df["fecha"].min()).date()
    max_date = pd.Timestamp(df["fecha"].max()).date()

    if min_date == max_date:
        st.sidebar.info(f"Única fecha disponible: {min_date:%d/%m/%Y}")
        start_date, end_date = min_date, max_date
    else:
        rango = st.sidebar.date_input(
            "Rango de análisis",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Streamlit puede devolver una tupla/lista de dos fechas o una sola
        # fecha mientras el usuario todavía está seleccionando el intervalo.
        if isinstance(rango, (tuple, list)):
            if len(rango) == 2:
                start_date, end_date = rango
            elif len(rango) == 1:
                start_date, end_date = rango[0], max_date
            else:
                start_date, end_date = min_date, max_date
        else:
            start_date, end_date = rango, max_date

        start_date = pd.Timestamp(start_date).date()
        end_date = pd.Timestamp(end_date).date()

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    df_filtered = df[
        (df["fecha"].dt.date >= start_date)
        & (df["fecha"].dt.date <= end_date)
    ].copy().reset_index(drop=True)

    if df_filtered.empty:
        st.warning("No hay datos dentro del rango seleccionado. Modificá las fechas.")
        st.stop()

    st.caption(
        f"Período analizado: {start_date:%d/%m/%Y} al {end_date:%d/%m/%Y} "
        f"· {len(df_filtered):,} registros".replace(",", ".")
    )

    # ==========================================
    # 3. PESTAÑAS
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Evolución & Spread",
            "Análisis Técnico",
            "Riesgo & Volatilidad",
            "Monte Carlo Estocástico",
            "Machine Learning (LSTM)",
        ]
    )

    # ==========================================
    # TAB 1: EVOLUCIÓN & SPREAD
    # ==========================================
    with tab1:
        st.subheader("Evolución temporal de cotizaciones y spread")

        fig_time = go.Figure()
        fig_time.add_trace(
            go.Scatter(
                x=df_filtered["fecha"], y=df_filtered["venta"],
                mode="lines", name="Dólar Venta",
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=df_filtered["fecha"], y=df_filtered["compra"],
                mode="lines", name="Dólar Compra",
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=df_filtered["fecha"], y=df_filtered["promedio"],
                mode="lines", name="Promedio", line=dict(dash="dot"),
            )
        )
        fig_time.update_layout(
            template="plotly_white",
            hovermode="x unified",
            title="Cotizaciones diarias",
            xaxis_title="Fecha",
            yaxis_title="Cotización ($)",
        )
        st.plotly_chart(fig_time, width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último dólar venta", f"${df_filtered['venta'].iloc[-1]:,.2f}")
        c2.metric("Último dólar compra", f"${df_filtered['compra'].iloc[-1]:,.2f}")
        c3.metric("Spread promedio", f"${df_filtered['spread'].mean():,.2f}")
        c4.metric("Spread relativo promedio", f"{df_filtered['spread_pct'].mean():.2f}%")

        fig_spread = px.line(
            df_filtered,
            x="fecha",
            y="spread",
            title="Evolución del spread (venta − compra)",
            labels={"fecha": "Fecha", "spread": "Spread ($)"},
        )
        fig_spread.update_layout(template="plotly_white")
        st.plotly_chart(fig_spread, width='stretch')

        with st.expander("Ver datos del período"):
            st.dataframe(df_filtered, width='stretch', hide_index=True)

    # ==========================================
    # TAB 2: ANÁLISIS TÉCNICO
    # ==========================================
    with tab2:
        st.subheader("Indicadores técnicos: EMA y Bandas de Bollinger")

        tecnico = df_filtered.copy()
        tecnico["ema_20"] = tecnico["promedio"].ewm(span=20, adjust=False).mean()
        tecnico["ema_50"] = tecnico["promedio"].ewm(span=50, adjust=False).mean()
        media_20 = tecnico["promedio"].rolling(window=20, min_periods=20).mean()
        rolling_std = tecnico["promedio"].rolling(window=20, min_periods=20).std()
        tecnico["bb_upper"] = media_20 + 2 * rolling_std
        tecnico["bb_lower"] = media_20 - 2 * rolling_std

        fig_tec = go.Figure()
        fig_tec.add_trace(
            go.Scatter(
                x=tecnico["fecha"], y=tecnico["promedio"],
                mode="lines", name="Precio promedio",
            )
        )
        fig_tec.add_trace(
            go.Scatter(x=tecnico["fecha"], y=tecnico["ema_20"], mode="lines", name="EMA 20")
        )
        fig_tec.add_trace(
            go.Scatter(x=tecnico["fecha"], y=tecnico["ema_50"], mode="lines", name="EMA 50")
        )
        fig_tec.add_trace(
            go.Scatter(
                x=tecnico["fecha"], y=tecnico["bb_upper"],
                mode="lines", name="Bollinger superior",
            )
        )
        fig_tec.add_trace(
            go.Scatter(
                x=tecnico["fecha"], y=tecnico["bb_lower"],
                mode="lines", name="Bollinger inferior",
                fill="tonexty", fillcolor="rgba(150,150,150,0.15)",
            )
        )
        fig_tec.update_layout(
            template="plotly_white",
            title="Tendencia y bandas de volatilidad",
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title="Cotización ($)",
        )
        st.plotly_chart(fig_tec, width='stretch')

        ultimo = tecnico.iloc[-1]
        if pd.notna(ultimo["ema_20"]) and pd.notna(ultimo["ema_50"]):
            if ultimo["ema_20"] > ultimo["ema_50"]:
                st.info("Lectura técnica descriptiva: EMA 20 por encima de EMA 50.")
            elif ultimo["ema_20"] < ultimo["ema_50"]:
                st.info("Lectura técnica descriptiva: EMA 20 por debajo de EMA 50.")
            else:
                st.info("Lectura técnica descriptiva: EMA 20 y EMA 50 coinciden.")
        st.caption("Los indicadores son descriptivos; no constituyen una señal garantizada de compra o venta.")

    # ==========================================
    # TAB 3: RIESGO & VOLATILIDAD
    # ==========================================
    with tab3:
        st.subheader("Métricas de riesgo cuantitativo")

        retornos = df_filtered["retorno_log"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(retornos) < 2:
            st.warning("Se necesitan al menos tres observaciones válidas para estimar el riesgo.")
        else:
            vol_anual = retornos.std(ddof=1) * np.sqrt(252)
            var_95 = np.percentile(retornos, 5)
            var_99 = np.percentile(retornos, 1)

            # Para el drawdown se construye un índice de crecimiento con retornos
            # logarítmicos. Es una métrica de trayectoria, no una pérdida monetaria.
            retorno_decimal = retornos / 100.0
            acumulado = np.exp(retorno_decimal.cumsum())
            pico = acumulado.cummax()
            drawdown = (acumulado / pico) - 1.0
            mdd = drawdown.min() * 100.0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Volatilidad anualizada", f"{vol_anual:.2f}%")
            r2.metric("VaR diario 95% (histórico)", f"{var_95:.2f}%")
            r3.metric("VaR diario 99% (histórico)", f"{var_99:.2f}%")
            r4.metric("Máxima caída (MDD)", f"{mdd:.2f}%")

            st.caption(
                "El VaR se presenta como percentil histórico de retornos: "
                "valores más negativos representan jornadas peores. La anualización "
                "usa 252 ruedas como aproximación."
            )

            df_dd = pd.DataFrame({
                "fecha": df_filtered.loc[retornos.index, "fecha"].values,
                "drawdown_pct": drawdown.values * 100,
            })
            fig_dd = px.line(
                df_dd, x="fecha", y="drawdown_pct",
                title="Evolución del drawdown histórico (%)",
                labels={"fecha": "Fecha", "drawdown_pct": "Drawdown (%)"},
            )
            fig_dd.update_layout(template="plotly_white")
            st.plotly_chart(fig_dd, width='stretch')

            cvar_95 = retornos[retornos <= var_95].mean()
            cvar_99 = retornos[retornos <= var_99].mean()
            k1, k2 = st.columns(2)
            k1.metric("CVaR diario 95%", f"{cvar_95:.2f}%")
            k2.metric("CVaR diario 99%", f"{cvar_99:.2f}%")

    # ==========================================
    # TAB 4: MONTE CARLO
    # ==========================================
    with tab4:
        st.subheader("Simulación Monte Carlo: movimiento browniano geométrico")
        n_sim = st.slider("Número de trayectorias", 50, 500, 100, key="mc_n_sim")
        dias_proy = st.slider("Días a proyectar", 10, 90, 30, key="mc_dias")

        precios = df_filtered["promedio"].astype(float)
        retornos_mc = np.log(precios / precios.shift(1)).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(retornos_mc) < 2:
            st.warning("Se necesitan más datos para realizar la simulación Monte Carlo.")
        else:
            ultimo_precio = float(precios.iloc[-1])
            drift = float(retornos_mc.mean())
            daily_vol = float(retornos_mc.std(ddof=1))

            rng = np.random.default_rng(42)
            shocks = rng.normal(0, 1, size=(dias_proy, n_sim))
            pasos = np.exp(
                (drift - 0.5 * daily_vol**2) + daily_vol * shocks
            )
            simulaciones = np.vstack([
                np.full((1, n_sim), ultimo_precio),
                ultimo_precio * np.cumprod(pasos, axis=0),
            ])

            fig_mc = go.Figure()
            for i in range(min(n_sim, 50)):
                fig_mc.add_trace(
                    go.Scatter(
                        x=np.arange(dias_proy + 1),
                        y=simulaciones[:, i],
                        mode="lines",
                        line=dict(width=0.8),
                        opacity=0.35,
                        showlegend=False,
                    )
                )
            fig_mc.add_trace(
                go.Scatter(
                    x=np.arange(dias_proy + 1),
                    y=np.median(simulaciones, axis=1),
                    mode="lines",
                    name="Mediana proyectada",
                    line=dict(width=3),
                )
            )
            fig_mc.update_layout(
                title=f"Simulación Monte Carlo a {dias_proy} días",
                xaxis_title="Días futuros",
                yaxis_title="Precio simulado ($)",
                template="plotly_white",
            )
            st.plotly_chart(fig_mc, width='stretch')

            finales = simulaciones[-1]
            q05, q50, q95 = np.percentile(finales, [5, 50, 95])
            m1, m2, m3 = st.columns(3)
            m1.metric("Percentil 5% final", f"${q05:,.2f}")
            m2.metric("Mediana final", f"${q50:,.2f}")
            m3.metric("Percentil 95% final", f"${q95:,.2f}")
            st.caption(
                "Escenarios simulados a partir de retornos históricos; no son "
                "predicciones ciertas ni contemplan cambios estructurales o shocks extraordinarios."
            )

    # ==========================================
    # TAB 5: MACHINE LEARNING (LSTM)
    # ==========================================
    with tab5:
        st.subheader("Red neuronal LSTM para precio promedio")
        st.warning(
            "Modelo experimental de serie temporal. La proyección es ilustrativa y "
            "no debe interpretarse como asesoramiento financiero."
        )

        seq_len = st.slider("Ventana histórica (días)", 5, 30, 10, key="lstm_seq")
        horizonte = st.slider("Días hábiles a proyectar", 1, 20, 10, key="lstm_horizonte")
        epochs = st.slider("Épocas de entrenamiento", 10, 150, 50, step=10, key="lstm_epochs")

        precios_lstm = df_filtered["promedio"].astype(float).values.reshape(-1, 1)

        if len(precios_lstm) <= seq_len + 10:
            st.warning(
                f"No hay suficientes datos filtrados. Para esta configuración "
                f"se recomiendan al menos {seq_len + 11} observaciones."
            )
        else:
            # Escalado solo sobre la serie seleccionada. Para evaluación rigurosa
            # fuera de muestra debe separarse entrenamiento y prueba temporal.
            scaler = StandardScaler()
            precios_norm = scaler.fit_transform(precios_lstm)

            X_seq, y_seq = [], []
            for i in range(len(precios_norm) - seq_len):
                X_seq.append(precios_norm[i:i + seq_len])
                y_seq.append(precios_norm[i + seq_len])

            X_t = torch.tensor(np.asarray(X_seq), dtype=torch.float32)
            y_t = torch.tensor(np.asarray(y_seq), dtype=torch.float32)

            class LSTMPrice(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=1,
                        hidden_size=32,
                        batch_first=True,
                        num_layers=2,
                        dropout=0.1,
                    )
                    self.fc = nn.Linear(32, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            net = LSTMPrice()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            with st.spinner("Entrenando red neuronal LSTM..."):
                net.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    pred = net(X_t)
                    loss = criterion(pred, y_t)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

            net.eval()
            secuencia = precios_norm[-seq_len:].copy()
            predicciones_norm = []

            with torch.no_grad():
                for _ in range(horizonte):
                    entrada = torch.tensor(
                        secuencia[np.newaxis, :, :], dtype=torch.float32
                    )
                    pred = float(net(entrada).item())
                    predicciones_norm.append(pred)
                    secuencia = np.vstack([secuencia[1:], [[pred]]])

            predicciones = scaler.inverse_transform(
                np.asarray(predicciones_norm).reshape(-1, 1)
            ).flatten()

            # Próximas fechas hábiles de lunes a viernes.
            ultima_fecha = pd.Timestamp(df_filtered["fecha"].iloc[-1])
            fechas_futuras = pd.bdate_range(
                start=ultima_fecha + pd.Timedelta(days=1),
                periods=horizonte,
            )
            df_pred = pd.DataFrame({
                "Fecha": fechas_futuras,
                "Predicción LSTM": predicciones,
            })

            fig_lstm = go.Figure()
            fig_lstm.add_trace(
                go.Scatter(
                    x=df_filtered["fecha"].tail(60),
                    y=df_filtered["promedio"].tail(60),
                    mode="lines",
                    name="Histórico reciente",
                )
            )
            fig_lstm.add_trace(
                go.Scatter(
                    x=df_pred["Fecha"],
                    y=df_pred["Predicción LSTM"],
                    mode="lines+markers",
                    name="Predicción LSTM",
                    line=dict(width=2, dash="dash"),
                )
            )
            fig_lstm.update_layout(
                title=f"Proyección LSTM: próximos {horizonte} días hábiles",
                template="plotly_white",
                hovermode="x unified",
                xaxis_title="Fecha",
                yaxis_title="Precio promedio ($)",
            )
            st.plotly_chart(fig_lstm, width='stretch')
            st.dataframe(
                df_pred.style.format({"Predicción LSTM": "${:,.2f}"}),
                width='stretch',
                hide_index=True,
            )

else:
    st.info(
        "Cargá un archivo Excel válido con las columnas fecha, compra y venta "
        "para comenzar el análisis."
    )
