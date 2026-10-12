import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import ccf, coint, grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA



warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Modelo Macro Bivariado & ML: Dólar vs IPC",
    page_icon=":material/robot_2:",
    layout="wide"
)

st.subheader(":material/robot_2: Análisis Bivariado Macro: Espín, Econometría & Aprendizaje Automático")
st.info("se trabaja con dos variables en simultaneo dolar e ipc (inflación)- período mensual")

# ==========================================
# 1. CARGA DE DATOS ROBUSTA
# ==========================================
@st.cache_data
def load_data():
    path = 'data/dolar-ipc.xlsx' if os.path.exists('data/dolar-ipc.xlsx') else 'dolar-ipc.xlsx'
    try:
        df = pd.read_excel(path, sheet_name=0)
        date_col = [c for c in df.columns if 'fecha' in str(c).lower() or 'periodo' in str(c).lower()][0]
        dolar_col = [c for c in df.columns if 'dolar' in str(c).lower() or 'venta' in str(c).lower()][0]
        ipc_col = [c for c in df.columns if 'ipc' in str(c).lower() or 'general' in str(c).lower() or 'total' in str(c).lower()][-1]
        
        df['fecha'] = pd.to_datetime(df[date_col], errors='coerce')
        df['Dolar_var'] = pd.to_numeric(df[dolar_col], errors='coerce')
        df['IPC_var'] = pd.to_numeric(df[ipc_col], errors='coerce')
        
        return df.dropna(subset=['fecha', 'Dolar_var', 'IPC_var']).sort_values('fecha').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error cargando base de datos conjunta: {e}")
        return None

uploaded_file = st.sidebar.file_uploader("Cargar archivo Excel personalizado", type=["xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['Dolar_var'] = df['Dolar-venta'] if 'Dolar-venta' in df.columns else df.iloc[:, 1]
        df['IPC_var'] = df['TotalNacional-Nivelgeneral'] if 'TotalNacional-Nivelgeneral' in df.columns else df.iloc[:, 2]
        df = df.sort_values('fecha').reset_index(drop=True)
    except Exception as e:
        st.error(f"Error procesando archivo subido: {e}")
        df = None
else:
    df = load_data()

if df is not None:
    mean_dolar, mean_ipc = df['Dolar_var'].mean(), df['IPC_var'].mean()
    std_dolar, std_ipc = df['Dolar_var'].std(), df['IPC_var'].std()
    corr = df['Dolar_var'].corr(df['IPC_var'])

    # Clasificación Teórica ARAD
    def classify_arad(dolar, ipc, m_d, m_i):
        if dolar > m_d and ipc > m_i:
            return 'Guerrero (G) - Expansión'
        elif dolar <= m_d and ipc > m_i:
            return 'Intelectual (I) - Ajuste'
        elif dolar > m_d and ipc <= m_i:
            return 'Logrero (L) - Agotamiento'
        else:
            return 'Menesteroso (M) - Estabilización'

    df['Estado_ARAD'] = df.apply(lambda r: classify_arad(r['Dolar_var'], r['IPC_var'], mean_dolar, mean_ipc), axis=1)

    # Sidebar Filtros
    st.sidebar.subheader(":material/settings: Filtros de Fecha")
    min_date, max_date = df['fecha'].min().date(), df['fecha'].max().date()
    start_date, end_date = st.sidebar.date_input("Rango de análisis", [min_date, max_date], min_value=min_date, max_value=max_date)

    df_filtered = df[(df['fecha'].dt.date >= start_date) & (df['fecha'].dt.date <= end_date)].copy()

    # Pestañas Principales
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        ":material/place_item: Fases y Espín", 
        ":material/place_item: Evolución Temporal", 
        ":material/place_item: Regímenes (K-Means vs ARAD)",
        ":material/place_item: Causalidad Granger & CCF",
        ":material/place_item: Cointegración de Largo Plazo",
        ":material/place_item: Escenarios opcionales",
        ":material/place_item: Monte Carlo Fractal",
        ":material/place_item: Modelos ML (VAR & LSTM)"
    ])

    with tab1:
        st.subheader(":material/list_alt: Diagrama de Fases Bivariado: Órbita Dólar vs IPC & Cuadrantes ARAD")
        
        # Identificar el último dato
        ultimo_dato = df_filtered.iloc[-1]
        
        fig_phase = px.scatter(
            df_filtered, x='Dolar_var', y='IPC_var', color='Estado_ARAD',
            hover_data=['fecha', 'Dolar_var', 'IPC_var'],
            labels={'Dolar_var': 'Variación Dólar (%)', 'IPC_var': 'IPC Mensual (%)'},
            template="plotly_white"
        )
        
        # Resaltar el último dato con una estrella o marcador grande
        fig_phase.add_trace(go.Scatter(
            x=[ultimo_dato['Dolar_var']], y=[ultimo_dato['IPC_var']],
            mode='markers+text',
            marker=dict(color='gold', size=16, symbol='star', line=dict(color='black', width=1)),
            text=["¡Último Dato!"],
            textposition="top center",
            name='Posición Actual'
        ))
        
        fig_phase.add_vline(x=mean_dolar, line_dash="dash", line_color="gray", annotation_text="Media Dólar")
        fig_phase.add_hline(y=mean_ipc, line_dash="dash", line_color="gray", annotation_text="Media IPC")
        
        st.plotly_chart(fig_phase, width='stretch')
        
        # Comentario Analítico Automático
        st.markdown(f"""
        ### Diagnóstico del Estado Actual (Último Registro: {pd.to_datetime(ultimo_dato['fecha']).strftime('%Y-%m-%d')})
        * **Cuadrante / Estado ARAD:** `{ultimo_dato['Estado_ARAD']}`
        * **Variación Dólar:** `{ultimo_dato['Dolar_var']:.2f}%` (Media histórica: `{mean_dolar:.2f}%`)
        * **Variación IPC:** `{ultimo_dato['IPC_var']:.2f}%` (Media histórica: `{mean_ipc:.2f}%`)
        * **Lectura para el Analista:** El sistema se encuentra operando en el cuadrante de {ultimo_dato['Estado_ARAD']}. 
        """)



    # TAB 2: COMPARATIVA TEMPORAL
    with tab2:
        st.subheader(":material/list_alt: Evolución Temporal Comparada")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=df_filtered['fecha'], y=df_filtered['Dolar_var'], mode='lines+markers', name='Var. Dólar (%)', line=dict(color='blue')))
        fig_time.add_trace(go.Scatter(x=df_filtered['fecha'], y=df_filtered['IPC_var'], mode='lines+markers', name='IPC Mensual (%)', line=dict(color='red')))
        fig_time.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_time, width='stretch')

        c1, c2, c3 = st.columns(3)
        c1.metric("Media Dólar", f"{mean_dolar:.2f}%")
        c2.metric("Media IPC", f"{mean_ipc:.2f}%")
        c3.metric("Correlación de Pearson", f"{corr:.4f}")

    # TAB 3: CLUSTERING K-MEANS PARA ARAD
    with tab3:
        st.subheader(":material/list_alt: Clusters No Supervisados (K-Means) vs. Matriz ARAD")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filtered[['Dolar_var', 'IPC_var']])
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_filtered['Cluster_KMeans'] = [f"Cluster {c}" for c in kmeans.fit_predict(X_scaled)]
        
        fig_km = px.scatter(
            df_filtered, x='Dolar_var', y='IPC_var', color='Cluster_KMeans', symbol='Estado_ARAD',
            hover_data=['fecha'], title="Regímenes Automáticos K-Means vs ARAD",
            template="plotly_white"
        )
        st.plotly_chart(fig_km, width='stretch')

    # TAB 4: CAUSALIDAD DE GRANGER Y CCF (NUEVO)
    with tab4:
        st.subheader(":material/list_alt: Causalidad de Granger y Función de Correlación Cruzada (CCF)")
        st.markdown("Determinación de la dirección de transmisión y los rezagos temporales clave.")

        col_g, col_ccf = st.columns(2)

        with col_g:
            st.markdown("### Test de Causalidad de Granger")
            data_gc = df_filtered[['IPC_var', 'Dolar_var']].dropna()
            
            gc_res_d_to_i = grangercausalitytests(data_gc[['IPC_var', 'Dolar_var']], maxlag=3) #, verbose=False
            gc_res_i_to_d = grangercausalitytests(data_gc[['Dolar_var', 'IPC_var']], maxlag=3) #, verbose=False

            p_vals_dict = {
                'Lag (Meses)': [1, 2, 3],
                'Dólar ➔ IPC (p-val)': [gc_res_d_to_i[l][0]['ssr_ftest'][1] for l in range(1, 4)],
                'IPC ➔ Dólar (p-val)': [gc_res_i_to_d[l][0]['ssr_ftest'][1] for l in range(1, 4)]
            }
            df_pvals = pd.DataFrame(p_vals_dict)
            st.dataframe(df_pvals.style.format({'Dólar ➔ IPC (p-val)': '{:.4f}', 'IPC ➔ Dólar (p-val)': '{:.4f}'}))
            st.info("💡 Un p-valor < 0.05 indica que la primera variable ayuda a predecir a la segunda con significancia estadística.")

        with col_ccf:
            st.markdown("### Correlación Cruzada (CCF)")
            ccf_vals = ccf(df_filtered['IPC_var'], df_filtered['Dolar_var'])[:10]
            
            fig_ccf = go.Figure(go.Bar(
                x=list(range(10)), y=ccf_vals,
                marker_color='teal', name='Correlación'
            ))
            fig_ccf.update_layout(
                title="Efecto del Dólar (mes t-k) sobre el IPC (mes t)",
                xaxis_title="Rezago k (Meses de atraso)",
                yaxis_title="Coeficiente de Correlación",
                template="plotly_white"
            )
            st.plotly_chart(fig_ccf, width='stretch')

    # TAB 5: COINTEGRACIÓN (NUEVO)
    with tab5:
        st.subheader(":material/list_alt: Cointegración de Engle-Granger (Equilibrio de Largo Plazo)")
        st.markdown("Evaluación de la relación de equilibrio estructural entre las series acumuladas.")
        
        series_dolar_cum = (1 + df_filtered['Dolar_var']/100).cumprod()
        series_ipc_cum = (1 + df_filtered['IPC_var']/100).cumprod()
        
        score, pvalue, _ = coint(series_dolar_cum, series_ipc_cum)
        
        st.metric("P-Valor Test de Cointegración", f"{pvalue:.4f}")
        if pvalue < 0.05:
            st.success(":material/check_box: Existe Cointegración de Largo Plazo (P-valor < 0.05). Las series comparten un equilibrio estructural.")
        else:
            st.warning(":material/warning: No se detecta Cointegración de Largo Plazo estadísticamente fuerte (P-valor >= 0.05). Son más propensas a divergencias en el corto plazo.")

        fig_coint = go.Figure()
        fig_coint.add_trace(go.Scatter(x=df_filtered['fecha'], y=series_dolar_cum, name='Índice Acumulado Dólar', line=dict(color='blue')))
        fig_coint.add_trace(go.Scatter(x=df_filtered['fecha'], y=series_ipc_cum, name='Índice Acumulado IPC', line=dict(color='red')))
        fig_coint.update_layout(title="Trayectorias Acumuladas Comparadas (Base 1)", template="plotly_white")
        st.plotly_chart(fig_coint, width='stretch')

    # TAB 6: FINANFOR
    with tab6:
        st.subheader(":material/list_alt: Escenario de Abanico")
        last_dolar, last_ipc = df['Dolar_var'].iloc[-1], df['IPC_var'].iloc[-1]
        v_d, v_i = df['Dolar_var'].abs().mean(), df['IPC_var'].abs().mean()
        
        escenarios_dict = {
            'Segunda Baja (-2x VarMed)': (last_dolar - 2*v_d, last_ipc - 2*v_i),
            'Primera Baja (-1x VarMed)': (last_dolar - v_d, last_ipc - v_i),
            'Último Dato Actual': (last_dolar, last_ipc),
            'Primera Alza (+1x VarMed)': (last_dolar + v_d, last_ipc + v_i),
            'Segunda Alza (+2x VarMed)': (last_dolar + 2*v_d, last_ipc + 2*v_i)
        }
        df_esc = pd.DataFrame(escenarios_dict, index=['Proyección Dólar (%)', 'Proyección IPC (%)']).T
        st.table(df_esc)

    # TAB 7: MONTE CARLO FRACTAL
    with tab7:
        st.subheader(":material/list_alt: Simulación Monte Carlo Fractal (p = 0.6366)")
        n_esc = st.slider("Simulaciones", 30, 150, 60)
        m_proy = st.slider("Meses Proyectados", 3, 24, 6)
        
        sim_d = np.zeros((m_proy, n_esc))
        sim_i = np.zeros((m_proy, n_esc))
        p_f = 0.6366
        
        np.random.seed(42)
        for e in range(n_esc):
            cd, ci = last_dolar, last_ipc
            for m in range(m_proy):
                sd = np.random.normal(mean_dolar, std_dolar) * (p_f if np.random.rand() < p_f else -p_f)
                si = np.random.normal(mean_ipc, std_ipc) * (p_f if np.random.rand() < p_f else -p_f)
                cd = 0.7 * cd + 0.3 * sd
                ci = 0.7 * ci + 0.3 * si
                sim_d[m, e], sim_i[m, e] = cd, ci

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(y=sim_d.mean(axis=1), mode='lines+markers', name='Consenso Dólar (%)', line=dict(color='blue', width=3)))
        fig_mc.add_trace(go.Scatter(y=sim_i.mean(axis=1), mode='lines+markers', name='Consenso IPC (%)', line=dict(color='red', width=3)))
        fig_mc.update_layout(template="plotly_white", yaxis_title="Variación Mensual (%)")
        st.plotly_chart(fig_mc, width='stretch')

    # TAB 8: ECONOMETRÍA Y MACHINE LEARNING (VAR & LSTM)
    with tab8:
        st.subheader(":material/list_alt: Modelos Avanzados Bivariados: VAR & LSTM")
        col_v, col_l = st.columns(2)
        
        with col_v:
            st.markdown("### Vector Autorregresivo (VAR)")
            var_data = df_filtered[['Dolar_var', 'IPC_var']].dropna()
            try:
                model_var = VAR(var_data)
                lag_order = model_var.select_order(maxlags=12)
                optimal_lag = lag_order.aic
                results_var = model_var.fit(optimal_lag)
                results_var = model_var.fit(maxlags=2)

                irf = results_var.irf(6)
                
                fig_irf = px.line(
                    x=list(range(7)), y=irf.irfs[:, 1, 0], markers=True,
                    title="Función Impulso-Respuesta: Transmisión Dólar ➔ IPC",
                    labels={'x': 'Meses tras el Shock', 'y': 'Respuesta IPC (%)'}
                )
                st.plotly_chart(fig_irf, width='stretch')
                
                var_pred = results_var.forecast(var_data.values[-results_var.k_ar:], steps=6)
                st.dataframe(pd.DataFrame(var_pred, columns=['VAR Dólar (%)', 'VAR IPC (%)']).style.format("{:.2f}%"))
            except Exception as e:
                st.error(f"Error VAR: {e}")

        with col_l:
            st.markdown("### Red Neuronal Recurrente (LSTM Bivariada)")
            data_mat = df_filtered[['Dolar_var', 'IPC_var']].dropna().values
            seq_len = 3
            
            scaler_mean, scaler_std = data_mat.mean(axis=0), data_mat.std(axis=0) + 1e-8
            norm_mat = (data_mat - scaler_mean) / scaler_std
            
            X_seq, y_seq = [], []
            for i in range(len(norm_mat) - seq_len):
                X_seq.append(norm_mat[i:i+seq_len])
                y_seq.append(norm_mat[i+seq_len])
                
            X_t = torch.tensor(np.array(X_seq), dtype=torch.float32)
            y_t = torch.tensor(np.array(y_seq), dtype=torch.float32)
            
            class BivariateLSTM(nn.Module):                
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True, dropout=0.2, num_layers=2)
                    self.fc = nn.Linear(32, 2)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            
            net = BivariateLSTM()
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            crit = nn.MSELoss()
            
            for _ in range(80):
                opt.zero_grad()
                loss = crit(net(X_t), y_t)
                loss.backward()
                opt.step()
                
            preds = []
            curr_seq = norm_mat[-seq_len:].copy()
            net.eval()
            with torch.no_grad():
                for _ in range(6):
                    inp = torch.tensor(curr_seq[np.newaxis, :, :], dtype=torch.float32)
                    p = net(inp).numpy()[0]
                    preds.append(p)
                    curr_seq = np.vstack([curr_seq[1:], p])
                    
            lstm_preds = np.array(preds) * scaler_std + scaler_mean
            df_lstm_pred = pd.DataFrame(lstm_preds, columns=['LSTM Dólar (%)', 'LSTM IPC (%)'])
            
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(y=df_lstm_pred['LSTM Dólar (%)'], mode='lines+markers', name='LSTM Dólar (%)', line=dict(color='blue')))
            fig_lstm.add_trace(go.Scatter(y=df_lstm_pred['LSTM IPC (%)'], mode='lines+markers', name='LSTM IPC (%)', line=dict(color='red')))
            fig_lstm.update_layout(title="Proyección LSTM a 6 Meses", template="plotly_white")
            st.plotly_chart(fig_lstm, width='stretch')
            st.dataframe(df_lstm_pred.style.format("{:.2f}%"))



with st.expander("Suite Multivariada: Nowcasting, VARX & Backtesting Out-of-Sample"):
     
    #==========================================
    # 1. ALINEACIÓN Y NOWCASTING (DESCALCE DE FRECUENCIAS)
    # ==========================================

    @st.cache_data
    def generar_datos_simulados():
        """Genera datos simulados diarios y mensuales para demostrar Nowcasting y Exógenas"""
        # Cambia el año de finalización de 2023 a 2026
        fechas_diarias = pd.date_range(start='2020-01-01', end='2026-09-30', freq='D')
        fechas_mensuales = pd.date_range(start='2020-01-01', end='2026-09-30', freq='ME')
        
        dolar_diario = np.cumsum(np.random.normal(0.1, 1.5, len(fechas_diarias))) + 100
        badlar_diaria = np.cumsum(np.random.normal(0.01, 0.5, len(fechas_diarias))) + 40
        df_diario = pd.DataFrame({'fecha': fechas_diarias, 'Dolar': dolar_diario, 'BADLAR': badlar_diaria})
        
        ipc_mensual = np.random.normal(5.0, 2.0, len(fechas_mensuales))
        m2_mensual = np.cumsum(np.random.normal(1000, 500, len(fechas_mensuales))) + 50000
        df_mensual = pd.DataFrame({'fecha': fechas_mensuales, 'IPC_var': ipc_mensual, 'M2': m2_mensual})
        
        return df_diario, df_mensual

    def alinear_frecuencias(df_diario, df_mensual, metodo='promedio'):
        """Resuelve el descalce llevando series diarias a mensuales (Agregación)"""
        df_diario = df_diario.set_index('fecha')
        df_mensual = df_mensual.set_index('fecha')
        
        if metodo == 'promedio':
            df_diario_resampled = df_diario.resample('ME').mean()
        else: # fin de mes
            df_diario_resampled = df_diario.resample('ME').last()
            
        df_diario_resampled['Dolar_var'] = df_diario_resampled['Dolar'].pct_change() * 100
        
        # Unificar el dataset
        df_final = df_mensual.join(df_diario_resampled, how='inner').dropna().reset_index()
        return df_final

    # ==========================================
    # Carga y Procesamiento
    # ==========================================
    df_diario, df_mensual = generar_datos_simulados()
    df_master = alinear_frecuencias(df_diario, df_mensual, metodo='promedio')

    tab1, tab2, tab3 = st.tabs([
        ":material/modeling: Nowcasting & Frecuencias", 
        ":material/modeling: Variables Exógenas (VARX)", 
        ":material/modeling: Backtesting (Out-of-Sample)"
    ])

    # TAB 1: NOWCASTING
    with tab1:
        st.subheader("Tratamiento de Descalce de Frecuencias (Nowcasting)")
        st.markdown("Agregación de variables diarias (Dólar, BADLAR) para alinear con publicaciones mensuales estructurales (IPC, Base Monetaria M2).")
        
        fig_now = go.Figure()
        fig_now.add_trace(go.Scatter(x=df_diario['fecha'], y=df_diario['Dolar'], opacity=0.3, name='Dólar Diario', line=dict(color='gray')))
        fig_now.add_trace(go.Scatter(x=df_master['fecha'], y=df_master['Dolar'], mode='lines+markers', name='Dólar Mensual (Agregado)', line=dict(color='blue')))
        fig_now.update_layout(template="plotly_white", title="Transformación Diario ➔ Mensual")
        st.plotly_chart(fig_now, width='stretch')

    # TAB 2: EXÓGENAS VARX / ARIMAX
    with tab2:
        st.subheader("Modelado Multivariado (Inclusión de M2 y BADLAR)")
        
        var_data = df_master[['Dolar_var', 'IPC_var', 'M2', 'BADLAR']].dropna()
        model_var = VAR(var_data)
        results_var = model_var.fit(maxlags=2)
        
        irf = results_var.irf(6)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_irf1 = px.line(x=list(range(7)), y=irf.irfs[:, var_data.columns.get_loc('M2'), var_data.columns.get_loc('IPC_var')], 
                            markers=True, title="Impulso-Respuesta: Shock en M2 ➔ Efecto en IPC")
            st.plotly_chart(fig_irf1, width='stretch')
        with col2:
            fig_irf2 = px.line(x=list(range(7)), y=irf.irfs[:, var_data.columns.get_loc('BADLAR'), var_data.columns.get_loc('Dolar_var')], 
                            markers=True, title="Impulso-Respuesta: Shock en BADLAR ➔ Efecto en Dólar")
            st.plotly_chart(fig_irf2, width='stretch')

    # TAB 3: BACKTESTING OUT-OF-SAMPLE
    with tab3:
        st.subheader("Evaluación Competitiva de Modelos (Holdout de 6 Meses)")
        
        # Split Train/Test (Últimos 6 meses para testing)
        horizonte = 6
        train_df = df_master.iloc[:-horizonte]
        test_df = df_master.iloc[-horizonte:]
        y_true_ipc = test_df['IPC_var'].values
        
        metricas = []

    # 1. ARIMA(1,1,1) Univariado
        arima = ARIMA(train_df['IPC_var'], order=(1,1,1)).fit()
        pred_arima = arima.forecast(steps=horizonte).values
        metricas.append({"Modelo": "ARIMA Univariado", 
                        "RMSE": np.sqrt(mean_squared_error(y_true_ipc, pred_arima)),
                        "MAE": mean_absolute_error(y_true_ipc, pred_arima),
                        "MAPE": mean_absolute_percentage_error(y_true_ipc, pred_arima)})

        # 2. VAR Bivariado
        var_train = train_df[['IPC_var', 'Dolar_var']]
        var_mod = VAR(var_train).fit(maxlags=2)
        pred_var = var_mod.forecast(var_train.values[-2:], steps=horizonte)[:, 0] # Índice 0 = IPC_var
        metricas.append({"Modelo": "VAR Bivariado", 
                        "RMSE": np.sqrt(mean_squared_error(y_true_ipc, pred_var)),
                        "MAE": mean_absolute_error(y_true_ipc, pred_var),
                        "MAPE": mean_absolute_percentage_error(y_true_ipc, pred_var)})

        # 3. Random Forest (Multivariado con Rezagos)

        # Corrección: Se utiliza 'M2' en lugar de 'M2_var'
        rf_data = df_master[['IPC_var', 'Dolar_var', 'M2', 'BADLAR']].copy()
        rf_data['IPC_Lag1'] = rf_data['IPC_var'].shift(1)
        rf_data['Dolar_Lag1'] = rf_data['Dolar_var'].shift(1)
        rf_data = rf_data.dropna()
        
        rf_train = rf_data.iloc[:-horizonte]
        rf_test = rf_data.iloc[-horizonte:]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(rf_train[['IPC_Lag1', 'Dolar_Lag1', 'M2', 'BADLAR']], rf_train['IPC_var'])
        pred_rf = rf.predict(rf_test[['IPC_Lag1', 'Dolar_Lag1', 'M2', 'BADLAR']])
        
        metricas.append({"Modelo": "Random Forest (Multivariado)", 
                        "RMSE": np.sqrt(mean_squared_error(y_true_ipc, pred_rf)),
                        "MAE": mean_absolute_error(y_true_ipc, pred_rf),
                        "MAPE": mean_absolute_percentage_error(y_true_ipc, pred_rf)})    

        # Tabla de Resultados
        df_metricas = pd.DataFrame(metricas).set_index("Modelo")
        st.markdown("**Métricas de Error Fuera de Muestra (Predecir IPC Mensual)**")
        st.dataframe(df_metricas.style.highlight_min(axis=0, color='lightgreen'))
        
        # Gráfico Comparativo
        fig_backtest = go.Figure()
        fechas_test = test_df['fecha']
        fig_backtest.add_trace(go.Scatter(x=fechas_test, y=y_true_ipc, mode='lines+markers', name='IPC Real', line=dict(color='black', width=3)))
        fig_backtest.add_trace(go.Scatter(x=fechas_test, y=pred_arima, mode='lines', name='ARIMA', line=dict(dash='dash')))
        fig_backtest.add_trace(go.Scatter(x=fechas_test, y=pred_var, mode='lines', name='VAR', line=dict(dash='dash')))
        fig_backtest.add_trace(go.Scatter(x=fechas_test, y=pred_rf, mode='lines', name='Random Forest', line=dict(dash='dash')))
        
        fig_backtest.update_layout(template="plotly_white", title="Predicciones vs. Valores Reales (Periodo Holdout)")
        st.plotly_chart(fig_backtest, width='stretch')