import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from gmdh import GMDHRegressor
from Utils.AI_helper import connect_ai_model_results
from Utils.i18n import t


st.title("🧠 МГУА-регрессия (GMDH)")
st.caption("ℹ Метод группового учёта аргументов — самоорганизующаяся полиномиальная регрессия")

# --- Страховка от отсутствия df ---
df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("📥 Сначала загрузите данные.")
    st.stop()

# --- Инициализация состояния ---
if "gmdh_state" not in st.session_state:
    st.session_state["gmdh_state"] = {}

# --- Выбор целевой переменной ---
options = list(df.columns)
target_col = st.selectbox("🎯 Целевая переменная (числовая)", options)
if not target_col:
    st.error("❌ Не выбрана целевая переменная")
    st.stop()

if not pd.api.types.is_numeric_dtype(df[target_col]):
    st.error("❌ МГУА работает только с числовой целевой переменной (регрессия).")
    st.stop()

feature_cols = [c for c in df.columns if c != target_col]
if not feature_cols:
    st.error("Нет признаков для обучения")
    st.stop()

# --- Настройки модели ---
with st.expander("⚙️ Настройки модели", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        max_layers = st.slider("Макс. слоёв (max_layers)", 1, 20, 8, step=1)
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    with col2:
        top_n = st.slider("Нейронов в слое (top_n)", 2, 50, 10, step=1)

# --- Кнопка обучения ---
if st.button("🚀 Обучить модель МГУА", use_container_width=True):
    try:
        with st.spinner("⏳ Обучение МГУА-модели..."):
            time.sleep(0.5)

            num_feature_cols = [
                c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(num_feature_cols) < 2:
                st.error("❌ МГУА требует минимум 2 числовых признака.")
                st.stop()

            X = df[num_feature_cols].dropna()
            y = df.loc[X.index, target_col]
            X = X.values
            y = y.values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = GMDHRegressor(max_layers=max_layers, top_n=top_n, random_state=42)
            model.fit(X_train, y_train)

            metrics = model.get_metrics(X_test, y_test)
            y_pred = model.predict(X_test)

            layer_structure = model.get_structure()  # [{"best_score": float, ...}, ...]

            st.session_state["gmdh_state"] = {
                "model": model,
                "metrics": metrics,
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "layer_structure": layer_structure,
                "target_col": target_col,
                "feature_cols": num_feature_cols,
                "params": {
                    "max_layers": max_layers,
                    "top_n": top_n,
                    "test_size": test_size,
                },
            }

        st.success("✅ МГУА-модель обучена и сохранена")

    except Exception as e:
        st.error(t("❌ Не удалось обучить модель: ") + str(e))

# --- Если модель уже обучена ---
state = st.session_state.get("gmdh_state")
if state:
    has_model = state.get("model") is not None

    with st.expander("📊 Метрики (таблица)", expanded=False):
        metrics_df = pd.DataFrame(
            [(k.upper(), f"{v:.4f}") for k, v in state["metrics"].items()],
            columns=["Метрика", "Значение"]
        )
        st.table(metrics_df)

    with st.expander("📈 Predicted vs Actual и остатки", expanded=True):
        y_test_arr = np.asarray(state["y_test"])
        y_pred_arr = np.asarray(state["y_pred"])
        residuals = y_test_arr - y_pred_arr

        col_left, col_right = st.columns(2)

        with col_left:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test_arr, y=y_pred_arr,
                mode="markers",
                marker=dict(color="steelblue", opacity=0.6, size=6),
                name=t("Предсказание")
            ))
            mn, mx = float(y_test_arr.min()), float(y_test_arr.max())
            fig_pred.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color="crimson", dash="dash"),
                name=t("Идеал")
            ))
            fig_pred.update_layout(
                title=t("Predicted vs Actual"),
                xaxis_title=t("Actual"),
                yaxis_title=t("Predicted"),
                height=380
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        with col_right:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_pred_arr, y=residuals,
                mode="markers",
                marker=dict(color="darkorange", opacity=0.6, size=6),
                name=t("Остаток")
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="crimson")
            fig_res.update_layout(
                title=t("Остатки (Residuals)"),
                xaxis_title=t("Predicted"),
                yaxis_title=t("Residual"),
                height=380
            )
            st.plotly_chart(fig_res, use_container_width=True)

    layer_structure = state.get("layer_structure") or []
    if layer_structure:
        with st.expander("📉 MSE по слоям (layer-wise validation MSE)", expanded=True):
            layers_x = list(range(1, len(layer_structure) + 1))
            layers_y = [lyr["best_score"] for lyr in layer_structure]

            fig_layers = go.Figure()
            fig_layers.add_trace(go.Scatter(
                x=layers_x,
                y=layers_y,
                mode="lines+markers",
                line=dict(color="steelblue", width=2),
                marker=dict(size=8, color="steelblue"),
                name=t("MSE лучшего нейрона"),
            ))
            # Выделяем победивший (последний) слой
            fig_layers.add_trace(go.Scatter(
                x=[layers_x[-1]],
                y=[layers_y[-1]],
                mode="markers",
                marker=dict(size=14, color="crimson", symbol="star"),
                name=t("Победивший слой"),
            ))
            fig_layers.update_layout(
                title=t("MSE по слоям МГУА (валидация)"),
                xaxis_title=t("Номер слоя"),
                yaxis_title=t("MSE на валидации"),
                xaxis=dict(tickmode="linear", dtick=1),
                height=380,
            )
            st.plotly_chart(fig_layers, use_container_width=True)


    if st.button("🤖 Объяснить результаты в ИИ", key="gmdh_ai_explain", use_container_width=True):
        with st.spinner("⏳ Отправляем результаты..."):
            time.sleep(1.0)
            connect_ai_model_results(
                metrics=state["metrics"],
                model_type="МГУА (GMDH)",
                target_col=state["target_col"],
                top_features=None
            )
        st.success("✅ Результаты отправлены в ИИ. Перейдите в Чат с ИИ и спросите о метриках!")

    st.markdown("---")
    st.subheader("📥 Экспорт модели")
    if not has_model:
        st.info("Экспорт модели недоступен для восстановленного снимка без обученного объекта модели.")
        st.stop()

    json_str = state["model"].to_json()
    st.download_button(
        label="💾 Скачать модель (.json)",
        data=json_str,
        file_name="gmdh_model.json",
        mime="application/json",
        use_container_width=True,
    )
