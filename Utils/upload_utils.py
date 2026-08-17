import pandas as pd
import streamlit as st
import re

def looks_like_number(s: str) -> bool:
    s = s.strip().replace(",", ".")
    return bool(re.match(r"^-?\d+(\.\d+)?$", s))

def load_data(uploaded_file) -> pd.DataFrame:
    st.session_state["original_filename"] = uploaded_file.name  
    fname = uploaded_file.name.lower()
    if fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif fname.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Неподдерживаемый формат файла")
        raise ValueError

    conversion_log = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object":
            stripped = df[col].astype("string").str.strip()
            normalized = stripped.str.replace(",", ".", regex=False)
            mask = normalized.fillna("").apply(looks_like_number)
            rate = mask.mean()
            if rate > 0.9:
                try:
                    df[col] = pd.to_numeric(normalized, errors="raise")
                    conversion_log.append(f"{col}: object → float ({rate:.0%})")
                except Exception:
                    df[col] = stripped
                    conversion_log.append(f"{col}: оставлен как текст")
            else:
                df[col] = stripped
                conversion_log.append(f"{col}: текст ({rate:.0%} чисел)")
        else:
            conversion_log.append(f"{col}: {dtype}")

    st.session_state["conversion_log"] = conversion_log
    return df

def get_base_info(df: pd.DataFrame) -> dict:
    return {
        "Строк": df.shape[0],
        "Столбцов": df.shape[1],
        "Пропусков": int(df.isnull().sum().sum()),
        "Дубликатов": int(df.duplicated().sum()),
        "Числовых": len(df.select_dtypes("number").columns),
        "Категориальных": len(df.select_dtypes("object").columns),
    }

def show_data_head(df: pd.DataFrame, n: int = 5):
    st.markdown(f"### 🧾 Пример данных (первые {n} строк):")
    st.dataframe(df.head(n), use_container_width=True)

def show_descriptive_stats(df: pd.DataFrame):
    st.markdown("### 📑 Описательная статистика (describe)")
    desc = df.describe(include="all").round(3).transpose()
    desc.index.name = "Признак"
    st.dataframe(desc, use_container_width=True)
    st.markdown(
        "Некоторые ячейки могут быть пустыми (None) —\n"
        "- для числовых признаков поля `unique`, `top` и `freq` не вычисляются;\n"
        "- для категориальных полей отсутствуют `min`, `25%`, `50%`, `75%`, `max`, так как они неприменимы.\n"
        "Это нормально."
    )

def display_base_info(base_info: dict):
    st.subheader("📊 Базовая информация")
    cols = st.columns(len(base_info))
    for col, (label, value) in zip(cols, base_info.items()):
        col.metric(label=label, value=value)
