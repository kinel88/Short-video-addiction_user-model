from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(page_title="KuaiRec Inspector", layout="wide")


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    elif suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


@st.cache_data
def load_full_table(path_str: str) -> pd.DataFrame:
    return read_table(path_str, nrows=None)


@st.cache_data
def load_preview_table(path_str: str, nrows: int) -> pd.DataFrame:
    return read_table(path_str, nrows=nrows)


def main() -> None:
    st.title("KuaiRec File Inspector")

    st.write("用这个页面查看大文件的列名、前几行、数据类型、缺失值和单列分布。")

    default_path = ""
    file_path = st.text_input("输入文件路径", value=default_path)

    preview_rows = st.slider("预览前多少行", min_value=20, max_value=500, value=100, step=20)

    if not file_path.strip():
        st.info("先输入你的 KuaiRec 文件路径。")
        return

    path = Path(file_path)
    if not path.exists():
        st.error(f"文件不存在: {path}")
        return

    try:
        preview_df = load_preview_table(str(path), preview_rows)
    except Exception as e:
        st.exception(e)
        return

    st.subheader("1. 基本信息")
    c1, c2, c3 = st.columns(3)
    c1.metric("预览行数", len(preview_df))
    c2.metric("列数", len(preview_df.columns))
    c3.metric("文件类型", path.suffix.lower())

    st.subheader("2. 所有列名")
    col_df = pd.DataFrame({
        "column_name": preview_df.columns,
        "dtype_in_preview": [str(preview_df[c].dtype) for c in preview_df.columns],
    })
    st.dataframe(col_df, use_container_width=True, height=400)

    st.subheader("3. 前几行数据")
    st.dataframe(preview_df, use_container_width=True, height=500)

    st.subheader("4. 缺失值情况（基于预览）")
    missing_df = pd.DataFrame({
        "column_name": preview_df.columns,
        "missing_count": [int(preview_df[c].isna().sum()) for c in preview_df.columns],
        "missing_ratio": [float(preview_df[c].isna().mean()) for c in preview_df.columns],
    }).sort_values("missing_count", ascending=False)
    st.dataframe(missing_df, use_container_width=True, height=400)

    st.subheader("5. 单列详细查看")
    selected_col = st.selectbox("选择一列", list(preview_df.columns))

    if selected_col:
        s = preview_df[selected_col]

        c1, c2 = st.columns(2)
        with c1:
            st.write("数据类型:", s.dtype)
            st.write("非空数量:", int(s.notna().sum()))
            st.write("空值数量:", int(s.isna().sum()))
            st.write("唯一值数量（预览中）:", int(s.nunique(dropna=True)))

        with c2:
            st.write("示例值（前 20 个非空）:")
            examples = s.dropna().astype(str).head(20).tolist()
            st.write(examples if examples else "无")

        st.write("Top 30 value counts（预览中）")
        vc = s.astype(str).value_counts(dropna=False).head(30).reset_index()
        vc.columns = ["value", "count"]
        st.dataframe(vc, use_container_width=True, height=400)

    st.subheader("6. 搜索列名")
    keyword = st.text_input("输入关键词，比如 user / video / tag / watch / duration")
    if keyword.strip():
        matched = [c for c in preview_df.columns if keyword.lower() in c.lower()]
        st.write("匹配到的列：", matched if matched else "没有")

    st.subheader("7. 加载完整数据（慎点，大文件会慢）")
    if st.button("读取完整文件"):
        try:
            full_df = load_full_table(str(path))
            st.success(f"完整读取成功：{full_df.shape[0]} 行 × {full_df.shape[1]} 列")

            info_df = pd.DataFrame({
                "column_name": full_df.columns,
                "dtype": [str(full_df[c].dtype) for c in full_df.columns],
                "missing_count": [int(full_df[c].isna().sum()) for c in full_df.columns],
                "missing_ratio": [float(full_df[c].isna().mean()) for c in full_df.columns],
                "nunique": [int(full_df[c].nunique(dropna=True)) for c in full_df.columns],
            })
            st.dataframe(info_df, use_container_width=True, height=500)

            st.write("完整数据前 100 行")
            st.dataframe(full_df.head(100), use_container_width=True, height=500)

        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()