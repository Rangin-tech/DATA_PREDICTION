import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Universal Data Analyzer", layout="wide")
st.title("🤖 Universal Data Analyzer & Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 🔹 Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # 🔹 Detect date columns
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except:
                pass

    # 🔹 Convert numeric-like columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except:
                pass

    # Show dataset preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique().tolist()
        if 1 < len(unique_vals) <= 50:  # skip huge cardinality
            selected_vals = st.sidebar.multiselect(
                f"Filter by {col.capitalize()}",
                unique_vals,
                default=unique_vals
            )
            df = df[df[col].isin(selected_vals)]

    # Numeric columns for analysis
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # 📈 Column statistics
    if numeric_cols:
        st.subheader("📈 Numeric Column Statistics")
        st.write(df[numeric_cols].describe())

    # 📊 Histogram / Distribution
    if numeric_cols:
        st.subheader("📊 Histogram / Distribution")
        selected_num = st.selectbox("Select column for histogram", numeric_cols)
        st.bar_chart(df[selected_num].dropna().value_counts().sort_index())
    else:
        # If no numeric columns, fallback to categorical frequencies
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            st.subheader("📊 Categorical Distribution")
            selected_cat = st.selectbox("Select column for frequency chart", cat_cols)
            st.bar_chart(df[selected_cat].value_counts())
        else:
            st.warning("⚠️ No numeric or categorical columns available for histogram.")

    # 🔎 Correlation matrix
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Between Numeric Columns")
        st.dataframe(df[numeric_cols].corr())

        # Scatter plot
        st.subheader("📌 Scatter Plot")
        x_axis = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
        y_axis = st.selectbox("Select Y-axis", [c for c in numeric_cols if c != x_axis], key="scatter_y")
        st.scatter_chart(df[[x_axis, y_axis]])

    # 🩺 Missing Values
    st.subheader("🩺 Missing Values Overview")
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        st.bar_chart(missing_vals[missing_vals > 0])
    else:
        st.success("✅ No missing values detected")

    # 🚨 Outlier Detection
    if numeric_cols:
        st.subheader("🚨 Outlier Detection (Z-score > 3)")
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        outliers = (z_scores > 3).any(axis=1)
        st.write(f"Found {outliers.sum()} potential outliers")
        if outliers.sum() > 0:
            st.dataframe(df[outliers].head())

    # 🔮 Prediction Section
    if len(numeric_cols) > 0:
        st.subheader("🔮 Predictive Modeling")
        target_col = st.selectbox("Select target column (to predict)", df.columns)

        if target_col in numeric_cols:
            # ---- Regression ----
            feature_cols = [col for col in numeric_cols if col != target_col]
            if feature_cols:
                X = df[feature_cols].dropna()
                y = df[target_col].dropna()
                X, y = X.align(y, join="inner", axis=0)

                if not X.empty and not y.empty:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.write(f"**Regression Performance for {target_col}:**")
                    st.write("R² Score:", round(r2_score(y_test, preds), 3))
                    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 3))

                    # User prediction input
                    st.subheader("📌 Make a Prediction")
                    user_input = {}
                    for col in feature_cols:
                        val = st.number_input(
                            f"Enter value for {col}",
                            float(df[col].min()) if not df[col].isna().all() else 0.0,
                            float(df[col].max()) if not df[col].isna().all() else 1.0,
                            float(df[col].mean()) if not df[col].isna().all() else 0.0
                        )
                        user_input[col] = val

                    if st.button("Predict", key="regression"):
                        input_df = pd.DataFrame([user_input])
                        prediction = model.predict(input_df)[0]
                        st.success(f"Predicted {target_col}: {round(prediction, 2)}")

        else:
            # ---- Classification ----
            st.info("Target column looks categorical → Running classification")

            X = df.dropna().drop(columns=[target_col])
            y = df.dropna()[target_col]

            # Encode categorical features
            X = pd.get_dummies(X, drop_first=True)

            if not X.empty and not y.empty:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                acc = accuracy_score(y_test, preds)
                st.write(f"**Classification Performance for {target_col}:**")
                st.write("Accuracy:", round(acc, 3))

                # Confusion Matrix
                st.subheader("📌 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # User classification input
                st.subheader("📌 Make a Prediction")
                user_input = {}
                for col in X.columns:
                    val = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
                    user_input[col] = val

                if st.button("Predict", key="classification"):
                    input_df = pd.DataFrame([user_input])
                    prediction = clf.predict(input_df)[0]
                    st.success(f"Predicted {target_col}: {prediction}")

    # 💾 Download processed dataset
    st.subheader("💾 Download Processed Data")
    st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), "processed_dataset.csv", "text/csv")

else:
    st.info("⬆️ Please upload a CSV or Excel file to begin.")
