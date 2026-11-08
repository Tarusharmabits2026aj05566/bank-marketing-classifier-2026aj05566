import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")
st.write("Compare five machine learning models on the Bank Marketing dataset.")

MODEL_DIR = "model"

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.joblib")),
        "KNN": joblib.load(os.path.join(MODEL_DIR, "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.joblib")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm.joblib")),
    }
    return preprocessor, models

preprocessor, models = load_artifacts()

uploaded_file = st.file_uploader(
    "Upload test CSV (must include input features and target column y)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "y" not in df.columns:
            st.error("CSV must contain the target column 'y'.")
            st.stop()

        expected_features = list(preprocessor.feature_names_in_)
        missing = [col for col in expected_features if col not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        X = df[expected_features]
        y = df["y"]

        # Accept either original labels or encoded target values.
        if y.dtype == "object":
            y = y.map({"no": 0, "yes": 1})

        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            st.error("Target column y must contain only 'no'/'yes' or 0/1.")
            st.stop()

        y = y.astype(int)
        X_processed = preprocessor.transform(X)

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            list(models.keys())
        )

        model = models[selected_model]
        predictions = model.predict(X_processed)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_processed)[:, 1]
        else:
            scores = model.decision_function(X_processed)

        st.subheader(f"Evaluation Results: {selected_model}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y, predictions):.4f}")
        col2.metric("Precision", f"{precision_score(y, predictions, zero_division=0):.4f}")
        col3.metric("Recall", f"{recall_score(y, predictions, zero_division=0):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1_score(y, predictions, zero_division=0):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y, predictions):.4f}")

        try:
            auc = roc_auc_score(y, scores)
            col6.metric("AUC", f"{auc:.4f}")
        except ValueError:
            col6.metric("AUC", "Not available")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )
        )

        st.subheader("Classification Report")
        st.text(classification_report(
            y, predictions,
            target_names=["No", "Yes"],
            zero_division=0
        ))

    except Exception as e:
        st.error(f"Could not process the uploaded CSV: {e}")

else:
    st.info("Upload the provided test_data.csv file to evaluate a model.")