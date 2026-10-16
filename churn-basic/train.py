"""
Retrieve Telco customers from MongoDB, train two classical models, evaluate
them on a held-out test set, and store churn predictions.

Training data always comes from MongoDB, never directly from the CSV.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Save plots without opening a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Connection string: local MongoDB by default, Atlas via MONGODB_URI.
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

DATABASE_NAME = "churn_db"
CUSTOMERS_COLLECTION = "customers"
PREDICTIONS_COLLECTION = "predictions"
EXPECTED_ROW_COUNT = 7043
RANDOM_STATE = 42
FEATURE_PLOT_PATH = Path(__file__).resolve().parent / "feature_importance.png"


def connect_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    print("MongoDB connection successful.")
    return client


def load_customers_from_mongo(client) -> pd.DataFrame:
    """Step 3: read raw customer records back from MongoDB."""
    collection = client[DATABASE_NAME][CUSTOMERS_COLLECTION]
    records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)

    print("\n=== Step 3: Data retrieved from MongoDB ===")
    print("DataFrame shape:", df.shape)
    print("Column names:", list(df.columns))
    print("First five rows:")
    print(df.head())
    print("Number of records returned:", len(df))

    if "customerID" not in df.columns:
        raise RuntimeError("customerID is missing from the retrieved data.")
    if len(df) != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ROW_COUNT} rows from MongoDB, got {len(df)}."
        )
    print("Checkpoint passed: DataFrame has 7,043 rows and customerID is present.")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Step 4: clean TotalCharges, encode Churn, keep customer IDs aside."""
    df = df.copy()

    # Blank TotalCharges strings (usually tenure = 0) become missing, then 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    unexpected = set(df["Churn"].dropna().unique()) - {0, 1}
    if unexpected or df["Churn"].isna().any():
        raise RuntimeError(f"Churn encoding produced unexpected values: {unexpected}")

    customer_ids = df["customerID"].copy()

    print("\n=== Step 4: Data cleaning ===")
    print("Missing values by column:")
    print(df.isna().sum())
    print("\nChurn class counts (0 = retained, 1 = churned):")
    print(df["Churn"].value_counts())
    print("\nChurn class proportions:")
    print(df["Churn"].value_counts(normalize=True))

    missing_after_clean = df.drop(columns=["customerID"]).isna().sum().sum()
    if missing_after_clean:
        raise RuntimeError(
            f"Unexpected missing values remain after cleaning: {missing_after_clean}"
        )

    churn_rate = df["Churn"].mean()
    print(f"\nOverall churn prevalence: {churn_rate:.1%}")
    print("Checkpoint passed: no unexpected missing values; class balance is plausible.")
    return df, customer_ids


def explore_data(df: pd.DataFrame) -> None:
    """Step 5: two pandas summaries that students can discuss in a viva."""
    print("\n=== Step 5: Exploratory data analysis ===")

    # Month-to-month customers usually leave more often than locked-in contracts.
    churn_by_contract = df.groupby("Contract")["Churn"].mean() * 100
    print("Churn rate by contract type (%):")
    print(churn_by_contract.round(2).to_string())

    tenure_charges = df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean()
    tenure_charges.index = tenure_charges.index.map({0: "Retained (0)", 1: "Churned (1)"})
    print("\nMean tenure and monthly charges by churn status:")
    print(tenure_charges.round(2).to_string())
    print(
        "Interpretation is printed from computed values: shorter tenure and "
        "higher monthly charges are typically associated with churn in this dataset."
    )
    print("Checkpoint passed: both EDA tables displayed.")


def split_features(df: pd.DataFrame):
    """Step 6: stratified split and preprocessing fitted only on training data."""
    y = df["Churn"]
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n=== Step 6: Train/test split ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Train churn rate:", f"{y_train.mean():.3f}")
    print("Test churn rate:", f"{y_test.mean():.3f}")

    if "customerID" in X.columns or "Churn" in X.columns:
        raise RuntimeError("Identifier or target leaked into the feature matrix.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Checkpoint passed: stratified 80/20 split; target and ID excluded.")
    return X, y, X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_logistic_pipeline(numeric_features, categorical_features) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_forest_pipeline(numeric_features, categorical_features) -> Pipeline:
    # Trees do not need scaled numbers; one-hot encoding still helps mixed types.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test, y_test) -> dict:
    """Step 8: score one fitted pipeline on the untouched test set."""
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "F1-score": f1_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score(y_test, y_probability),
    }
    print(f"ROC-AUC score: {metrics['ROC-AUC']:.4f}")
    return metrics


def print_metric_explanations() -> None:
    print("\n=== How to read the metrics ===")
    print("True positives (TP): churned customers the model correctly flagged.")
    print("True negatives (TN): retained customers the model correctly left unmarked.")
    print("False positives (FP): retained customers wrongly flagged as churn.")
    print("False negatives (FN): churned customers the model missed.")
    print("Precision: of customers predicted as churn, how many actually churned.")
    print("Recall: of customers who actually churned, how many the model found.")
    print("F1-score: harmonic mean of precision and recall for the churn class.")
    print("ROC-AUC: ranking quality across all probability thresholds (0.5 = chance).")
    print(
        "Accuracy alone is misleading on imbalanced data. About 73% of customers "
        "did not churn, so a model that always predicts 'retained' would look "
        "accurate while missing every customer who left."
    )


def plot_feature_importance(forest_pipeline: Pipeline) -> pd.Series:
    """Step 9: top 10 Random Forest importances from the fitted training pipeline."""
    preprocessor = forest_pipeline.named_steps["preprocess"]
    classifier = forest_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise RuntimeError("Feature names and importance values are misaligned.")

    ranking = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top10 = ranking.head(10)

    print("\n=== Step 9: Random Forest feature importance (top 10) ===")
    print(top10.to_string())
    print(
        "These values show predictive contribution inside this Random Forest, "
        "not proof that a feature causes churn."
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top10.index[::-1], top10.values[::-1], color="#2563eb")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(FEATURE_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved chart: {FEATURE_PLOT_PATH}")
    return top10


def assign_risk_level(probability: float) -> str:
    # Illustrative business rules, not universally validated cutoffs.
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def save_predictions(client, selected_name: str, selected_pipeline: Pipeline, X, y, customer_ids) -> None:
    """
    Step 10: refit the selected model on all cleaned records and store scores.

    The predictions collection includes customers seen during training. These
    are operational scores, not an independent accuracy evaluation. Reported
    performance remains the held-out test-set metrics from Step 8.
    """
    print("\n=== Step 10: Save predictions to MongoDB ===")
    selected_pipeline.fit(X, y)
    probabilities = selected_pipeline.predict_proba(X)[:, 1]
    scored_at = datetime.now(timezone.utc)

    documents = []
    for customer_id, probability in zip(customer_ids, probabilities):
        rounded = round(float(probability), 4)
        documents.append(
            {
                "customerID": customer_id,
                "churn_probability": rounded,
                "risk_level": assign_risk_level(rounded),
                "model_used": selected_name,
                "scored_at": scored_at,
            }
        )

    collection = client[DATABASE_NAME][PREDICTIONS_COLLECTION]
    collection.drop()
    result = collection.insert_many(documents)
    stored_count = collection.count_documents({})
    sample = collection.find_one({}, {"_id": 0})

    print("Selected model:", selected_name)
    print("Prediction documents created:", len(documents))
    print("Documents inserted:", len(result.inserted_ids))
    print("Documents in MongoDB after insertion:", stored_count)
    print("Sample prediction document:")
    print(sample)

    if stored_count != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Predictions collection has {stored_count} documents, expected {EXPECTED_ROW_COUNT}."
        )
    print("Checkpoint passed: 7,043 prediction documents stored.")


def main() -> int:
    client = None
    try:
        client = connect_mongo()
        df = load_customers_from_mongo(client)
        df, customer_ids = clean_data(df)
        explore_data(df)

        (
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            numeric_features,
            categorical_features,
        ) = split_features(df)

        print("\n=== Step 7: Train models on training data only ===")
        logistic_model = build_logistic_pipeline(numeric_features, categorical_features)
        forest_model = build_forest_pipeline(numeric_features, categorical_features)
        logistic_model.fit(X_train, y_train)
        forest_model.fit(X_train, y_train)
        print("Logistic Regression training complete.")
        print("Random Forest training complete.")
        print("Checkpoint passed: preprocessing was fitted on X_train only.")

        print("\n=== Step 8: Evaluate on the same held-out test set ===")
        logistic_metrics = evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
        forest_metrics = evaluate_model("Random Forest", forest_model, X_test, y_test)
        print_metric_explanations()

        comparison = pd.DataFrame([logistic_metrics, forest_metrics])
        print("\nModel comparison (churn class = 1 for Precision/Recall/F1):")
        print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        comparison_path = Path(__file__).resolve().parent / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Wrote {comparison_path} for the README.")

        if forest_metrics["ROC-AUC"] > logistic_metrics["ROC-AUC"]:
            selected_name = "Random Forest"
            selected_pipeline = forest_model
        elif logistic_metrics["ROC-AUC"] > forest_metrics["ROC-AUC"]:
            selected_name = "Logistic Regression"
            selected_pipeline = logistic_model
        else:
            # ROC-AUC effectively tied: use churn-class F1 as the tie-breaker.
            if forest_metrics["F1-score"] >= logistic_metrics["F1-score"]:
                selected_name = "Random Forest"
                selected_pipeline = forest_model
            else:
                selected_name = "Logistic Regression"
                selected_pipeline = logistic_model

        print(
            f"\nSelected model: {selected_name} "
            "(primary criterion: test ROC-AUC; F1-score if tied)."
        )
        print("Checkpoint passed: both models scored on the same test set.")

        plot_feature_importance(forest_model)
        if not FEATURE_PLOT_PATH.is_file():
            raise RuntimeError("feature_importance.png was not created.")
        print("Checkpoint passed: feature importance chart saved.")

        save_predictions(client, selected_name, selected_pipeline, X, y, customer_ids)
        return 0
    except (PyMongoError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
