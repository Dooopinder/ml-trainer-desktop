import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(task: str, model_name: str):
    if task == "Classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=2000)
        if model_name == "Random Forest Classifier":
            return RandomForestClassifier(n_estimators=300, random_state=42)
        raise ValueError("Unknown classification model")

    if task == "Regression":
        if model_name == "Ridge Regression":
            return Ridge()
        if model_name == "Random Forest Regressor":
            return RandomForestRegressor(n_estimators=300, random_state=42)
        raise ValueError("Unknown regression model")

    if task == "Clustering":
        if model_name == "KMeans":
            return KMeans(n_clusters=3, random_state=42, n_init="auto")
        raise ValueError("Unknown clustering model")

    if task == "Anomaly Detection":
        if model_name == "Isolation Forest":
            return IsolationForest(random_state=42)
        raise ValueError("Unknown anomaly model")

    if task == "Dimensionality Reduction":
        if model_name == "PCA":
            return PCA(n_components=2)
        raise ValueError("Unknown reduction model")

    raise ValueError("Unknown task")


def run_task(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    target: str | None,
    test_size: float = 0.2,
    random_state: int = 42,
):

    report: dict = {"task": task, "model": model_name}

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if task in ("Classification", "Regression"):
        if not target:
            raise ValueError("Target column is required.")
        if target not in df.columns:
            raise ValueError("Target column not found in dataset.")

        df2 = df.dropna(subset=[target]).copy()
        y = df2[target]
        X = df2.drop(columns=[target])

        pre = make_preprocessor(X)
        est = build_model(task, model_name)
        pipe = Pipeline([("preprocess", pre), ("model", est)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,test_size=test_size,random_state=random_state
        )
        del X  # free original full dataset


        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if task == "Classification":
            acc = float(accuracy_score(y_test, y_pred))
            f1w = float(f1_score(y_test, y_pred, average="weighted"))
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
            report.update({
                "accuracy": acc,
                "f1_weighted": f1w,
                "confusion_matrix": cm.tolist(),
                "class_labels": labels,
            })
            preds = pd.DataFrame({"y_true": y_test.astype(str), "y_pred": pd.Series(y_pred).astype(str)})
            plot_payload = {"type": "confusion_matrix", "cm": cm, "labels": labels}
            return pipe, preds, report, plot_payload

        # Regression
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        report.update({"rmse": rmse, "r2": r2})
        preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        plot_payload = {"type": "pred_vs_actual", "y_true": np.asarray(y_test), "y_pred": np.asarray(y_pred)}
        return pipe, preds, report, plot_payload

    # Unsupervised tasks
    X = df.copy()
    pre = make_preprocessor(X)

    if task == "Clustering":
        model = build_model(task, model_name)
        Xp = pre.fit_transform(X)
        labels = model.fit_predict(Xp)
        out = df.copy()
        out["cluster"] = labels
        report.update({
            "k": int(getattr(model, "n_clusters", 0)),
            "cluster_counts": out["cluster"].value_counts().to_dict(),
        })

        # 2D viz via PCA on processed features
        pca2 = PCA(n_components=2, random_state=42)
        Z = pca2.fit_transform(Xp)
        plot_payload = {"type": "scatter2d", "x": Z[:, 0], "y": Z[:, 1], "c": labels, "title": "Clusters (PCA view)"}
        return Pipeline([("preprocess", pre)]), out, report, plot_payload

    if task == "Anomaly Detection":
        model = build_model(task, model_name)
        Xp = pre.fit_transform(X)
        pred = model.fit_predict(Xp)  # -1 anomaly, 1 normal
        scores = model.decision_function(Xp)  # higher = more normal
        out = df.copy()
        out["is_anomaly"] = (pred == -1)
        out["anomaly_score"] = scores
        report.update({"anomalies": int(out["is_anomaly"].sum())})
        plot_payload = {"type": "hist", "values": scores, "title": "Anomaly scores (higher = more normal)"}
        return Pipeline([("preprocess", pre)]), out, report, plot_payload

    if task == "Dimensionality Reduction":
        pca = build_model(task, model_name)
        Xp = pre.fit_transform(X)
        Z = pca.fit_transform(Xp)
        out = df.copy()
        out["pc1"] = Z[:, 0]
        out["pc2"] = Z[:, 1]
        evr = getattr(pca, "explained_variance_ratio_", None)
        report.update({
            "components": 2,
            "explained_variance_ratio": evr.tolist() if evr is not None else None,
        })
        plot_payload = {"type": "scatter2d", "x": Z[:, 0], "y": Z[:, 1], "c": None, "title": "PCA (PC1 vs PC2)"}
        return Pipeline([("preprocess", pre), ("pca", pca)]), out, report, plot_payload

    raise ValueError("Unknown task")


def save_report_json(path: str, report: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_model_joblib(path: str, model_obj):
    joblib.dump(model_obj, path)
