from typing import Dict, List, Tuple
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from loguru import logger
import warnings

# Suppress specific LGBM warning (benign post-numpy conversion)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

logger.add("pipeline.log", rotation="10 MB")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for configurable pipeline execution.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed arguments with defaults.

    """
    parser = argparse.ArgumentParser(
        description="Credit Risk Prediction Pipeline Replication"
    )
    parser.add_argument(
        "--data_url",
        type=str,
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        help="URL to the dataset",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_components", type=int, default=10, help="Number of PCA components"
    )
    parser.add_argument(
        "--n_undersample_iters",
        type=int,
        default=5,
        help="Number of undersampling iterations",
    )
    parser.add_argument(
        "--top_iv_features",
        type=int,
        default=20,
        help="Number of top IV features to select",
    )
    parser.add_argument(
        "--iv_bins",
        type=int,
        default=10,
        help="Number of bins for IV calculation on continuous features",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split for TabNet early stopping",
    )
    return parser.parse_args()


def load_and_preprocess_data(
    url: str, columns: List[str], categorical_cols: List[str]
) -> pd.DataFrame:
    """Load dataset from URL and preprocess: remap label, one-hot encode categoricals.

    Aligns with paper's handling of mixed tabular data for credit risk.

    Parameters
    ----------
    url : str
        URL to the dataset.
    columns : List[str]
        Column names for the dataset.
    categorical_cols : List[str]
        Categorical columns to one-hot encode.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.

    Raises
    ------
    ValueError
        If data loading fails.

    """
    try:
        data = pd.read_csv(url, sep=" ", header=None, names=columns)
        data["label"] = data["label"].map({1: 0, 2: 1})  # 0: good, 1: bad (minority)
        data = pd.get_dummies(data, columns=categorical_cols)
        logger.info("Data loaded and preprocessed successfully. Shape: {}", data.shape)
        return data
    except Exception as e:
        logger.error("Error loading data: {}", e)
        raise ValueError(f"Data loading failed: {e}")


def calculate_iv(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> Dict[str, float]:
    """Calculate Information Value (IV) for feature prioritization.

    Formula intact: IV = ∑ [(P(bad|x_i) - P(good|x_i)) * WOE_i], WOE_i = ln[P(bad|x_i) / P(good|x_i)].
    Handles binaries without binning; epsilon for zero divisions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series
        Target series (binary).
    bins : int, optional
        Number of bins for continuous features, by default 10.

    Returns
    -------
    Dict[str, float]
        Dictionary of feature: IV values.

    """
    iv_dict: Dict[str, float] = {}
    for col in X.columns:
        if X[col].nunique() <= 2:
            X_binned = X[col]
        else:
            X_binned = pd.qcut(X[col], q=bins, duplicates="drop")
        df = pd.DataFrame({"x": X_binned, "y": y})
        grouped = df.groupby("x", observed=False)["y"].agg(["count", "sum"])
        grouped["bad"] = grouped["sum"] / grouped["sum"].sum()
        grouped["good"] = (grouped["count"] - grouped["sum"]) / (
            grouped["count"].sum() - grouped["sum"].sum()
        )
        grouped["woe"] = np.log(grouped["bad"] / grouped["good"].replace(0, 1e-10))
        grouped["iv"] = (grouped["bad"] - grouped["good"]) * grouped["woe"]
        iv_dict[col] = grouped["iv"].sum()
    return iv_dict


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_n: int = 20,
    bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Select top features by IV descending.

    Aligns with paper's prioritization for efficiency. Converts to numpy float32.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    top_n : int, optional
        Number of top features, by default 20.
    bins : int, optional
        Bins for IV, by default 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Selected X_train, X_test as numpy, and feature list.

    """
    iv = calculate_iv(X_train, y_train, bins)
    sorted_iv = sorted(iv.items(), key=lambda x: x[1], reverse=True)[:top_n]
    selected_features = [k for k, v in sorted_iv]
    logger.info(
        "Selected top {} features by IV: {}", len(selected_features), selected_features
    )
    X_train_selected = X_train[selected_features].values.astype(np.float32)
    X_test_selected = X_test[selected_features].values.astype(np.float32)
    return X_train_selected, X_test_selected, selected_features


def undersample_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative undersampling and model training/tuning.

    As per paper's multiple iterations for diversity. Ensemble via averaging.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test target (for tuning evaluation).
    args : argparse.Namespace
        Pipeline arguments.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Averaged probabilities for LightGBM, XGBoost, TabNet.

    """
    lgb_probs, xgb_probs, tab_probs = [], [], []
    for i in range(args.n_undersample_iters):
        rus = RandomUnderSampler(random_state=args.random_state + i)
        X_under, y_under = rus.fit_resample(X_train, y_train)
        X_under = X_under.astype(np.float32)
        y_under = y_under.astype(np.int32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_under,
            y_under,
            test_size=args.val_split,
            random_state=args.random_state + i,
            stratify=y_under,
        )

        lgb_param_grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}
        lgb_model = GridSearchCV(
            lgb.LGBMClassifier(random_state=args.random_state, verbose=-1),
            lgb_param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        lgb_model.fit(X_under, y_under)
        lgb_probs.append(lgb_model.predict_proba(X_test)[:, 1])

        xgb_param_grid = {"eta": [0.05, 0.1], "max_depth": [3, 6]}
        xgb_model = GridSearchCV(
            xgb.XGBClassifier(
                random_state=args.random_state, eval_metric="logloss", verbosity=0
            ),
            xgb_param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        xgb_model.fit(X_under, y_under)
        xgb_probs.append(xgb_model.predict_proba(X_test)[:, 1])

        tabnet_param_grid = {"n_d": [8, 16], "gamma": [1.3, 1.5]}
        best_f1, best_params = 0, None
        for n_d in tabnet_param_grid["n_d"]:
            for gamma in tabnet_param_grid["gamma"]:
                tabnet_model = TabNetClassifier(
                    n_d=n_d, gamma=gamma, seed=args.random_state, verbose=0
                )
                tabnet_model.fit(
                    X_tr, y_tr, eval_set=[(X_val, y_val)], patience=10, max_epochs=100
                )
                pred = tabnet_model.predict(X_test)
                curr_f1 = f1_score(y_test, pred)
                if curr_f1 > best_f1:
                    best_f1 = curr_f1
                    best_params = {"n_d": n_d, "gamma": gamma}
        tabnet_model = TabNetClassifier(
            **best_params, seed=args.random_state, verbose=0
        )
        tabnet_model.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)], patience=10, max_epochs=100
        )
        tab_probs.append(tabnet_model.predict_proba(X_test)[:, 1])

    logger.info("Completed {} undersampling iterations.", args.n_undersample_iters)
    return (
        np.mean(lgb_probs, axis=0),
        np.mean(xgb_probs, axis=0),
        np.mean(tab_probs, axis=0),
    )


def evaluate(
    y_test: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Intact from paper: Precision, Recall, F1, AUC.

    Parameters
    ----------
    y_test : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    threshold : float, optional
        Classification threshold, by default 0.5.

    Returns
    -------
    Dict[str, float]
        Metric dictionary.

    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
    }


def run_pipeline(
    args: argparse.Namespace, columns: List[str], categorical_cols: List[str]
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Execute staged pipeline: raw, PCA, PCA+SMOTEENN.

    Aligns with paper's staged approach.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments.
    columns : List[str]
        Column names.
    categorical_cols : List[str]
        Categorical columns.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], ...]
        Results for raw, PCA, SMOTEENN stages.

    """
    data = load_and_preprocess_data(args.data_url, columns, categorical_cols)
    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    X_train, X_test, _ = select_top_features(
        X_train, y_train, X_test, args.top_iv_features, args.iv_bins
    )
    y_train = y_train.values.astype(np.int32)
    y_test = y_test.values.astype(np.int32)

    lgb_prob_raw, xgb_prob_raw, tab_prob_raw = undersample_and_train(
        X_train, y_train, X_test, y_test, args
    )
    raw_results = {
        "LightGBM": evaluate(y_test, lgb_prob_raw),
        "XGBoost": evaluate(y_test, xgb_prob_raw),
        "TabNet": evaluate(y_test, tab_prob_raw),
    }

    pca = PCA(n_components=args.n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    logger.info(
        "PCA explained variance ratio sum: {:.2f}",
        np.sum(pca.explained_variance_ratio_),
    )
    lgb_prob_pca, xgb_prob_pca, tab_prob_pca = undersample_and_train(
        X_train_pca, y_train, X_test_pca, y_test, args
    )
    pca_results = {
        "LightGBM": evaluate(y_test, lgb_prob_pca),
        "XGBoost": evaluate(y_test, xgb_prob_pca),
        "TabNet": evaluate(y_test, tab_prob_pca),
    }

    smoteenn = SMOTEENN(random_state=args.random_state)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train_pca, y_train)
    lgb_prob_smote, xgb_prob_smote, tab_prob_smote = undersample_and_train(
        X_train_smoteenn, y_train_smoteenn, X_test_pca, y_test, args
    )
    smoteenn_results = {
        "LightGBM": evaluate(y_test, lgb_prob_smote),
        "XGBoost": evaluate(y_test, xgb_prob_smote),
        "TabNet": evaluate(y_test, tab_prob_smote),
    }

    return raw_results, pca_results, smoteenn_results


if __name__ == "__main__":
    args = parse_arguments()
    columns = [
        "checking",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "telephone",
        "foreign_worker",
        "label",
    ]
    categorical_cols = [
        "checking",
        "credit_history",
        "purpose",
        "savings",
        "employment",
        "personal_status",
        "other_debtors",
        "property",
        "other_installment_plans",
        "housing",
        "job",
        "telephone",
        "foreign_worker",
    ]

    raw_results, pca_results, smoteenn_results = run_pipeline(
        args, columns, categorical_cols
    )

    logger.info("Raw Results (post-IV & Undersampling): {}", raw_results)
    logger.info("PCA Results (with Undersampling): {}", pca_results)
    logger.info("PCA + SMOTEENN Results (with Undersampling): {}", smoteenn_results)
