# =============================================================================
# Medical Cost Personal Dataset — End-to-End Regression Implementation
# Dataset: https://www.kaggle.com/mirichoi0218/insurance
# Author: Codersarts (https://www.codersarts.com)
# Labs:   https://labs.codersarts.com
# Codersarts Blog: https://www.codersarts.com/post/medical-cost-personal-dataset-regression
# =============================================================================
#
# INSTALL DEPENDENCIES:
#   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
#
# USAGE:
#   1. Download insurance.csv from Kaggle link above
#   2. Place it in the same directory as this script
#   3. Run: python medical_cost_regression.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Skipping XGBoost model.")
    print("       Install with: pip install xgboost\n")


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data(filepath="insurance.csv"):
    """Load the insurance dataset."""
    try:
        df = pd.read_csv(filepath)
        print("=" * 60)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"Shape       : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns     : {list(df.columns)}")
        print()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            "insurance.csv not found.\n"
            "Download from: https://www.kaggle.com/mirichoi0218/insurance"
        )


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda(df):
    """Full EDA: summary stats, distributions, correlations."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Basic info ---
    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Values ---")
    print(df.isnull().sum().to_string())

    print("\n--- Statistical Summary ---")
    print(df.describe().round(2).to_string())

    print("\n--- Categorical Value Counts ---")
    for col in ["sex", "smoker", "region"]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())

    # --- Plots ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Medical Cost Dataset — EDA Overview", fontsize=16, fontweight="bold", y=1.01)

    # Numerical distributions
    for i, col in enumerate(["age", "bmi", "charges"]):
        ax = axes[0][i]
        ax.hist(df[col], bins=30, edgecolor="white", color="#378ADD")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Categorical counts
    cat_colors = ["#1D9E75", "#D85A30", "#7F77DD"]
    for i, col in enumerate(["sex", "smoker", "region"]):
        ax = axes[1][i]
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color=cat_colors[i], edgecolor="white")
        ax.set_title(f"Count of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Charges vs key features
    ax = axes[2][0]
    smoker_yes = df[df["smoker"] == "yes"]["charges"]
    smoker_no  = df[df["smoker"] == "no"]["charges"]
    ax.boxplot([smoker_no, smoker_yes], labels=["Non-Smoker", "Smoker"])
    ax.set_title("Charges by Smoker Status")
    ax.set_ylabel("Charges ($)")

    ax = axes[2][1]
    ax.scatter(df["age"], df["charges"], alpha=0.4, color="#185FA5", s=15)
    ax.set_title("Age vs Charges")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges ($)")

    ax = axes[2][2]
    ax.scatter(df["bmi"], df["charges"], alpha=0.4, color="#BA7517", s=15)
    ax.set_title("BMI vs Charges")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges ($)")

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[Saved] eda_overview.png")

    return df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def feature_engineering(df):
    """
    Encode categoricals, create new features, return X and y.
    
    New features added:
    - bmi_category   : Underweight / Normal / Overweight / Obese
    - age_group      : Young / Middle / Senior
    - smoker_obese   : interaction term (smoker AND obese)
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # --- Encode binary categoricals ---
    le = LabelEncoder()
    df["sex_enc"]    = le.fit_transform(df["sex"])        # female=0, male=1
    df["smoker_enc"] = le.fit_transform(df["smoker"])     # no=0, yes=1

    # --- One-hot encode region ---
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)

    # --- BMI category (clinical thresholds) ---
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )
    bmi_dummies = pd.get_dummies(df["bmi_category"], prefix="bmi", drop_first=True)
    df = pd.concat([df, bmi_dummies], axis=1)

    # --- Age group ---
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 50, 100],
        labels=["young", "middle", "senior"]
    )
    age_dummies = pd.get_dummies(df["age_group"], prefix="age", drop_first=True)
    df = pd.concat([df, age_dummies], axis=1)

    # --- Interaction: smoker × obese (high-risk group) ---
    df["smoker_obese"] = ((df["smoker"] == "yes") & (df["bmi"] >= 30)).astype(int)

    # --- Select final features ---
    feature_cols = [
        "age", "bmi", "children",
        "sex_enc", "smoker_enc",
        "smoker_obese",
        "region_northwest", "region_southeast", "region_southwest",
        "bmi_overweight", "bmi_obese",
        "age_middle", "age_senior"
    ]

    # Keep only columns that exist (handles dtype issues)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["charges"]

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Target       : charges")
    print(f"X shape      : {X.shape}")
    print(f"y range      : ${y.min():.2f} – ${y.max():.2f}")

    return X, y, feature_cols


# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict, and return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        "Model"  : name,
        "MAE"    : round(mae, 2),
        "RMSE"   : round(rmse, 2),
        "R²"     : round(r2, 4),
        "_model" : model,
        "_pred"  : y_pred,
    }


def train_models(X, y):
    """Train all regression models, compare, plot results."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

    # --- Define models ---
    models = {
        "Linear Regression"   : LinearRegression(),
        "Ridge Regression"    : Ridge(alpha=1.0),
        "Random Forest"       : RandomForestRegressor(
                                    n_estimators=200,
                                    max_depth=10,
                                    min_samples_split=5,
                                    random_state=42,
                                    n_jobs=-1
                                ),
        "Gradient Boosting"   : GradientBoostingRegressor(
                                    n_estimators=200,
                                    learning_rate=0.05,
                                    max_depth=4,
                                    random_state=42
                                ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )

    # --- Evaluate each model ---
    results = []
    for name, model in models.items():
        print(f"Training: {name}...")
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)
        print(f"  MAE: ${res['MAE']:,.2f}  |  RMSE: ${res['RMSE']:,.2f}  |  R²: {res['R²']}")

    # --- Results table ---
    print("\n--- Model Comparison ---")
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]).sort_values("R²", ascending=False)
    print(results_df.to_string(index=False))

    # --- Best model ---
    best = max(results, key=lambda r: r["R²"])
    print(f"\n[BEST MODEL] {best['Model']} — R²: {best['R²']}, RMSE: ${best['RMSE']:,.2f}")

    # --- Plot: actual vs predicted (best model) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Model Results — {best['Model']}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, best["_pred"], alpha=0.5, color="#185FA5", s=20)
    line_min = min(y_test.min(), best["_pred"].min())
    line_max = max(y_test.max(), best["_pred"].max())
    ax.plot([line_min, line_max], [line_min, line_max], "r--", lw=1.5)
    ax.set_xlabel("Actual Charges ($)")
    ax.set_ylabel("Predicted Charges ($)")
    ax.set_title("Actual vs Predicted")

    # --- Plot: residuals ---
    ax = axes[1]
    residuals = y_test.values - best["_pred"]
    ax.scatter(best["_pred"], residuals, alpha=0.4, color="#1D9E75", s=20)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Charges ($)")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")

    # --- Plot: model R² comparison bar ---
    ax = axes[2]
    model_names = results_df["Model"].tolist()
    r2_scores   = results_df["R²"].tolist()
    colors = ["#378ADD" if n != best["Model"] else "#1D9E75" for n in model_names]
    bars = ax.barh(model_names, r2_scores, color=colors, edgecolor="white")
    ax.set_xlabel("R² Score")
    ax.set_title("R² Score Comparison")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] model_results.png")

    return best["_model"], X_train, X_test, y_train, y_test, results_df


# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_cols):
    """Plot feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        print("\n[INFO] Feature importance not available for this model type.")
        return

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature"   : feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    print(fi_df.sort_values("Importance", ascending=False).to_string(index=False))

    plt.figure(figsize=(8, 6))
    colors = ["#378ADD" if v < fi_df["Importance"].max() else "#1D9E75"
              for v in fi_df["Importance"]]
    plt.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="white")
    plt.xlabel("Importance Score")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] feature_importance.png")


# =============================================================================
# 6. CROSS-VALIDATION
# =============================================================================

def cross_validate_best(model, X, y):
    """5-fold cross-validation on best model."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)

    cv_r2   = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mae  = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error"))

    print(f"R²   — Mean: {cv_r2.mean():.4f}  |  Std: {cv_r2.std():.4f}  |  Folds: {cv_r2.round(4)}")
    print(f"MAE  — Mean: ${cv_mae.mean():,.2f}  |  Std: ${cv_mae.std():,.2f}")
    print(f"RMSE — Mean: ${cv_rmse.mean():,.2f}  |  Std: ${cv_rmse.std():,.2f}")


# =============================================================================
# 7. PREDICT NEW SAMPLE
# =============================================================================

def predict_sample(model, feature_cols):
    """
    Predict charges for a new patient.
    Mirrors the same feature engineering used in training.
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)

    # Raw sample
    sample = {
        "age": 35, "sex": "male", "bmi": 28.5,
        "children": 2, "smoker": "no", "region": "southeast"
    }
    print(f"Input: {sample}")

    # Replicate feature engineering on single row
    row = {}
    row["age"]      = sample["age"]
    row["bmi"]      = sample["bmi"]
    row["children"] = sample["children"]
    row["sex_enc"]  = 1 if sample["sex"] == "male" else 0
    row["smoker_enc"] = 1 if sample["smoker"] == "yes" else 0

    is_obese   = sample["bmi"] >= 30
    is_smoker  = sample["smoker"] == "yes"
    row["smoker_obese"] = int(is_smoker and is_obese)

    # Region dummies (drop_first removes northwest as baseline)
    row["region_northwest"] = 1 if sample["region"] == "northwest" else 0
    row["region_southeast"] = 1 if sample["region"] == "southeast" else 0
    row["region_southwest"] = 1 if sample["region"] == "southwest" else 0

    # BMI category dummies (baseline = underweight)
    row["bmi_overweight"] = 1 if 25 <= sample["bmi"] < 30 else 0
    row["bmi_obese"]      = 1 if sample["bmi"] >= 30 else 0

    # Age group dummies (baseline = young)
    row["age_middle"] = 1 if 30 <= sample["age"] < 50 else 0
    row["age_senior"] = 1 if sample["age"] >= 50 else 0

    # Build input array in correct feature order
    input_data = pd.DataFrame([[row.get(f, 0) for f in feature_cols]], columns=feature_cols)
    prediction = model.predict(input_data)[0]

    print(f"\nPredicted Insurance Charges: ${prediction:,.2f}")
    return prediction


# =============================================================================
# 8. SUMMARY
# =============================================================================

def print_summary(results_df):
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    insights = [
        "Smoking status is the strongest predictor of medical charges.",
        "BMI >= 30 (obese) combined with smoking creates a very high-cost segment.",
        "Age has a positive linear relationship with charges.",
        "Region has a relatively small effect on charges.",
        "Tree-based models (Random Forest, XGBoost) outperform linear models",
        "  because the relationship between features and charges is non-linear."
    ]
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    print("\n" + "=" * 60)
    print("CODERSARTS SERVICES")
    print("=" * 60)
    print("  Need help with your ML assignment or project?")
    print()
    print("  Assignment Help  → https://www.codersarts.com/machine-learning-assignment-help")
    print("  Build a Product  → https://labs.codersarts.com")
    print("  1:1 Training     → https://training.codersarts.com")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # 1. Load
    df = load_data("insurance.csv")

    # 2. EDA
    run_eda(df)

    # 3. Feature engineering
    X, y, feature_cols = feature_engineering(df)

    # 4. Train & evaluate all models
    best_model, X_train, X_test, y_train, y_test, results_df = train_models(X, y)

    # 5. Feature importance
    plot_feature_importance(best_model, feature_cols)

    # 6. Cross-validation
    cross_validate_best(best_model, X, y)

    # 7. Sample prediction
    predict_sample(best_model, feature_cols)

    # 8. Summary
    print_summary(results_df)
