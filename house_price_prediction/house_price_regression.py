"""House-price prediction with simple and multiple linear regression.

The data is the California Housing dataset bundled by scikit-learn. Run this
module to download the dataset (on first use), train both models, and create
an assessment report in ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "MedHouseVal"
SIMPLE_FEATURE = "MedInc"


def make_pipeline(features: list[str]) -> Pipeline:
    """Create a reproducible preprocessing-and-regression pipeline."""
    preprocess = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", LinearRegression())])


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, name: str) -> tuple[dict[str, float | str], pd.Series]:
    """Evaluate a fitted model on held-out data."""
    predictions = model.predict(x_test)
    result: dict[str, float | str] = {
        "model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }
    return result, pd.Series(predictions, index=y_test.index, name="prediction")


def coefficient_lines(model: Pipeline, features: list[str]) -> list[str]:
    """Explain standardized coefficients and the intercept in report-ready text."""
    regressor = model.named_steps["regressor"]
    lines = []
    for feature, coefficient in zip(features, regressor.coef_):
        direction = "increases" if coefficient > 0 else "decreases"
        lines.append(
            f"- **{feature}**: a one-standard-deviation increase {direction} the predicted "
            f"house value by **{abs(coefficient):.3f}** $100,000 units, holding the other features constant."
        )
    lines.append(
        f"- **Intercept ({regressor.intercept_:.3f})**: predicted house value when every scaled "
        "feature is at its mean (zero after standardization)."
    )
    return lines


def write_report(metrics: pd.DataFrame, multiple_model: Pipeline, features: list[str], path: Path) -> None:
    """Write the required evaluation and interpretation in Markdown."""
    best = metrics.loc[metrics["R2"].idxmax(), "model"]
    table = metrics.copy()
    for column in ["MAE", "RMSE", "R2"]:
        table[column] = table[column].map("{:.4f}".format)
    content = [
        "# House Price Prediction: Results",
        "",
        "## Evaluation on the held-out test set",
        "",
        table.to_markdown(index=False),
        "",
        f"The model with the higher R² in this run is **{best}**. Lower MAE/RMSE indicate smaller prediction errors; higher R² indicates more variance in house values explained by the model.",
        "",
        "## Coefficient interpretation (multiple linear regression)",
        "",
        "Features are median-imputed and standardized before fitting. Coefficients therefore describe a one-standard-deviation change, and values are in $100,000 units.",
        *coefficient_lines(multiple_model, features),
        "",
        "## Intercept interpretation",
        "",
        "Because the predictor variables are standardized, the intercept is the model's prediction for a district whose features are all at their average values. It is a baseline, not a literal house with zero bedrooms or zero population.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    """Load data, train both required models, and save assessment artifacts."""
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.rename(columns={dataset.target.name: TARGET})
    x = frame.drop(columns=TARGET)
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    simple = make_pipeline([SIMPLE_FEATURE]).fit(x_train, y_train)
    multiple_features = x.columns.tolist()
    multiple = make_pipeline(multiple_features).fit(x_train, y_train)

    simple_metrics, _ = evaluate(simple, x_test, y_test, "Simple Linear Regression (MedInc)")
    multiple_metrics, predictions = evaluate(multiple, x_test, y_test, "Multiple Linear Regression (all features)")
    metrics = pd.DataFrame([simple_metrics, multiple_metrics])

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    metrics.to_csv(artifacts / "house_price_metrics.csv", index=False)
    write_report(metrics, multiple, multiple_features, artifacts / "house_price_report.md")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.35)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, "r--", label="Perfect prediction")
    plt.xlabel("Actual median house value ($100,000)")
    plt.ylabel("Predicted median house value ($100,000)")
    plt.title("Multiple Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print(metrics.to_string(index=False, float_format="{:.4f}".format))
    print("\nSaved results to artifacts/house_price_metrics.csv and artifacts/house_price_report.md")


if __name__ == "__main__":
    main()