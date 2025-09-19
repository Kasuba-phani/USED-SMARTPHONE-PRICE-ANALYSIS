![Project banner](Generated Image September 19, 2025 - 2_08PM.png)

Used Smartphone Price Analytics

Fair prices • Smarter decisions • Stronger trust.

This project analyzes the resale value of used smartphones and builds predictive (regression) and classification models using R. It reflects the exact approach in my Week-4 report—manual scripts (no automated pipeline) with clear documentation of preprocessing, model choices, and results.

🔎 What’s inside

EDA & Missingness (with visuals in the report)

Data preprocessing aligned to business context

Regression: Linear, Stepwise, Random Forest

Classification: Logistic Regression, Decision Tree, Random Forest (median-cut label)

Figures and metrics saved as outputs and included in the report/slides
.
Data & preprocessing (as done in the report)

Dataset: data/used_device_data.csv

Missing data: ~up to ~5% in a few features (e.g., camera MP, RAM, weight).
Assumed MCAR → used listwise deletion to avoid imputation bias and keep interpretation clean.

Zeros audit: Validated potential zero placeholders (e.g., front camera) vs. true device specs.

Normalization: normalized_used_price and normalized_new_price were already normalized—kept as provided.

Feature handling: Converted categorical fields (brand, OS, 4G/5G) to factors for modeling.
Problem framing

Regression: Predict normalized_used_price.

Classification: Create price_high label using the median of normalized_used_price
(price_high = 1 if ≥ median; else 0).

🧪 Train / validation / test

Hold-out split with validation (no cross-validation): 60/20/20 (train/val/test) for classification;
regression splits followed the same manual split logic used in the scripts/report.

Random seed set for reproducibility: set.seed(42) where applicable.

🛠 Models (as in the report)

Regression

Linear Regression

Stepwise Regression (for interpretability/feature parsimony)

Random Forest Regression (to capture non-linearities & interactions)

Classification

Logistic Regression (explainable baseline)

Decision Tree

Random Forest

✅ Results (reported)

Best regression: Random Forest with R² ≈ 0.860 and RMSE ≈ 0.226 (test).

Classification: Overall accuracy reported around ≈ 87.9% (see report/screens for per-model details).

Rationale: Random Forest captured non-linear/interaction effects better than linear/stepwise in this data, while linear/stepwise remained useful for explainability and verifying directional effects.

Full metrics, confusion matrices, and plots appear in docs/ (report/slides) and in outputs//figs/ from manual runs.
