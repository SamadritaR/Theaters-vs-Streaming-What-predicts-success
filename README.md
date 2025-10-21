# 🎬 Theaters vs. Streaming — What Predicts Success?

> “Movies are a machine that generates empathy.” — Roger Ebert

This project studies what actually drives movie success across **theatrical releases** and **streaming platforms**. I cleaned a 2018–2024 dataset, engineered features (log transforms, seasonality, genres/platforms), and trained **interpretable models**. Results are explained with **SHAP** and **PDP** so anyone can see *why* the models say what they say.

---

## 🔗 Quick Links (files in this repo)

- 📒 **Notebook:** [`MoviePlatformAnalysisCode.ipynb`](MoviePlatformAnalysisCode.ipynb)
- 🖼️ **Figures (PDF):** [`Figures_Final_Report.pdf`](Figures_Final_Report.pdf) · [`Figures.pdf`](Figures.pdf)
- 🧾 **Report:** [`Final Project Report.pdf`](Final%20Project%20Report.pdf)
- 🖥️ **Presentation:** [`Theaters vs. Streaming What Predicts Success.pptx`](Theaters%20vs.%20Streaming%20What%20Predicts%20Success.pptx)
- 🗂️ **Cleaned data:** [`movies_merged.csv`](movies_merged.csv)

---

## 🧭 Project Overview

**Two prediction tasks, two targets:**

1) **Theatrical (Regression)** → predict `log(revenue)`  
2) **Streaming (Classification)** → predict whether a title lands in the **top quartile** (success = 1)

**Why two setups?** The outcomes and business questions differ. Theatrical success is a *continuous* box office signal; streaming success here is a *ranked outcome* (top quartile vs. not), closer to a greenlight decision.

---

## 🧱 Features (what went into the models)

- **Core continuous:** `log_budget`, `log_popularity`, `log_votes`, `runtime`
- **Categorical dummies:** main genre one-hots (e.g., `main_genre_Action`), seasonal flags (e.g., `release_season_Summer`)
- **Streaming platforms:** `netflix`, `hulu`, `prime_video`, `disney_plus` (0/1)
- **Target (streaming):** `target_top_quartile` (0/1)

> Log transforms stabilize skew and make effects easier to model/interpret.

---

## 🧪 Models Used (what they are + why we used them)

### 1) Elastic Net (for interpretability + stability)
**What it is:** a linear model with **L1 + L2** regularization.  
**Objective:** minimize  
\[
\text{MSE}(\beta) + \alpha \left( \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2 \right)
\]  
where L1 (lasso) can **shrink to zero** (feature selection) and L2 (ridge) **stabilizes** coefficients under multicollinearity.

**Why here:**  
- We want a **clear, directional story** (e.g., “higher budget ↑ revenue”).  
- **Robust** to correlated inputs (budget/popularity).  
- Good **baseline** for both tasks.

**How to read it:** coefficients are **marginal effects** holding others fixed; signs show direction; magnitude shows strength (on the log scale for transformed vars).

---

### 2) Gradient Boosting (for nonlinearity + interactions)
**What it is:** an **ensemble** of shallow decision trees trained **sequentially**. Each new tree focuses on the prior model’s **residuals/gradients**, slowly reducing error.

**Why here:**  
- Captures **nonlinear** patterns (e.g., runtime plateaus) and **feature interactions** (genre × season).  
- Often **higher accuracy** than linear models while remaining explainable with SHAP.

**How we kept it honest:**  
- Small trees (depth), **learning rate**, **early stopping**, and validation splits.  
- Feature importance + **PDP/SHAP** to ensure we’re not overfitting spurious patterns.

**How to read it:** feature importances rank drivers; **PDP** shows shape of effect; **SHAP** shows per-feature contributions for each prediction.

---

### 3) SHAP (Shapley values) — *Why the model said that*
**What it is:** a game-theoretic way to attribute a model’s prediction to each feature for a given data point.  
**Why here:**  
- Gives **local explanations** (per movie) and **global** ones (summary/beeswarm).  
- **Additive and consistent**: contributions sum to the prediction difference from baseline.

**How to read it:**  
- **Beeswarm plot:** each dot = a movie; color = feature value; horizontal position = contribution (SHAP value).  
- If higher `log_budget` dots cluster on the **right**, higher budget **raises** predicted revenue/success.

---

### 4) PDP (Partial Dependence Plots) — *Shape of the effect*
**What it is:** shows the **marginal** effect of one feature on the prediction, averaging over others.  
**Why here:**  
- Visualizes **monotonic lifts** and **plateaus** (e.g., runtime).  
- Complements SHAP by showing the **function shape**.

**Caveats:**  
- Assumes feature independence when averaging; interactions can blur the curve.  
- Don’t over-interpret in **sparse regions** (extrapolation risk).

---

## 📊 Key Results

### Streaming (Classification, test year 2021)
Confusion matrix: **TN=200, FP=21, FN=49, TP=50**

| Metric | Value |
|---|---:|
| Accuracy | **78.1%** |
| Precision (Top-quartile) | **70.4%** |
| Recall (Top-quartile) | **50.5%** |
| F1-score | **58.8%** |

**Interpretation:** Model favors **precision** → good at flagging likely winners (fewer false positives), but it **misses some sleepers**. Adjusting the threshold could trade precision for recall based on business goals (e.g., broad A/B tests vs. high-confidence greenlights).

### Theatrical (Regression)
- **Directionally:** `log_budget` and `log_popularity` are the **dominant positive** drivers of `log(revenue)`.  
- **Runtime** shows **diminishing returns** beyond typical feature-length ranges.  
- Genres and seasonal flags have **smaller, directional** effects.

> See the PDFs for the exact figures: [`Figures_Final_Report.pdf`](Figures_Final_Report.pdf)

---

## 🔎 What the Figures Show (how to read them)

- **Feature Importance (Bars):** ranks drivers by contribution. Large bars for `log_budget`/`log_popularity` (theatrical) and `log_votes` (streaming) mean those features move the model the most.
- **PDP:**  
  - Theatrical: `log_budget` → **monotonic lift** in predicted revenue; `runtime` → **plateau**.  
  - Streaming: **threshold-like** gains in success probability with higher `log_votes`.
- **SHAP Beeswarm:**  
  - Theatrical: higher `log_budget`/`log_popularity` dots sit to the **right** (positive lift).  
  - Streaming: higher `log_votes` dots sit to the **right** → **engagement signal is king**.

---
# open the notebook
jupyter notebook MoviePlatformAnalysisCode.ipynb
