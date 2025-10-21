# 🎬 Theaters vs. Streaming — What Predicts Success?

> “Movies are a machine that generates empathy.” — Roger Ebert

In that spirit, this project blends **art and analytics** to understand what actually drives movie success across **theatrical releases** and **streaming platforms**. I cleaned and engineered features from modern movie datasets (2018–2024), then built **interpretable** models that separate signal from noise—so we can talk about outcomes without hand-wavy buzzwords.

---

## 📦 What’s Here

- 📒 **Notebook** → `notebooks/MoviePlatformAnalysisCode.ipynb`
- 🖼️ **Figures (PDF)** → `figures/Figures_Final_Report.pdf` (final) and `figures/Figures.pdf` (earlier)
- 🧾 **Reports** → `reports/Final Project Report.pdf`, `reports/Preliminary_Report_Samadrita_Roy.pdf`
- 🖥️ **Presentation** → `presentation/Theaters vs. Streaming What Predicts Success.pptx`
- 🗂️ **Data** → `movies_merged/`

---

## 🧰 Libraries Used

- **Python**: pandas, numpy, scikit-learn, shap, matplotlib
- (Optional) seaborn, jupyter, statsmodels
- Utility: pathlib, json, itertools

---

## 🗃️ Data & Feature Engineering

- **Scope:** 2018–2024 movies; consistent IDs/titles; unified date formats
- **Cleaning:** de-duplication, outlier checks, missingness treatment
- **Transforms:** `log_budget`, `log_popularity`, `log_votes` to stabilize skew
- **Signals:** `runtime`, main-genre one-hots, seasonal flags
- **Platforms:** provider dummies (e.g., netflix/hulu/prime/disney) for streaming
- **Targets:**
  - **Theatrical** → regression on `log(revenue)`
  - **Streaming** → classification of **top-quartile** success

---

## 🧠 Model Details

### Theatrical (Regression)
- **Goal:** Predict `log(revenue)` using interpretable features
- **Why this approach:** Stable coefficients & SHAP for direction/magnitude
- **What to look at:** Feature importance + partial dependence

![Theatrical — Feature Importance](figures/theatrical_feature_importance.png)
![Theatrical — Partial Dependence](figures/theatrical_pdp.png)
![Theatrical — SHAP Beeswarm](figures/theatrical_shap_beeswarm.png)

> If you don’t see images, export key plots as PNG with the filenames above (they’re also in the PDFs).

### Streaming (Classification)
- **Goal:** Predict whether a title lands in the **top quartile**
- **Setup:** Year-based split (e.g., test=2021), threshold chosen for **precision**
- **What to look at:** Confusion matrix + feature importance + SHAP

![Streaming — Confusion Matrix](figures/streaming_confusion_matrix.png)
![Streaming — Feature Importance](figures/streaming_feature_importance.png)
![Streaming — SHAP Beeswarm](figures/streaming_shap_beeswarm.png)

---

## 📊 Evaluation Snapshot (Streaming — test year 2021)

**Confusion matrix:** TN=200, FP=21, FN=49, TP=50

| Metric | Value |
|---|---:|
| Accuracy | **78.1%** |
| Precision (Top-quartile) | **70.4%** |
| Recall (Top-quartile) | **50.5%** |
| F1-score | **58.8%** |

**Read:** The operating point favors **precision** → fewer false alarms when calling a “hit,” at the cost of missing some sleepers. Tune threshold ↑ if you want to catch more potential winners.

<details>
<summary><strong>Confusion Matrix Table</strong></summary>

|               | Pred: Not Top | Pred: Top |
|---|---:|---:|
| **Actual: Not Top** | 200 | 21 |
| **Actual: Top**     | 49  | 50 |

</details>

---

## 🔎 Model Analysis (What the figures show)

**Theatrical**
- **Budget & Popularity dominate** revenue prediction; **runtime** exhibits plateaus.
- Genres & seasonality have **small but directional** effects.
- SHAP confirms: higher `log_budget`/`log_popularity` push predictions **up**.

**Streaming**
- **Audience signal rules**: `log_votes` > `runtime` > select genres/platforms.
- Confusion matrix shows a **conservative** stance on positives (precision > recall).
- SHAP: higher `log_votes` consistently increases top-quartile probability.

_All visuals are consolidated in the PDFs under `figures/`._

---

## ✨ Notable Findings (Like “Model’s Output”, but for insights)

- **Theatrical:** Bigger budgets + pre-release buzz (popularity) **pay off**. Runtime helps to a point, then **diminishing returns**.
- **Streaming:** Engagement metrics (vote counts) are **king**. A few genres/platform effects are present but more subtle.
- **Practical POV:** If you must bet early, a **precision-leaning** threshold is safer for greenlighting—fewer false positives.

---
