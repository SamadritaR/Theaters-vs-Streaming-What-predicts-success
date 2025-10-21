# 🎬 Theaters vs. Streaming — What Predicts Success?

> “Movies are a machine that generates empathy.” — Roger Ebert

I analyzed what truly drives movie success across **theatrical releases** and **streaming platforms**. I cleaned a 2018–2024 dataset, engineered features (log transforms, seasonality, genres/platforms), and built **interpretable** models so results are explainable—not black-box magic.

---

## 📦 What’s Here (clickable)
- 📒 **Notebook:** [`MoviePlatformAnalysisCode.ipynb`](MoviePlatformAnalysisCode.ipynb)
- 🖼️ **Figures (PDF):** [`Figures_Final_Report.pdf`](Figures_Final_Report.pdf) · [`Figures.pdf`](Figures.pdf)
- 🧾 **Report:** [`Final Project Report.pdf`](Final%20Project%20Report.pdf)
- 🖥️ **Presentation:** [`Theaters vs. Streaming What Predicts Success.pptx`](Theaters%20vs.%20Streaming%20What%20Predicts%20Success.pptx)
- 🗂️ **Cleaned Data:** [`movies_merged.csv`](movies_merged.csv)

> Note: Links with spaces are **URL-encoded** (`%20`). They’ll work as-is.

---

## 🧰 Libraries Used
`pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `jupyter`

---

## 🗃️ Data & Features (quick)
- **Scope:** movies 2018–2024  
- **Transforms:** `log_budget`, `log_popularity`, `log_votes`  
- **Signals:** `runtime`, one-hot **main_genre**, seasonal flags  
- **Platforms:** provider dummies (netflix, hulu, prime_video, disney_plus)  
- **Targets:**  
  - **Theatrical:** regression on `log(revenue)`  
  - **Streaming:** classification of **top-quartile** titles

---

## 🧠 Models

### Theatrical (Regression)
- Interpretable importance + partial dependence (see PDFs above).
- Headline: **budget & popularity dominate**; runtime shows **plateaus**; genre/season effects are smaller but directional.

### Streaming (Classification)
- Year-based split (e.g., test = 2021), threshold chosen for **precision**.
- Headline: **audience signal (`log_votes`) is king**; runtime and a few genres/platforms contribute.

---

## 📊 Evaluation Snapshot — Streaming (Test 2021)

**Confusion matrix:** TN=200, FP=21, FN=49, TP=50

| Metric | Value |
|---|---:|
| Accuracy | **78.1%** |
| Precision (Top-quartile) | **70.4%** |
| Recall (Top-quartile) | **50.5%** |
| F1-score | **58.8%** |

**Read:** The operating point favors **precision**—fewer false “hits,” at the cost of missing some sleepers.

<details>
<summary><strong>Confusion Matrix Table</strong></summary>

|               | Pred: Not Top | Pred: Top |
|---|---:|---:|
| **Actual: Not Top** | 200 | 21 |
| **Actual: Top**     | 49  | 50 |

</details>

---

## 🔎 What the Figures Say (see PDFs)
- **Theatrical:** `log_budget` and `log_popularity` push predictions **up**; runtime has diminishing returns.  
- **Streaming:** `log_votes` drives top-quartile probability; runtime + select genres/platforms matter; chosen threshold makes precision > recall.

---
